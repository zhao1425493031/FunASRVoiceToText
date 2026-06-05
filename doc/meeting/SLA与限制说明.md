# 多人会议 SLA 与限制说明（对外必读）

## 服务承诺范围

- **单麦克风**、2～8 人轮流对话场景下的**实时字幕**。
- 日语（`language: ja`）与中文（`language: zh`）由 `config.meeting.yaml` 配置；客户端 `start.language` 可覆盖。

## 延迟（UAT 实测后填入 v3验收清单）

| 指标 | 设计目标 | 实测 |
|------|----------|------|
| partial P95（CPU） | ≤ 10 s | 待本机 UAT |
| partial P95（CUDA） | ≤ 3 s | 待 GPU UAT |
| final P95 | ≤ 15 s | 待本机 UAT |

## 限制与免责

1. **同时多人抢话**：识别与 `speaker_id` 准确率下降，可能合并为一行字幕。
2. **speaker_id**：Pyannote 会话内聚类标签，**不**保证跨会话同一人同 ID。
3. **partial 可被 final 覆盖**：业务入库以 `final` 为准。
4. **重连**：新 WebSocket = 新会话（不复用 `session_id` 状态）。
5. **说话人分离（行业双通道，`meeting_spk_primary`）**：
   - **主通道（默认 `embedding`）**：每句 **final** 对去静音后的句音频做 campplus 在线聚类（动态人数，阈值 `meeting_spk_embedding_threshold`）。
   - **辅通道**：Pyannote 累积上下文时间轴（`pyannote_context_sec`）；仅当与句时间重叠 ≥ `meeting_pyannote_min_overlap_ms` 时采信，embedding 失败时作回退。
   - **融合模式** `fusion`：两路不一致时优先 embedding（实时字幕行业惯例）。
   - partial 显示沿用当前 embedding 说话人，避免 Pyannote 滞后导致全程话者1。
6. **device**：`auto` 在无 CUDA 环境回退 CPU；生产 GPU 请显式 `device: cuda` 并安装 CUDA 版 PyTorch。
7. **会话时长**：默认最长 7200 s，超限 `session_too_long`。
8. **SenseVoice 流式**：窗级重识别，partial 延迟与抖动可能高于 v2 Qwen 方案。
9. **环境噪声**：默认 `vad_energy_threshold` + 连续 2 块噪声门；浏览器降噪开启。极嘈杂环境仍可能误触，需物理拾音或继续调高阈值。
10. **新 WebSocket 会话**：服务端重置 Pyannote ring/segments 与 embedding 聚类，避免跨会话污染。

## 资源

- 建议并发：默认 `max_ws_connections: 1`（说话人状态按 WS 隔离；多路需 GPU 与独立算力）。
- v3 单进程加载 SenseVoice + FSMN-VAD + Pyannote；无 Docker Runtime 侧车。
