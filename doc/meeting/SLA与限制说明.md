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
5. **说话人分离（双通道）**：
   - **主通道**：Pyannote 累积上下文时间轴（`pyannote_context_sec`）与 ASR 时间戳合并。
   - **回退通道**：当实时 Pyannote 仅检出 1 个标签时，对 **final 句音频** 做 campplus 句级 embedding 聚类（动态人数，阈值 `meeting_spk_embedding_threshold`）。
   - 滑窗步长导致 Pyannote 标签相对 final 可能有数秒～十余秒延迟；句级回退用于缩短双人对话首句区分时间。
6. **device**：`auto` 在无 CUDA 环境回退 CPU；生产 GPU 请显式 `device: cuda` 并安装 CUDA 版 PyTorch。
7. **会话时长**：默认最长 7200 s，超限 `session_too_long`。
8. **SenseVoice 流式**：窗级重识别，partial 延迟与抖动可能高于 v2 Qwen 方案。
9. **新 WebSocket 会话**：服务端重置 Pyannote ring/segments 与 embedding 聚类，避免跨会话污染。

## 资源

- 建议并发：≤ `max_ws_connections`（默认 8）。
- v3 单进程加载 SenseVoice + FSMN-VAD + Pyannote；无 Docker Runtime 侧车。
