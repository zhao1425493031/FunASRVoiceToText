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
5. **说话人分离**：`speaker_id` 来自 Pyannote 滑窗时间轴与 ASR 时间戳合并；滑窗步长导致标签相对 final 可能有 ≤15s 延迟。
6. **会话时长**：默认最长 7200 s，超限 `session_too_long`。
7. **SenseVoice 流式**：窗级重识别，partial 延迟与抖动可能高于 v2 Qwen 方案。

## 资源

- 建议并发：≤ `max_ws_connections`（默认 8）。
- v3 单进程加载 SenseVoice + FSMN-VAD + Pyannote；无 Docker Runtime 侧车。
