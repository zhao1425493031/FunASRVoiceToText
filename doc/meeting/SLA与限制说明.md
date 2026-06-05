# 多人会议 SLA 与限制说明（对外必读）

## 服务承诺范围

- **单麦克风**、2～8 人轮流对话场景下的**实时字幕**。
- 日语（`language: ja`）为生产默认；中文/自动语种按 Runtime SenseVoice 配置支持。

## 延迟（UAT 实测后填入验收清单）

| 指标 | 设计目标 | 实测 |
|------|----------|------|
| partial P95 | ≤ 8 s | 待本机 UAT |
| final P95 | ≤ 15 s | 待本机 UAT |

## 限制与免责

1. **同时多人抢话**：识别与 `speaker_id` 准确率下降，可能合并为一行字幕。
2. **speaker_id**：会话内聚类标签，**不**保证跨会话同一人同 ID。
3. **partial 可被 final 覆盖**：业务入库以 `final` 为准。
4. **重连**：新 WebSocket = 新会话（不复用 `session_id` 状态）。
5. **说话人分离**：Runtime 2pass 不直接输出 spk 时，网关使用 cam++/时间间隔启发式，见 [FunASR运行时调研.md](./FunASR运行时调研.md)。
6. **会话时长**：默认最长 7200 s，超限 `session_too_long`。

## 资源

- 建议并发：≤ `max_ws_connections`（默认 10）。
- 模型与单人服务共用 `models/` 目录，会议 Runtime 独立容器（10096）。
