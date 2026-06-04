# 阶段一 SenseVoice 验收记录

依据公开发布定稿方案。代码与配置已就绪；**需在本机 GPU/CPU 实测** 的项标为「待实测」。

| # | 验收项 | 结果 | 备注 |
|---|--------|------|------|
| 1 | `asr_backend=sensevoice`，`punc_model: ct-punc`（兜底） | 通过 | `config.enterprise.yaml` |
| 2 | 连续说 1～3 分钟，停止后 `final` **有标点** | 待实测 | 整段 `use_itn=True` + ct-punc 兜底 |
| 3 | 识别中 `partial` **无句中乱句号** | 待实测 | 窗级 `use_itn=False` + `merge_utterance_segments` |
| 4 | `session_pcm_max_seconds` 默认 600，超限 `session_too_long` | 通过（单测）/ 待实测 | 可临时改为 5s 验 WS |
| 5 | PC `https://127.0.0.1:8765` partial/final | 待实测 | |
| 6 | 手机同 WiFi HTTPS | 待实测 | mkcert + `?key=`（enterprise） |
| 7 | 无 VAD list/int 错误；长稳 | 待实测 | |
| 8 | API Key 鉴权 WS | 通过 | 单元测试 |
| 9 | `GET /ready` | 通过 | |
| 10 | `POST /api/transcribe` 与 WS final 同策略 | 通过 | `finalize_utterance` |
| 11 | `pytest` 全通过 | 通过 | **41 passed** |

## 实施 Todo（公开发布 e1–e9）

| ID | 内容 | 状态 |
|----|------|------|
| e1–e8 | 见 [`企业公开发布实施方案.md`](企业公开发布实施方案.md) | 完成 |
| e9-uat-sensevoice | 上表 #2–#6 实测勾选 | 待实测 |
