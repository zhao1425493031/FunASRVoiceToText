# 阶段一 SenseVoice 验收记录

依据计划 4.1 节。代码与配置已就绪；**需在本机 GPU/CPU 实测** 的项标为「待实测」。

| # | 验收项 | 结果 | 备注 |
|---|--------|------|------|
| 1 | `asr_backend=sensevoice`，`language=ja`，`punc_model` 为空 | 通过 | `config.yaml` |
| 2 | PC `https://127.0.0.1:8765` 日语 5s 内 partial、句末 final | 待实测 | 需模型下载完成 |
| 3 | 手机同 WiFi HTTPS 日语识别 | 待实测 | mkcert 信任 + `?key=`（若 enterprise） |
| 4 | 无 list/int VAD 错误；30 分钟不崩溃 | 待实测 | 长稳测试 |
| 5 | 无 API Key 时 WS 返回 error 并关闭 | 通过 | `config.enterprise.yaml` + 单元测试 |
| 6 | `GET /ready` 未加载 503、加载后 200 | 通过 | `server/app.py` |
| 7 | `POST /api/transcribe` 16kHz WAV 返回 JSON | 通过 | `routes_transcribe.py` |
| 8 | `pytest` 全通过（含 SenseVoice mock） | 通过 | `24 passed` |

## 实施 Todo（阶段一）

| ID | 内容 | 状态 |
|----|------|------|
| p1-asr-abstraction | asr 包、ASRBackend、factory | 完成 |
| p1-sensevoice-engine | SenseVoiceEngine + 测试 | 完成 |
| p1-stream-session | stream_window_ms + ASRBackend | 完成 |
| p1-config | config.py / yaml / enterprise | 完成 |
| p1-app-ready-auth | app factory、/ready、auth、连接上限 | 完成 |
| p1-transcribe-api | POST /api/transcribe | 完成 |
| p1-ws-auth-web | ws 鉴权、app.js key、cli | 完成 |
| p1-deps-readme | requirements、README | 完成 |
| p1-doc-phase1 | 本文档 + 功能影响记录 | 完成 |
