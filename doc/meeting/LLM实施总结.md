# 会议 LLM 纪要实施总结

## 改动清单

| 模块 | 文件 | 改动 |
|------|------|------|
| 配置 | `src/voicetotext/config.py` | LLM 全字段解析、`_validate_llm`、`require_llm_api_key`、`llm_ready` |
| 配置 | `config.meeting.yaml` | 补全 `meeting_output_dir`、`meeting_ws_wait_sec` |
| 依赖 | `requirements-meeting.txt` | 追加 `httpx` |
| LLM 栈 | `src/voicetotext/llm/*` | schemas/client/parse/prompts/summary_service/providers/factory |
| 会议专用 | `meeting_transcript_buffer.py`、`meeting_export.py` | final 累积 + json/md 落盘 |
| WS | `meeting_ws_protocol.py` | buffer 拦截 final、end/cleanup 异步纪要、新消息类型 |
| HTTP | `meeting_app.py` | summarize/session GET、`/ready` detail.llm |
| 启动 | `scripts/run_meeting.py` | `require_llm_api_key` |
| 前端 | `web/meeting.html/js`、`style.css` | 纪要面板、end 后等待推送、复制/下载 md |
| 测试 | `tests/test_meeting_llm_*.py` 等 | 11 个新/更新测试文件 |
| 夹具 | `tests/fixtures/config_meeting_llm_test.yaml` | mock HTTP 用假 key |

## 已有 vs 新建

| 类别 | 路径 |
|------|------|
| 复用 | `meeting_ws_protocol` 框架、`meeting_stream_session`、鉴权、`config` 加载 |
| 新建 | 完整 `llm/` 栈、buffer、export、HTTP 端点、前端纪要面板、测试夹具 |

## Todo 完成状态表（T01–T08）

| ID | 任务 | 状态 | 证据 |
|----|------|------|------|
| T01 | 配置层 + httpx + require_llm_api_key | pass | `test_meeting_llm_config.py`、`test_meeting_config.py` |
| T02 | LLM 全栈移植 | pass | `test_meeting_llm_summarize.py`、`test_meeting_llm_client.py` |
| T03 | buffer + export | pass | `test_meeting_transcript_buffer.py`、`test_meeting_llm_export.py` |
| T04 | WS 集成 | pass | `test_meeting_ws_summary.py`、`test_meeting_ws_skip_summary.py`、`test_meeting_llm_disconnect.py` |
| T05 | HTTP API | pass | `test_meeting_llm_api.py`、`test_meeting_app_ready.py` |
| T06 | 前端纪要面板 | pass | `web/meeting.html`、`meeting.js` 已实现 |
| T07 | 测试全绿 | pass | `pytest tests/ -q` → **128 passed** |
| T08 | 文档 + UAT | pass* | 本文档 + 协议/架构/验收清单更新；*人工 UAT 见下 |

## UAT 结果

| 项 | 结果 | 说明 |
|----|------|------|
| 自动化验收 M-A1–M-A12 | pass | pytest 128 项全绿，无真实 LLM API 调用 |
| 人工 UAT（真实录音 → end → 页面纪要） | 待执行 | 需配置有效 `llm.api_key` 与 GPU/CPU ASR 环境后按 [v3验收清单](v3验收清单.md) F10–F19 执行 |

人工 UAT 步骤：

1. `python scripts/run_meeting.py`，确认 `/ready` 中 `detail.llm.ready=true`。
2. 打开 `/meeting?key=...`，开始字幕，说话 1–2 分钟。
3. 点击「終了」，等待「纪要生成中…」→ 纪要面板显示 markdown。
4. 检查 `out/meetings/{session_id}.json` 与 `.summary.md` 已生成。

## pytest 命令

```powershell
$env:VOICETOTEXT_SKIP_HF_CHECK = "1"
pytest tests/ -q
```
