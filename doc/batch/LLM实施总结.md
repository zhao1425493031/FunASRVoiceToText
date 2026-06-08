# LLM 会议纪要实施总结

## Todo 完成状态表

| ID | 任务 | 结果 | 证据 |
|----|------|------|------|
| T01 | 扩展 config.py + config.yaml + httpx | pass | `test_llm_config.py`, `test_config.py` |
| T02 | 扩展 schemas.py（MeetingSummary） | pass | `test_llm_parse.py` |
| T03 | transcript_format.py + prompts.py | pass | `test_llm_transcript_format.py` |
| T04 | client.py + parse.py | pass | `test_llm_client.py`, `test_llm_parse.py` |
| T05 | providers + factory | pass | `test_llm_provider.py` |
| T06 | summary_service.py（map-reduce + repair） | pass | `test_llm_map_reduce.py` |
| T07 | batch_pipeline 集成 + summary.md | pass | `test_llm_pipeline.py` |
| T08 | run_batch.py + batch_app.py | pass | `test_llm_api.py`, `test_llm_stub.py` |
| T09 | 全部测试 | pass | pytest **104 passed** |
| T10 | 文档 + UAT | pass | 本文档 + `out/uat_summary_result.json` |

## 已有可用代码（复用/扩展）

| 模块 | 路径 | 方式 |
|------|------|------|
| LLM Schema 骨架 | `src/voicetotext/llm/schemas.py` | 扩展 MeetingSummary |
| Provider Protocol + Stub | `src/voicetotext/llm/summary.py` | 保留接口 |
| BatchTranscript | `src/voicetotext/asr/batch_pipeline.py` | 改 summary 类型并接线 |
| 配置加载 | `src/voicetotext/config.py` | 扩展 _parse_llm |
| HTTP 鉴权 | `src/voicetotext/server/auth.py` | 不变 |
| CLI 入口 | `scripts/run_batch.py` | 追加 --no-summary |

## 新开发代码

| 文件 | 职责 |
|------|------|
| `src/voicetotext/llm/transcript_format.py` | segments 格式化与分块 |
| `src/voicetotext/llm/prompts.py` | ja/zh Prompt 模板 |
| `src/voicetotext/llm/client.py` | httpx Chat Completions 客户端 |
| `src/voicetotext/llm/parse.py` | JSON 解析与兜底 |
| `src/voicetotext/llm/summary_service.py` | 单次/map-reduce 编排 |
| `src/voicetotext/llm/providers/openai_compatible.py` | DashScope 主路径 |
| `src/voicetotext/llm/providers/azure.py` | Azure OpenAI |
| `src/voicetotext/llm/factory.py` | Provider 工厂 |
| `tests/fixtures/config_llm_*.yaml` | 测试配置 |
| `tests/test_llm_*.py`（8 个） | 单元/集成测试 |

## 修改文件清单

- `src/voicetotext/config.py` — AppConfig LLM 全字段、校验、require_llm_api_key
- `config.yaml` — llm 段补全
- `requirements.txt` / `pyproject.toml` — httpx
- `src/voicetotext/llm/schemas.py` — MeetingSummary、SummaryResponse
- `src/voicetotext/llm/summary.py` / `__init__.py`
- `src/voicetotext/asr/batch_pipeline.py` — LLM 集成、export summary.md
- `scripts/run_batch.py` — --no-summary
- `src/voicetotext/server/batch_app.py` — summarize API、job summary、ready llm
- `tests/test_config.py`, `test_batch_pipeline.py`, `test_llm_stub.py`

## 配置示例

```yaml
llm:
  enabled: true
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-plus"
  api_key: "sk-..."
  temperature: 0.3
  max_tokens: 8096
  timeout: 120
  max_input_chars: 120000
  chunk_chars: 30000
  retry_max: 2
  retry_backoff_sec: 2
  on_failure: warn
  output_summary_md: true
```

## UAT 结果

对已有转写 `out/125a4f41-1718-4d57-86fb-e85388861a55.json` 调用真实 DashScope API：

| 验收项 | 结果 |
|--------|------|
| status | ok |
| title | 案件進捗確認会議 |
| 金曜夕方まで展開 | action_items[0] に含む |
| レビュー会議火曜午後 | decisions + action_items に含む |
| latency | ~29s（qwen-plus 单次） |
| chunks | 1（短会议无需 map-reduce） |

完整结果见 `out/uat_summary_result.json`。

## 验收标准 A1–A12

| ID | 要求 | 结果 |
|----|------|------|
| A1 | load_config 读取 llm 全字段 | pass |
| A2 | enabled 时 summary 非 null | pass（UAT） |
| A3 | 结构化字段完整 | pass（UAT） |
| A4 | 关键点提取正确 | pass（UAT） |
| A5 | 生成 summary.md | pass（test_llm_pipeline） |
| A6 | --no-summary 跳过 | pass（test_llm_pipeline） |
| A7 | on_failure=warn 不阻断转写 | pass（test_llm_pipeline） |
| A8 | on_failure=fail 抛错 | pass（test_llm_pipeline） |
| A9 | POST /summarize segments 200 | pass（test_llm_api） |
| A10 | POST /summarize job_id 读盘 | pass（test_llm_api） |
| A11 | job status 含 summary | pass（batch_app 实现） |
| A12 | pytest 全通过 | pass（104 passed） |
