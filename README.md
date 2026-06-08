# 录音文件转写

SenseVoice + Pyannote：日文/中文转写、说话人分离、时间戳。批处理采用 **Diar-First**（先说话人分段，再逐段转写）。

## 安装

```powershell
pip install -r requirements.txt
pip install -e ".[dev]"
copy secrets.meeting.yaml.example secrets.meeting.yaml
# 编辑 secrets.meeting.yaml 填入 hf_token
```

## 使用（命令行，无需启动服务）

```powershell
python scripts/run_batch.py 录音.wav -o out/
```

完成后在 `out/` 目录查看：

- `{job_id}.json` — 结构化数据（时间戳 + 说话人 + 文本 + 会议纪要）
- `{job_id}.md` — 可读文本，便于复制
- `{job_id}.srt` — 字幕格式
- `{job_id}.summary.md` — 会议纪要（`llm.enabled: true` 时生成）

可选参数：

```powershell
# 指定输出目录
python scripts/run_batch.py D:\audio\meeting.wav -o D:\transcripts

# 指定配置文件
python scripts/run_batch.py meeting.wav -c config.yaml -o out/

# 跳过 LLM 纪要（即使 llm.enabled 为 true）
python scripts/run_batch.py meeting.wav -o out/ --no-summary
```

## 配置

`config.yaml` — 语言（`ja`/`zh`）、设备等。说话人分离需配置 `secrets.meeting.yaml` 中的 `hf_token`。

默认按 **2 人会议** 优化（`pyannote_min_speakers: 2`、`pyannote_max_speakers: 2`）。更多说话人时请修改 `config.yaml` 中对应字段。

| 字段 | 默认 | 说明 |
|------|------|------|
| `batch_diar_merge_gap_ms` | 500 | 同说话人相邻段合并间隙 |
| `batch_diar_min_segment_ms` | 300 | 过短 diar 段吸收阈值 |
| `batch_asr_parallel_workers` | 4 | 并行 ASR 线程数 |

输出 JSON 的 `meta.pipeline` 为 `diar_first` 表示新管线。

### LLM 会议纪要

在 `config.yaml` 的 `llm` 段配置 DashScope（Qwen OpenAI 兼容模式）：

```yaml
llm:
  enabled: true
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-plus"
  api_key: "sk-..."          # API 密钥写在这里
  temperature: 0.3
  max_tokens: 8096
  timeout: 120
  on_failure: warn           # LLM 失败时不阻断转写
  output_summary_md: true
```

| 字段 | 默认 | 说明 |
|------|------|------|
| `on_failure` | warn | `warn`=转写成功 summary 可为 null；`fail`=整 job 失败 |
| `max_input_chars` | 120000 | 超长会议触发 map-reduce 分块 |
| `output_summary_md` | true | 额外写 `{job_id}.summary.md` |

独立摘要 API：`POST /api/v1/summarize`（传 `segments` 或仅 `job_id`）。
