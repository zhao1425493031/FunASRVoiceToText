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

- `{job_id}.json` — 结构化数据（时间戳 + 说话人 + 文本）
- `{job_id}.md` — 可读文本，便于复制
- `{job_id}.srt` — 字幕格式

可选参数：

```powershell
# 指定输出目录
python scripts/run_batch.py D:\audio\meeting.wav -o D:\transcripts

# 指定配置文件
python scripts/run_batch.py meeting.wav -c config.yaml -o out/
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
