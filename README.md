# python_voicetotext — 多人会议实时字幕 v3

单麦、多人会议实时字幕（协议 v2：`speaker_id` + `seg_id`）。  
**栈**：FSMN-VAD + SenseVoiceSmall + Pyannote Community-1。  
配置：[`config.meeting.yaml`](config.meeting.yaml)，文档：[`doc/meeting/`](doc/meeting/)。

## 快速启动

```powershell
cd C:\python\workspace\python_voicetotext
pip install -r requirements-meeting.txt
pip install -e .

# 一次性配置 HF Token（无需每次设环境变量）
copy secrets.meeting.yaml.example secrets.meeting.yaml
# 编辑 secrets.meeting.yaml，填入 hf_token

python scripts/run_meeting.py
```

等待日志 **MeetingSenseVoiceEngine ready**（首次会下载模型，需联网）。

手机/PC：`https://<局域网IP>:8766/meeting?key=meeting-dev-7k9m2p4x`

## 语言

`config.meeting.yaml` 中 `language: ja` 或 `zh`；页面握手可覆盖。

## LLM 会议纪要（可选）

在 `config.meeting.yaml` 中配置 `llm` 段（主开关 `llm.enabled`）：

```yaml
llm:
  enabled: true
  api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  model: "qwen-plus"
  api_key: "sk-..."    # DashScope API Key
  meeting_output_dir: "out/meetings"
```

- 会议中**不会**调用 LLM；点击「終了」后 WebSocket 保持连接，页面显示「纪要生成中…」，随后展示 markdown 纪要。
- 产物：`out/meetings/{session_id}.json` 与 `{session_id}.summary.md`。
- 跳过纪要：WebSocket `end` 消息设 `skip_summary: true`。
- HTTP 重试：`POST /api/v1/meeting/summarize`（Header `X-API-Key`）。

详见 [LLM纪要方案](doc/meeting/LLM纪要方案.md)。

## 设备

| 环境 | `device` |
|------|----------|
| Windows 本地 | `cpu` |
| GPU 服务器 | `cuda` |

## HTTPS

```powershell
python scripts/generate_cert.py
# 或 mkcert，见 doc/meeting/模型与依赖.md
```

## 测试

```powershell
$env:VOICETOTEXT_SKIP_HF_CHECK = "1"
pytest tests/ -q
```

## 文档

- [v3方案与背景](doc/meeting/v3方案与背景.md)
- [架构与数据流](doc/meeting/架构与数据流.md)
- [v3验收清单](doc/meeting/v3验收清单.md)
- [LLM纪要方案](doc/meeting/LLM纪要方案.md)
- [LLM实施总结](doc/meeting/LLM实施总结.md)
