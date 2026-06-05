# python_voicetotext — 多人会议实时字幕 v2

单麦、多人会议实时字幕（协议 v2：`speaker_id` + `seg_id`）。  
**栈**：FSMN-VAD + Qwen3-ASR（0.6B partial / 1.7B final）+ Pyannote Community-1。  
配置：[`config.meeting.yaml`](config.meeting.yaml)，文档：[`doc/meeting/`](doc/meeting/)。

## 快速启动

```powershell
cd C:\python\workspace\python_voicetotext
pip install -r requirements-meeting.txt
pip install -e .
# 若 Qwen3 报错 No module named 'qwen_asr'：pip install qwen-asr

# 一次性配置 HF Token（无需每次设环境变量）
copy secrets.meeting.yaml.example secrets.meeting.yaml
# 编辑 secrets.meeting.yaml，填入 hf_token

python scripts/run_meeting.py
```

等待日志 **MeetingQwenEngine ready**（首次会下载模型，需联网）。

手机/PC：`https://<局域网IP>:8766/meeting?key=meeting-dev-7k9m2p4x`

## 语言

`config.meeting.yaml` 中 `language: ja` 或 `zh`；页面握手可覆盖。

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

- [v2方案与背景](doc/meeting/v2方案与背景.md)
- [架构与数据流](doc/meeting/架构与数据流.md)
- [v2验收清单](doc/meeting/v2验收清单.md)
