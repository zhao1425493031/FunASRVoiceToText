# python_voicetotext — 多人会议实时字幕

单麦、多人日语对话，实时字幕（协议 v2，`speaker_id` + `seg_id`）。  
**Windows 纯 Python**（SenseVoice 进程内推理），无需 Docker。  
配置：[`config.meeting.yaml`](config.meeting.yaml)，文档：[`doc/meeting/`](doc/meeting/)。

## 快速启动

```powershell
cd C:\python\workspace\python_voicetotext
pip install -r requirements.txt
pip install -e .

python scripts/run_meeting.py
```

等待日志 **`Embedded ASR model ready`**（首次会下载模型，需联网）。

手机/PC 打开（**可只输入 IP，会自动带上 key**）：

`https://<你的局域网IP>:8766/meeting`

或完整地址：`https://<IP>:8766/meeting?key=meeting-dev-7k9m2p4x`

## 架构

| 组件 | 端口 |
|------|------|
| 会议网关 + SenseVoice + 说话人 (cam++) | 8766 |

## HTTPS（手机麦克风）

```powershell
mkcert -install
mkcert -key-file certs\key.pem -cert-file certs\cert.pem localhost 127.0.0.1 <你的局域网IP>
```

`config.meeting.yaml` 中已配置 `ssl_certfile` / `ssl_keyfile`。

## 测试

```powershell
python -m pytest tests/ -q
```
