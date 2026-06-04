# python_voicetotext — 企业日语实时语音识别

局域网实时**日语**语音识别（SenseVoice + FunASR）。PC/手机浏览器访问 `https://<电脑IP>:8765`，麦克风边说边出字。

**两阶段架构（均已实现）：**

| 阶段 | `asr_backend` | 说明 |
|------|---------------|------|
| 一 | `sensevoice`（默认） | 进程内 `iic/SenseVoiceSmall`，2s 窗口伪流式；**停止时整段定稿** + `ct-punc` 兜底 |
| 二 | `runtime` | FunASR Runtime 2pass 侧车（Docker），网关 WebSocket 转发 |

本文档**合并了安装说明与按顺序操作手顺**。企业配置见 [`config.enterprise.yaml`](config.enterprise.yaml)；公开发布定稿见 [`doc/企业公开发布实施方案.md`](doc/企业公开发布实施方案.md)；技术方案见 [`doc/企业日语生产方案.md`](doc/企业日语生产方案.md)。

**关键配置（公开发布）：** `session_pcm_max_seconds: 600`（单次录音上限）、`punc_model: ct-punc`（定稿标点兜底）、`auto_finalize_on_silence: false`（仅 `end` 定稿）。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | 3.10+ |
| 系统 | Windows 10/11（开发）或 Linux（部署） |
| 网络 | 首次启动需联网下载模型（约 1.5～2GB） |
| 浏览器 | PC：Chrome / Edge；手机：Chrome / Safari（**必须 HTTPS**） |
| 可选 | NVIDIA GPU + CUDA 版 PyTorch，`config.yaml` 中 `device: cuda:0` |
| 阶段二 | Docker（FunASR Runtime 侧车） |

**准备勾选：**

- [ ] 电脑已联网
- [ ] 已安装 Python 3.10+
- [ ] 手机与电脑将使用同一 WiFi

项目路径示例：`C:\python\workspace\python_voicetotext`

---

## 安装手顺（按顺序执行）

### 步骤 1：进入项目目录

```powershell
cd C:\python\workspace\python_voicetotext
```

Linux：

```bash
cd /path/to/python_voicetotext
```

---

### 步骤 2：安装 PyTorch（必须先于 funasr）

[PyTorch 官网](https://pytorch.org/get-started/locally/) 按环境选择，或 CPU：

```powershell
pip install torch torchaudio
```

---

### 步骤 3：安装项目依赖

```powershell
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` 已包含 `funasr==1.3.9`、`python-multipart`（批处理上传 `/api/transcribe` 必需）。若单独报错缺少 multipart：

```powershell
pip install python-multipart
```

**若失败：Windows 路径过长**（`pip install` 报 `modelscope\...` FileNotFoundException）

1. **管理员** PowerShell：

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

2. **重启电脑**
3. 清理并重装：

```powershell
pip uninstall modelscope funasr -y
pip cache purge
pip install -r requirements.txt
pip install -e .
```

**替代：** 将 Python 安装到短路径（如 `C:\Python311`），项目也放在较短目录下。

- [ ] `pip install` 无报错

---

### 步骤 4：查看电脑局域网 IP

```powershell
ipconfig
```

记录 WLAN/以太网 **IPv4**，例如 `192.168.0.118`（下文 `<你的IP>`）。

Linux：`hostname -I`

- [ ] 已记录 `<你的IP>`

---

### 步骤 5：安装 mkcert 并生成本地 CA

iPhone 在 `http://局域网IP` 下无法使用麦克风，必须使用 **`https://<你的IP>:8765`**。

#### 5.1 获取 mkcert（任选其一）

| 方式 | 操作 |
|------|------|
| A 直接下载 | [mkcert Releases](https://github.com/FiloSottile/mkcert/releases) → `mkcert-*-windows-amd64.exe` → 如 `C:\Tool\mkcert\mkcert.exe` |
| B winget | `winget install FiloSottile.mkcert`（新开 PowerShell） |
| C Chocolatey | 需先有 choco：`choco install mkcert`（无 `choco` 命令则用 A 或 B） |

#### 5.2 安装本地 CA（只需一次）

```powershell
C:\Tool\mkcert\mkcert.exe -install
```

- 出现 `installed in the system trust store` 即成功
- `keytool` / Java `cacerts` 报错 **可忽略**

- [ ] mkcert CA 已安装

---

### 步骤 6：生成 HTTPS 证书

```powershell
cd C:\python\workspace\python_voicetotext
C:\Tool\mkcert\mkcert.exe -key-file certs\key.pem -cert-file certs\cert.pem localhost 127.0.0.1 <你的IP>
```

示例：

```powershell
C:\Tool\mkcert\mkcert.exe -key-file certs\key.pem -cert-file certs\cert.pem localhost 127.0.0.1 192.168.0.118
```

**备选（需已安装 openssl）：**

```powershell
python scripts/generate_cert.py --ip <你的IP>
```

**确认文件：**

```powershell
dir certs\cert.pem
dir certs\key.pem
```

- [ ] `cert.pem` 与 `key.pem` 均存在

---

### 步骤 7：确认配置文件

#### 7.1 开发 / 阶段一（默认）

[`config.yaml`](config.yaml)：

```yaml
asr_backend: "sensevoice"
language: "ja"
asr_model: "iic/SenseVoiceSmall"
punc_model: ""
stream_window_ms: 2000
ssl_certfile: "certs/cert.pem"
ssl_keyfile: "certs/key.pem"
api_key: ""   # 空表示不鉴权
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `asr_backend` | `sensevoice` | `sensevoice` / `runtime` / `paraformer`（仅回归） |
| `language` | `ja` | SenseVoice 识别语言 |
| `asr_model` | `iic/SenseVoiceSmall` | 日语生产模型 |
| `stream_window_ms` | `2000` | 伪流式累积窗口；约满 2s 音频后才出 partial |
| `api_key` | 空 | 非空时启用 HTTP/WS 鉴权 |
| `max_ws_connections` | `20` | WebSocket 连接上限 |
| `device` | `auto` | `auto` / `cpu` / `cuda:0` |
| `vad_silence_ms` | `800` | 静音判句（毫秒） |
| `vad_energy_threshold` | `0.02` | 窗级能量门控，低于此不推理 |
| `min_partial_chars` | `2` | 有效字符过少则不下发 partial |
| `ssl_certfile` / `ssl_keyfile` | `certs/*.pem` | 手机麦克风必填 |

#### 7.2 企业生产（API Key）

[`config.enterprise.yaml`](config.enterprise.yaml)：`api_key` 改为生产密钥，启动时指定：

```powershell
python scripts/run_server.py --config config.enterprise.yaml
```

浏览器访问须带 Key：`https://127.0.0.1:8765/?key=<与配置一致的 api_key>`

部署前请修改 `change-me-in-production` 为强密码。

#### 7.3 阶段二（Runtime 侧车）

[`config.runtime.yaml`](config.runtime.yaml)：`asr_backend: runtime`，网关不加载 SenseVoice，仅转发至 `runtime_host:runtime_port`（默认 `127.0.0.1:10095`）。见下文「Runtime 部署手顺」。

暂不用 HTTPS 可置空 `ssl_certfile` / `ssl_keyfile`（手机麦克风仍可能失败）。

- [ ] 端口、证书、backend 与预期一致

---

### 步骤 8：放行防火墙

**Windows（管理员）：**

```powershell
New-NetFirewallRule -DisplayName "VoiceToText" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
```

**Linux：** `sudo ufw allow 8765/tcp`（Runtime 另放行 `10095/tcp`）

- [ ] 防火墙已放行

---

### 步骤 9：启动服务（首次会下载模型）

```powershell
cd C:\python\workspace\python_voicetotext
python scripts/run_server.py
```

企业生产：

```powershell
python scripts/run_server.py --config config.enterprise.yaml
```

等待出现：

```text
ASR backend: sensevoice language=ja
https://0.0.0.0:8765
Preloading ASR models (sensevoice)...
Server ready ...
Uvicorn running on https://0.0.0.0:8765
```

| 控制台提示 | 含义 |
|------------|------|
| `https://...` | 证书已加载 |
| `http://...` + WARNING | 无证书，回退 HTTP；手机麦克风可能失败 |
| `Preloading ASR models` | 加载/下载 SenseVoice，请等待 |
| `Server ready` | 可访问 |
| `Runtime gateway mode` | 使用 `config.runtime.yaml`，未加载进程内模型 |

- 首次下载约 **1.5～2GB**，可能 **30 分钟～1 小时+**，勿关窗口
- 缓存：`%USERPROFILE%\.cache\modelscope\`

- [ ] 显示 `https://` 且 `Server ready`
- [ ] 保持窗口运行

---

### 步骤 10：启动后健康检查

模型加载完成前 `/ready` 可能为 503，加载完成后应为 200。

**PowerShell：**

```powershell
Invoke-WebRequest -Uri https://127.0.0.1:8765/health -SkipCertificateCheck
Invoke-WebRequest -Uri https://127.0.0.1:8765/ready -SkipCertificateCheck
```

**curl（Linux / Git Bash）：**

```bash
curl -k https://127.0.0.1:8765/health
curl -k https://127.0.0.1:8765/ready
```

| 端点 | 200 含义 |
|------|----------|
| `/health` | 进程存活 |
| `/ready` | SenseVoice 已加载，或 Runtime 可达 |

- [ ] `/ready` 返回 200（模型就绪后）

---

### 步骤 11：PC 浏览器验证（日语）

1. 打开 `https://127.0.0.1:8765`（企业模式：`https://127.0.0.1:8765/?key=<api_key>`）
2. **开始识别** → 允许麦克风 → **用日语说话**
3. 约 **2～5 秒**内「识别中」出现 partial（受 `stream_window_ms=2000` 影响）
4. 停顿约 0.8s（`vad_silence_ms`）或点 **停止识别** → 「已确认」出现 final

- [ ] PC 端日语 partial / final 正常

---

### 步骤 12：iPhone 信任 mkcert 根证书

```powershell
C:\Tool\mkcert\mkcert.exe -CAROOT
```

1. 将文件夹内 **`rootCA.pem`** 发到 iPhone 并安装
2. **设置 → 通用 → 关于本机 → 证书信任设置** → 启用完全信任

- [ ] iPhone 已信任根证书

---

### 步骤 13：手机浏览器访问

1. 手机与电脑 **同一 WiFi**
2. 打开 `https://<你的IP>:8765`（企业模式加 `?key=`）
3. **开始识别** → 允许麦克风 → 日语说话

- [ ] 手机能实时出字

---

### 步骤 14（可选）：批处理 API（WAV）

上传 **16 kHz、单声道、16-bit PCM** 的 `.wav`，返回 JSON 全文。

**无鉴权（`config.yaml`）：**

```powershell
curl -k -X POST "https://127.0.0.1:8765/api/transcribe" -F "file=@sample.wav"
```

**有鉴权（`config.enterprise.yaml`）：**

```powershell
curl -k -X POST "https://127.0.0.1:8765/api/transcribe" `
  -H "X-API-Key: change-me-in-production" `
  -F "file=@sample.wav"
```

响应示例：`{"text":"…","language":"ja","backend":"sensevoice"}`

`asr_backend=runtime` 时批处理接口返回 501（请用 Web 实时或 Runtime 原生接口）。

---

### 步骤 15（可选）：CLI 本机测试

```powershell
cd C:\python\workspace\python_voicetotext
$env:PYTHONPATH="src"
python -m voicetotext.cli mic
python -m voicetotext.cli wav --path your.wav
```

指定配置：

```powershell
python -m voicetotext.cli mic --config config.enterprise.yaml
```

---

### 步骤 16（可选）：单元测试

```powershell
cd C:\python\workspace\python_voicetotext
$env:PYTHONPATH="src"
python -m pytest tests/ -q -p no:flask
```

预期：`24 passed`

---

## 完成检查清单

- [ ] 依赖安装完成（含 `python-multipart`）
- [ ] HTTPS 证书已生成
- [ ] 服务 `https://` 启动且 `Server ready`
- [ ] `/ready` 为 200
- [ ] PC `https://127.0.0.1:8765` 日语识别正常
- [ ] iPhone `https://<你的IP>:8765` 可识别（已信任根证书）

---

## Runtime 部署手顺（阶段二）

适用：`asr_backend=runtime`，降低延迟、官方 Runtime 生产路径。

### R-1：准备模型与 env

1. 按 [FunASR Runtime 文档](https://github.com/modelscope/FunASR/tree/main/runtime) 准备 **日语 2pass** 模型，放到 `deploy/models`（或自定义挂载目录）。
2. 复制环境文件并编辑模型路径：

```powershell
copy deploy\runtime.env.example deploy\runtime.env
notepad deploy\runtime.env
```

修改 `RUNTIME_MODEL_PATH`、`RUNTIME_ONLINE_MODEL_PATH` 等为日语模型目录。

### R-2：启动 Runtime（Docker）

```powershell
cd C:\python\workspace\python_voicetotext\deploy
docker compose --env-file runtime.env up -d funasr-runtime
```

确认端口 `10095` 监听。防火墙需放行 `10095`（若跨机访问 Runtime）。

### R-3：启动网关

```powershell
cd C:\python\workspace\python_voicetotext
python scripts/run_server.py --config config.runtime.yaml
```

控制台应出现 `Runtime gateway mode`，且 **不会** 长时间下载 SenseVoice。

### R-4：验证

```powershell
curl -k http://127.0.0.1:8765/ready
```

- Runtime 正常：200，`"backend":"runtime"`
- Runtime 未启动：503，`FunASR Runtime unreachable`

### R-5：Web 验收

PC/手机访问方式与阶段一相同（HTTPS + 可选 `?key=`）。协议不变；网关将 Runtime 的 `2pass-online` / `2pass-offline` 映射为 partial / final。

**Linux 一键（WSL/服务器）：** `bash scripts/start_runtime_stack.sh`  
**Windows：** `.\scripts\start_runtime_stack.ps1`（先 Docker Runtime，再本地 gateway）

可将 Windows 的 `%USERPROFILE%\.cache\modelscope` 拷到 Linux 同路径，避免 SenseVoice 阶段重复下载。

---

## Linux 服务器差异（阶段一）

| 项目 | Linux 命令 |
|------|------------|
| 依赖 | `pip3 install torch torchaudio` → `pip3 install -r requirements.txt` → `pip3 install -e .` |
| 证书 | `mkcert -install` → `mkcert -key-file certs/key.pem -cert-file certs/cert.pem localhost 127.0.0.1 <IP>` |
| 防火墙 | `sudo ufw allow 8765/tcp` |
| IP | `hostname -I` |
| 启动 | `python3 scripts/run_server.py` |

阶段二见上一节 Runtime 手顺；Compose 启动顺序：**先 `funasr-runtime`，再 gateway**。

---

## 故障速查

| 现象 | 处理 |
|------|------|
| `Form data requires python-multipart` | `pip install python-multipart` 或重装 `requirements.txt` |
| 步骤 3 pip 路径过长 | 启用长路径并重启，见步骤 3 |
| `choco` 无法识别 | 用 mkcert 直接下载或 winget |
| `openssl not found` | 用 mkcert，见步骤 5～6 |
| `mkcert -install` keytool 报错 | 可忽略，继续步骤 6 |
| 步骤 9 只有 http | 检查 `certs/*.pem` 是否存在 |
| 手机 `getUserMedia` 报错 | 必须 `https://` + 步骤 12 |
| WebSocket `unauthorized` | 企业配置须 URL `?key=` 或 start 带 `api_key` |
| WebSocket 连上即断 | 多为麦克风/证书问题，完成步骤 12～13 |
| 有页无 partial | 等满约 2s 窗口；确认 `Server ready` 与 `/ready` 200 |
| PC 报错 `list / int` | 确认 `asr_backend=sensevoice`，勿对 SenseVoice 用 Paraformer 的 vad 捆绑 |
| `/ready` 503（runtime） | 先 `docker compose up funasr-runtime`，查 `runtime_host/port` |
| 能开页无文字 | 查 `logs/app.log`、防火墙 |

---

## 日志与其他文档

- 日志：`logs/app.log`
- [`doc/平台兼容性.md`](doc/平台兼容性.md) — 浏览器说明
- [`doc/AI_Chat集成与内网部署.md`](doc/AI_Chat集成与内网部署.md) — **第三方 / 手机 H5 Chat 接语音、语言配置、内网 HTTP/HTTPS**
- [`doc/企业日语生产方案.md`](doc/企业日语生产方案.md) — 两阶段拓扑与切换
- [`doc/方案与实施说明.md`](doc/方案与实施说明.md) — 实施说明
- [`doc/WebSocket协议.md`](doc/WebSocket协议.md) — 接口协议（含 `api_key`、Runtime 映射）
- [`doc/阶段一SenseVoice验收.md`](doc/阶段一SenseVoice验收.md) / [`doc/阶段二Runtime验收.md`](doc/阶段二Runtime验收.md)

## 项目结构

```
config.yaml / config.enterprise.yaml / config.runtime.yaml
deploy/docker-compose.yml
certs/
scripts/run_server.py
src/voicetotext/asr/    # SenseVoice / Paraformer / Runtime 客户端
src/voicetotext/server/ # app, ws_protocol, auth, transcribe
web/
requirements.txt
```
