# 阶段二 Runtime 2pass 验收记录

依据计划 4.2 节。网关与客户端代码已就绪；**需启动 FunASR Runtime Docker** 后实测。

| # | 验收项 | 结果 | 备注 |
|---|--------|------|------|
| 1 | `deploy/docker-compose.yml` 启动 Runtime | 通过 | 镜像与命令见 compose |
| 2 | `asr_backend=runtime` 不加载进程内 SenseVoice | 通过 | `app.py` lifespan 分支 |
| 3 | `2pass-online`→partial，`2pass-offline`→final | 通过 | `RuntimeWSClient.map_runtime_to_client` |
| 4 | PC/手机 runtime 模式日语识别 | 待实测 | 需日语 2pass 模型目录 |
| 5 | Runtime 不可达 `/ready` 503 | 通过 | `RuntimeGatewayEngine.check_ready` |
| 6 | Linux 文档含启动顺序 | 通过 | README + 本文档 |

## 实施 Todo（阶段二）

| ID | 内容 | 状态 |
|----|------|------|
| p2-runtime-client | runtime_client.py + 测试 | 完成 |
| p2-ws-runtime-branch | ws_protocol / app runtime 分支 | 完成 |
| p2-docker-deploy | compose、runtime.env、启动脚本 | 完成 |
| p2-linux-doc | README Linux+Runtime | 完成 |
| p2-doc-phase2 | 企业方案、架构、协议、验收清单 | 完成 |

## 启动顺序（必守）

1. 配置 `deploy/runtime.env` 中模型路径（日语 2pass 按 FunASR 官方文档）。
2. `docker compose -f deploy/docker-compose.yml up -d funasr-runtime`
3. `python scripts/run_server.py --config config.runtime.yaml`
4. `curl http://127.0.0.1:8765/ready`
