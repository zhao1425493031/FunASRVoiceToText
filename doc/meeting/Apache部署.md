# Apache 反代会议服务（v2）

## 拓扑

- 对外：`https://meeting.example.com`（443）
- 内网：`https://127.0.0.1:8766`（FastAPI + 自签证书）

## 模块

```apache
LoadModule proxy_module modules/mod_proxy.so
LoadModule proxy_http_module modules/mod_proxy_http.so
LoadModule proxy_wstunnel_module modules/mod_proxy_wstunnel.so
LoadModule ssl_module modules/mod_ssl.so

<VirtualHost *:443>
    ServerName meeting.example.com
    SSLEngine on
    SSLCertificateFile /path/to/fullchain.pem
    SSLCertificateKeyFile /path/to/privkey.pem

    ProxyPreserveHost On

    # 静态页（可选，也可直接访问 8766）
    ProxyPass /meeting https://127.0.0.1:8766/meeting
    ProxyPassReverse /meeting https://127.0.0.1:8766/meeting

    # WebSocket v2
    ProxyPass /ws/meeting/asr wss://127.0.0.1:8766/ws/meeting/asr
    ProxyPassReverse /ws/meeting/asr wss://127.0.0.1:8766/ws/meeting/asr

    SSLProxyEngine on
    SSLProxyVerify none
    SSLProxyCheckPeerCN off
    SSLProxyCheckPeerExpire off
</VirtualHost>
```

## 服务器配置

`config.meeting.yaml`：

```yaml
device: cuda
host: "0.0.0.0"
port: 8766
```

环境变量：

```bash
export HF_TOKEN=hf_xxxx
```

## 健康检查

- `GET https://127.0.0.1:8766/health`
- `GET https://127.0.0.1:8766/ready`（需模型与 HF_TOKEN 就绪）
