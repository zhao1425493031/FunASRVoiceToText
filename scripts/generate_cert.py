#!/usr/bin/env python3
"""Generate self-signed TLS cert for LAN HTTPS (iPhone microphone)."""

from __future__ import annotations

import argparse
import ipaddress
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CERTS_DIR = ROOT / "certs"
OPENSSL_CFG = CERTS_DIR / "openssl.cnf"


def build_openssl_config(extra_ips: list[str]) -> str:
    san_parts = ["DNS:localhost", "IP:127.0.0.1"]
    for ip in extra_ips:
        try:
            ipaddress.ip_address(ip)
            san_parts.append(f"IP:{ip}")
        except ValueError:
            san_parts.append(f"DNS:{ip}")
    san = ",".join(san_parts)
    return f"""[req]
distinguished_name = req_distinguished_name
x509_extensions = v3_req
prompt = no

[req_distinguished_name]
CN = voicetotext

[v3_req]
subjectAltName = {san}
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate certs/cert.pem and certs/key.pem")
    parser.add_argument(
        "--ip",
        action="append",
        default=[],
        help="LAN IPv4 to include (repeatable). Example: --ip 192.168.0.118",
    )
    args = parser.parse_args()

    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    OPENSSL_CFG.write_text(build_openssl_config(args.ip), encoding="utf-8")

    key_path = CERTS_DIR / "key.pem"
    cert_path = CERTS_DIR / "cert.pem"

    cmd = [
        "openssl",
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-keyout",
        str(key_path),
        "-out",
        str(cert_path),
        "-days",
        "825",
        "-nodes",
        "-config",
        str(OPENSSL_CFG),
        "-extensions",
        "v3_req",
    ]
    print("Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("openssl not found. Install OpenSSL or use mkcert:", file=sys.stderr)
        print("  mkcert -install", file=sys.stderr)
        print(f"  mkcert -key-file {key_path} -cert-file {cert_path} localhost 127.0.0.1 {' '.join(args.ip)}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"openssl failed: {exc}", file=sys.stderr)
        return 1

    print(f"Created {cert_path}")
    print(f"Created {key_path}")
    print("Restart server: python scripts/run_server.py")
    print("Phone: https://<your-lan-ip>:8765 (trust cert on iOS: Settings > General > About > Certificate Trust Settings)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
