#!/usr/bin/env python3
"""Generate self-signed TLS cert for LAN HTTPS (iPhone microphone)."""

from __future__ import annotations

import argparse
import ipaddress
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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


def _collect_san_names(extra_ips: list[str]) -> list[str]:
    names = ["localhost", "127.0.0.1"]
    for ip in extra_ips:
        if ip not in names:
            names.append(ip)
    return names


def generate_with_cryptography(key_path: Path, cert_path: Path, extra_ips: list[str]) -> None:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    san_names = _collect_san_names(extra_ips)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "voicetotext")]
    )
    alt_names: list[x509.GeneralName] = []
    for name in san_names:
        try:
            addr = ipaddress.ip_address(name)
            alt_names.append(x509.IPAddress(addr))
        except ValueError:
            alt_names.append(x509.DNSName(name))

    now = datetime.now(timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(alt_names), critical=False)
        .sign(key, hashes.SHA256())
    )

    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def generate_with_openssl(extra_ips: list[str], key_path: Path, cert_path: Path) -> None:
    CERTS_DIR.mkdir(parents=True, exist_ok=True)
    OPENSSL_CFG.write_text(build_openssl_config(extra_ips), encoding="utf-8")
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
    subprocess.run(cmd, check=True)


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
    key_path = CERTS_DIR / "key.pem"
    cert_path = CERTS_DIR / "cert.pem"

    try:
        generate_with_openssl(args.ip, key_path, cert_path)
    except FileNotFoundError:
        print("openssl not found, using Python cryptography...")
        try:
            generate_with_cryptography(key_path, cert_path, args.ip)
        except ImportError:
            print("Install cryptography: pip install cryptography", file=sys.stderr)
            print("Or install OpenSSL / mkcert:", file=sys.stderr)
            print(
                f"  mkcert -key-file {key_path} -cert-file {cert_path} "
                f"localhost 127.0.0.1 {' '.join(args.ip)}",
                file=sys.stderr,
            )
            return 1
    except subprocess.CalledProcessError as exc:
        print(f"openssl failed: {exc}", file=sys.stderr)
        return 1

    print(f"Created {cert_path}")
    print(f"Created {key_path}")
    san = ", ".join(_collect_san_names(args.ip))
    print(f"SAN: {san}")
    print("Restart: python scripts/run_meeting.py")
    print("Phone: https://<your-lan-ip>:8766/meeting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
