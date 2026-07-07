"""Secret storage — encrypt API keys at rest, never return plaintext to the UI."""
from __future__ import annotations

import base64
import hashlib
import os

from .config import SECRET_KEY_PATH

try:
    from cryptography.fernet import Fernet

    _HAS_CRYPTO = True
except Exception:  # pragma: no cover - optional dependency fallback
    _HAS_CRYPTO = False


def _load_or_create_key() -> bytes:
    if SECRET_KEY_PATH.exists():
        return SECRET_KEY_PATH.read_bytes()
    key = Fernet.generate_key() if _HAS_CRYPTO else os.urandom(32)
    SECRET_KEY_PATH.write_bytes(key)
    try:
        os.chmod(SECRET_KEY_PATH, 0o600)
    except OSError:
        pass
    return key


def encrypt(plaintext: str) -> str:
    """Encrypt a secret for storage. Returns an opaque token string."""
    if not plaintext:
        return ""
    key = _load_or_create_key()
    if _HAS_CRYPTO:
        return Fernet(key).encrypt(plaintext.encode()).decode()
    # Fallback: XOR-mask (not for production, but keeps app functional).
    stream = _keystream(key, len(plaintext.encode()))
    masked = bytes(a ^ b for a, b in zip(plaintext.encode(), stream))
    return "x:" + base64.urlsafe_b64encode(masked).decode()


def decrypt(token: str) -> str:
    if not token:
        return ""
    key = _load_or_create_key()
    if _HAS_CRYPTO and not token.startswith("x:"):
        try:
            return Fernet(key).decrypt(token.encode()).decode()
        except Exception:
            return ""
    if token.startswith("x:"):
        raw = base64.urlsafe_b64decode(token[2:].encode())
        stream = _keystream(key, len(raw))
        return bytes(a ^ b for a, b in zip(raw, stream)).decode(errors="ignore")
    return ""


def mask(plaintext: str) -> str:
    """Produce a display-safe masked hint of a secret, e.g. '••••••cd12'."""
    if not plaintext:
        return ""
    tail = plaintext[-4:] if len(plaintext) >= 4 else ""
    return "••••••" + tail


def _keystream(key: bytes, n: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < n:
        out += hashlib.sha256(key + counter.to_bytes(4, "big")).digest()
        counter += 1
    return out[:n]
