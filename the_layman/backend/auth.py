import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone

# Use scrypt for password hashing as it's built into python and memory-hard
def hash_password(password: str, salt: bytes = None) -> str:
    if salt is None:
        salt = os.urandom(16)
    key = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, maxmem=33554432)
    return f"{salt.hex()}:{key.hex()}"

def verify_password(password: str, hashed: str) -> bool:
    try:
        salt_hex, key_hex = hashed.split(":")
        salt = bytes.fromhex(salt_hex)
        expected_key = bytes.fromhex(key_hex)
        key = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=16384, r=8, p=1, maxmem=33554432)
        return secrets.compare_digest(key, expected_key)
    except Exception:
        return False

def generate_session_token() -> str:
    return secrets.token_urlsafe(32)

def generate_session_expiry(days: int = 30) -> str:
    expires = datetime.now(timezone.utc) + timedelta(days=days)
    return expires.isoformat()
