import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.settings import settings


class EncryptionError(RuntimeError):
    pass


def _get_fernet() -> Fernet:
    if not settings.master_passphrase:
        raise EncryptionError("AILOGGER_MASTER_PASSPHRASE is required for key storage")
    digest = hashlib.sha256(settings.master_passphrase.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(digest)
    return Fernet(key)


def encrypt_secret(value: str) -> bytes:
    fernet = _get_fernet()
    return fernet.encrypt(value.encode("utf-8"))


def decrypt_secret(token: bytes) -> str:
    fernet = _get_fernet()
    try:
        return fernet.decrypt(token).decode("utf-8")
    except InvalidToken as exc:
        raise EncryptionError("Invalid encrypted secret") from exc
