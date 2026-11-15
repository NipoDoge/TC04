"""Leitura e identificação do conteúdo dos artefatos locais."""

import hashlib
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path):
    """Lê uma única versão do arquivo para uso e registro do mesmo conteúdo."""
    path = Path(path).resolve(strict=True)
    payload = path.read_bytes()
    return payload, {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest()}
