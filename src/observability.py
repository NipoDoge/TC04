"""Log central do pipeline: eventos estruturados (JSON Lines) e medição de etapas.

Cada etapa (preparação, treino, ingestão, escoragem, drift, avaliação, alertas)
grava uma linha em ``logs/pipeline_events.jsonl`` com o mesmo formato. O arquivo
é a fonte única do dashboard e não recebe linhas de clientes, apenas agregados.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import uuid

import numpy as np

from src import config

log = logging.getLogger("pipeline")
_context = {"run_id": None}
LEVELS = {"INFO": logging.INFO, "WARNING": logging.WARNING,
          "ALERT": logging.WARNING, "CRITICAL": logging.ERROR, "ERROR": logging.ERROR}


def new_run_id(prefix="run"):
    """Identificador único da execução; fica em todos os eventos gerados por ela."""
    _context["run_id"] = f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"
    return _context["run_id"]


def setup_logging(level=logging.INFO):
    """Console + ``logs/pipeline.log``; chamado pelos pontos de entrada dos scripts."""
    handlers = [logging.StreamHandler()]
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(config.TEXT_LOG, encoding="utf-8"))
    except OSError as error:
        print(f"Log em arquivo indisponível: {error}")
    logging.basicConfig(level=level, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Tipo não serializável: {type(value)!r}")


def log_event(stage, status, *, batch=None, level="INFO", duration_s=None, rows=None,
              run_id=None, **details):
    """Acrescenta um evento ao log central. Falha de escrita nunca derruba o pipeline."""
    event = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "run_id": run_id or _context["run_id"],
        "stage": stage,
        "batch": batch,
        "status": status,
        "level": level,
        "duration_s": None if duration_s is None else round(float(duration_s), 4),
        "rows": None if rows is None else int(rows),
        "details": details,
    }
    try:
        config.EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with config.EVENT_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=_plain) + "\n")
    except (OSError, TypeError) as error:
        log.warning("Evento não gravado no log central: %s", error)
    log.log(LEVELS.get(level, logging.INFO), "stage=%s batch=%s status=%s rows=%s duration_s=%s",
            stage, batch, status, event["rows"], event["duration_s"])
    return event


@contextmanager
def track(stage, *, batch=None):
    """Mede uma etapa e registra sucesso ou falha. ``info`` recebe linhas e detalhes."""
    info, start = {}, time.perf_counter()
    try:
        yield info
    except Exception as error:
        log_event(stage, "failed", batch=batch, level="ERROR",
                  duration_s=time.perf_counter() - start, error=f"{type(error).__name__}: {error}")
        raise
    info["duration_s"] = time.perf_counter() - start
    log_event(stage, "success", batch=batch, duration_s=info["duration_s"],
              rows=info.get("rows"), **{k: v for k, v in info.items() if k not in ("rows", "duration_s")})


def read_events(path=None):
    """Lê o log central, ignorando linhas truncadas por uma escrita interrompida."""
    path, events = Path(path or config.EVENT_LOG), []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events
