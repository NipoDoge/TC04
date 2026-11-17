"""Eventos reais de execução, sem valores de clientes ou mensagens de exceção."""

from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import fcntl
import json
import logging
import os
from pathlib import Path
import resource
import shutil
import sys
from time import perf_counter
from uuid import uuid4

import numpy as np

DEFAULT_LOG = Path("reports/observability/executions.jsonl")
_current = ContextVar("tc04_execution", default=None)


def now():
    return datetime.now(timezone.utc).isoformat()


def record(**metrics):
    """Acrescenta somente contagens, métricas e referências definidas pelo código."""
    event = _current.get()
    if event is not None:
        event.update(metrics)


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Tipo não serializável: {type(value).__name__}")


def append_event(event):
    path = Path(os.environ.get("TC04_EXECUTION_LOG", DEFAULT_LOG))
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, allow_nan=False, default=_plain) + "\n"
    # Evita intercalar linhas de processos que compartilham o mesmo log (POSIX).
    with path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def infrastructure():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    disk = shutil.disk_usage(Path.cwd())
    return {
        "process_peak_rss_mb": usage.ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024),
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "cpu_count": os.cpu_count(),
    }


def tracked(stage, *, exit_codes=False):
    """Preserva retorno/exceções; grava início e término com IDs correlacionáveis."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            execution_id = uuid4().hex
            started = now()
            base = {
                "schema_version": 1,
                "run_id": os.environ.get("TC04_RUN_ID") or execution_id,
                "execution_id": execution_id,
                "source": stage,
                "started_at_utc": started,
            }
            append_event({**base, "ts": started, "event": "execution_started", "status": "running"})
            details = {"volume": None, "volume_unit": "rows", "error_type": None}
            start = perf_counter()
            cpu_start = resource.getrusage(resource.RUSAGE_SELF)
            token = _current.set(details)
            status = "ok"
            operation_error = None
            try:
                result = function(*args, **kwargs)
                if exit_codes:
                    details["exit_code"] = result
                    status = {0: "ok", 1: "rejected"}.get(result, "fail")
                elif details.get("alert_count", 0):
                    status = "alert"
                return result
            except BaseException as error:
                operation_error = error
                status = "rejected" if type(error).__name__ in {"SchemaError", "SchemaErrors", "TrainingDataError"} else "fail"
                details["error_type"] = type(error).__name__
                raise
            finally:
                try:
                    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
                    append_event({
                        **base, **details, "ts": now(), "event": "execution_finished",
                        "status": status, "duration_ms": round((perf_counter() - start) * 1000, 3),
                        "cpu_seconds": round(cpu_end.ru_utime + cpu_end.ru_stime - cpu_start.ru_utime - cpu_start.ru_stime, 6),
                        "infrastructure": infrastructure(),
                    })
                except Exception as logging_error:
                    if operation_error is None:
                        raise
                    # Uma falha secundária de telemetria não deve ocultar o erro original.
                    operation_error.add_note(f"Falha ao registrar término: {type(logging_error).__name__}")
                    logging.getLogger("pipeline").error(
                        "Término não registrado: execution_id=%s error_type=%s",
                        execution_id, type(logging_error).__name__,
                    )
                finally:
                    _current.reset(token)
        return wrapped
    return decorate
