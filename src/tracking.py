"""Rastreamento de experimentos e do monitoramento no MLflow (SQLite local).

O rastreamento é auxiliar: se o MLflow não estiver disponível, o pipeline segue
e o log central continua sendo gravado.
"""
import logging
import os

from src import config

log = logging.getLogger("tracking")


def start(experiment):
    """Configura o MLflow no experimento indicado. Devolve o módulo ou ``None``."""
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")   # silencia aviso informativo
    try:
        import mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", config.MLFLOW_URI))
        mlflow.set_experiment(experiment)
        return mlflow
    except Exception as error:
        log.warning("MLflow indisponível, seguindo sem rastreamento: %s", error)
        return None


def safely(action, *args, **kwargs):
    """Executa uma chamada de rastreamento sem deixar que ela derrube o pipeline."""
    try:
        return action(*args, **kwargs)
    except Exception as error:
        log.warning("Falha ao registrar no MLflow: %s", error)
        return None
