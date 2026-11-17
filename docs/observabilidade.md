# Observabilidade — métricas de saúde (Etapa 3)

[Voltar ao README](../README.md) · [Arquitetura](arquitetura.md) · [Data drift (Etapa 2)](data-drift.md)

## Objetivo (Opção A)

Dar **visibilidade operacional** do pipeline e do modelo neste repositório: um local único de eventos/métricas e um **dashboard HTML estático**, sem cloud, sem regenerar Evidently nem `models/current`.

A Opção A produz, sob `reports/observability/`:

| Artefato | Papel |
| --- | --- |
| `events.jsonl` | Eventos tipados (uma linha JSON por evento), sem PII |
| `metrics.json` | Snapshot agregado de saúde |
| [`index.html`](../reports/observability/index.html) | Painel visual que consome o snapshot (+ amostra de eventos) |

Coleta e contratos de campos: `src/observability.py` (pós-coleta sobre artefatos já existentes → `events.jsonl` / `metrics.json` / `index.html`).

## Como gerar

Na raiz do repositório (mesmo interpretador do restante do projeto):

```bash
make observe
# equivalente:
python -m src.observability
```

Escrita **somente** em `reports/observability/`. Não chama `detect_drift` / `train` e não apaga `reports/drift/` nem `models/current`.

Abrir o painel: [`reports/observability/index.html`](../reports/observability/index.html).

## Métricas de saúde

Definições alinhadas ao que **hoje** aparece em `metrics.json`, `events.jsonl` e `index.html` (amostra pós `make observe` em 2026-11-17).

| Nome | Definição | Origem no repo | Onde aparece |
| --- | --- | --- | --- |
| Status das etapas | Estado derivado dos artefatos: `ingest` (quarentena), `train` (métricas do modelo), `detect_drift` (cenários/alertas). Valores: `ok` \| `fail` \| `rejected` \| `alert` \| `unknown` | `data/quarantine/` · `models/current/metrics.json` · `reports/drift/manifest.json` via `src/observability.py` | `metrics.json` → `pipeline.stages[].status`; `events.jsonl` (`pipeline_status` / `model_snapshot`); `index.html` — tabela “Status das etapas” + badges |
| `duration_ms` | Latência da etapa em milissegundos, se conhecida | Campo do contrato de eventos; **hoje sempre `null`** (não há cronometragem no pós-processamento) | `metrics.json` → `pipeline.stages[].duration_ms`; `events.jsonl` → `duration_ms`; `index.html` — coluna “Duração (ms)” mostra “—” |
| Volume | Contagem agregada (nunca conteúdo de linha): arquivos em quarentena (`ingest`), nº de cenários (`detect_drift`), linhas do cenário nos eventos de drift | Quarentena · `manifest.json` (`scenarios[].rows`, lista de cenários) | `metrics.json` → `pipeline.stages[].volume`; `events.jsonl` → `volume`; `index.html` — coluna “Volume” e amostra de eventos |
| Alertas de contrato / quarentena | Contagem de arquivos em `data/quarantine/` (rejeições de contrato Pandera) | `data/quarantine/` | `metrics.json` → `contract_alerts.quarantine_file_count`; evento `ingest`/`pipeline_status` com `refs.files`; `index.html` — banner + seção “Alertas de contrato / quarentena” |
| Alertas de drift | Por cenário: `dataset_drift_detected`, `drift_share`, `drifted_features`; agregados `dataset_drift_count` e `any_alert` | `reports/drift/manifest.json` (`scenarios[]`) | `metrics.json` → `drift.scenarios` / `drift.alerts`; eventos `drift_scenario` + `pipeline_status` de `detect_drift`; `index.html` — banner, tabela “Drift por cenário”, links `../drift/<cenário>.html` |
| Snapshot do modelo (AUC / recall / KS) | Métricas publicadas do baseline atual (referência offline, não “live prod”) | `models/current/metrics.json` (`auc`, `recall_at_0.5`, `ks`) | `metrics.json` → `model.*`; evento `train`/`model_snapshot`; `index.html` — cards + tabela “Métricas do modelo” |

### Leitura rápida do snapshot atual

- Etapas: `ingest=rejected` (volume 2), `train=ok`, `detect_drift=alert` (volume 4).  
- Contrato: `quarantine_file_count=2`.  
- Drift: 2 cenários com `dataset_drift_detected=true` → `any_alert=true`.  
- Modelo: AUC ≈ 0,8572; recall@0,5 ≈ 0,7723; KS ≈ 0,5676.

## Limitações (honestas)

- **Pós-coleta, não live:** o snapshot reflete o estado dos artefatos no momento de `make observe`, não um stream contínuo de execução.
- **`duration_ms` pode ser (e hoje é) `null`:** a camada de obs. não cronometra `ingest`/`train`/`detect_drift`.
- **Não regenera Evidently:** apenas lê `reports/drift/` e oferece links relativos aos HTML já versionados.
- **Sem PII:** eventos e métricas usam contagens, nomes de arquivo, paths relativos e hashes — não gravam linhas de cliente nem conteúdo de `*_errors.csv`.
- Saúde do modelo é **snapshot** de `models/current`, não monitoramento de produção em tempo real.

## Referências

- Dashboard: [`reports/observability/index.html`](../reports/observability/index.html)
- Arquitetura geral do projeto: [arquitetura.md](arquitetura.md)
- Drift / Evidently (Etapa 2): [data-drift.md](data-drift.md)
- Entrega / checklist: [entrega.md](entrega.md)
