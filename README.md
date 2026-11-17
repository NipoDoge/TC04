# TC04 | Governança e confiabilidade de um modelo de crédito

Projeto acadêmico MLET/FIAP: validação de dados, detecção de drift e análise da confiabilidade de um modelo de crédito com o dataset **Give Me Some Credit** e XGBoost.

Os cenários são sintéticos, sem série temporal real ou comprovação de causa econômica. O projeto não realiza concessões reais de crédito.

## Situação da entrega

| Etapa | Situação |
| --- | --- |
| 1 — Qualidade e contratos | Executada: Pandera, bloqueio/quarentena e baseline treinado |
| 2 — Drift | Executada: data drift, concept drift, predições e relatórios Evidently |
| 3 — Observabilidade | Executada: logs reais por operação, recursos, histórico e dashboard com `make observe` |
| 4 — Governança e vídeo | Plano documentado; procedência dos dados, responsáveis e vídeo pendentes |

**Vídeo STAR:** pendente de gravação e publicação, com duração máxima de cinco minutos. [Roteiro e checklist](docs/entrega.md).

## Executar

Usar **Python 3.12**, a partir da raiz do repositório. No macOS, instalar o runtime OpenMP (`brew install libomp`).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
make test PYTHON=.venv/bin/python
```

A suíte possui **114 testes**, incluindo os casos da entrega final adaptados aos scripts desta branch. A cobertura e os resultados estão em [Reprodução e validação](docs/reproducao.md#testes-adaptados-da-entrega-final). O notebook é opcional; suas dependências estão em [requirements-notebooks.txt](requirements-notebooks.txt).

| Comando `make` | Finalidade |
| --- | --- |
| `prepare` / `train` | Preparar partições e treinar/publicar uma nova versão do baseline |
| `demo-bad` / `demo-good` | Demonstrar rejeição e aprovação do contrato, nessa ordem |
| `simulate-drift` / `predict` / `detect-drift` | Gerar cenários, predições e relatórios de data drift |
| `compare-data-drift` | Comparar o perfil anterior com o atual, usando o mesmo modelo/holdout |
| `concept-drift` | Alterar o mecanismo do alvo com features e modelo fixos |
| `observe` | Consolidar logs reais, recursos e alertas no dashboard (Etapa 3) |
| `demo-observability` | Executar demonstração isolada com falhas, bloqueio, recuperação e drift |

Adicionar `PYTHON=.venv/bin/python` para escolher o interpretador. `make` exibe ajuda.

Os artefatos já estão versionados. Simulação, inferência e relatórios exigem **destinos inexistentes**; para novas execuções, usar os argumentos `--output-dir` dos scripts. Preparação e geração dos lotes sobrescrevem suas saídas; o treino publica uma nova versão. [Reprodução e ressalvas](docs/reproducao.md).

## Observabilidade (Etapa 3)

Os scripts registram início e término reais em `reports/observability/executions.jsonl`, com IDs, duração, linhas processadas, tipo de erro, CPU e pico de memória do processo. O dashboard consolida o histórico, a última execução de cada etapa, espaço em disco e alertas de data drift e concept drift. As métricas do baseline aparecem separadas dos cenários de degradação.

A demonstração preservada contém **10 operações / 20 eventos**: duas falhas provocadas, um bloqueio de contrato e recuperação, com alertas de drift ativos ao final. [Painel da demonstração](artifacts/operational_demo/dashboard/index.html) e [resumo](artifacts/operational_demo/summary.json). Para reproduzir em uma pasta ainda inexistente: `make demo-observability PYTHON=.venv/bin/python`. O painel é atualizado sob demanda; etapas sem logs ficam `unknown` e registros vencidos ficam `stale`.

**1. Gerar / atualizar o painel** (na raiz do repositório):

```bash
make observe PYTHON=.venv/bin/python
# equivalente:
python -m src.observability
```

Saída em `reports/observability/`: `events.jsonl`, `metrics.json` e [`index.html`](reports/observability/index.html).

**2. Abrir o dashboard** — escolha uma das formas:

- **Arquivo local:** abra `reports/observability/index.html` no navegador (duplo clique no Explorer/Finder, ou arraste o arquivo para o Chrome/Firefox).
- **Pelo terminal (caminho file://):**

```bash
# Linux / WSL (exemplo)
xdg-open reports/observability/index.html
# macOS
open reports/observability/index.html
```

- **Servidor HTTP local** (útil se o navegador bloquear links relativos via `file://`):

```bash
cd reports && python -m http.server 8765
# no navegador: http://127.0.0.1:8765/observability/index.html
```

O journal acumulativo é preservado a cada atualização. Os testes usam logs temporários. Para agrupar comandos, definir `TC04_RUN_ID`; para mudar o journal, `TC04_EXECUTION_LOG`. No painel há links para os HTML do Evidently. Detalhe das métricas: [Observabilidade](docs/observabilidade.md).

## Resultados principais

| Evidência | Resultado | Relatório |
| --- | --- | --- |
| Baseline na validação | AUC 0,857208; recall 77,23% no limiar 0,5 | [Etapa 1](docs/etapa-1.md) |
| Data drift atual | Utilização de crédito detectada em t1–t3; atrasos de 90 dias em t2–t3 | [Evidently t3](reports/drift/t3.html) |
| Comparativo de intervenções | Em t3, AUC 0,842989 no perfil anterior e 0,800850 no atual | [Análise comparativa](reports/drift_comparison/analysis.html) |
| Concept drift | AUC 0,843257 → 0,669626 e recall 74,09% → 45,35%, com features fixas | [Análise do experimento](artifacts/concept_drift/analysis.html) |
| Observabilidade | Execuções reais, recuperação, duração, volumes, recursos e alertas de dados/desempenho | [Dashboard](reports/observability/index.html) |

As métricas dos experimentos comparam o mesmo holdout ao seu controle; a validação do baseline é outra partição. As doses dos perfis de data drift não são economicamente equivalentes.

## Governança e privacidade

Não foram identificados nome, CPF, telefone ou e-mail. O índice original é removido; isso **não comprova anonimização**. Dados financeiros, combinações de atributos e valores em quarentena exigem revisão antes da divulgação.

A base legal proposta para a fintech hipotética é proteção do crédito; o enquadramento do uso acadêmico e os termos do dataset continuam pendentes. As políticas são propostas documentais, sem comprovação de conformidade operacional integral.

A retenção proposta é: dados brutos/derivados até 90 dias após a avaliação final; quarentena até 30 dias após a triagem; logs sem valores individuais por 180 dias após a execução; relatórios por 12 meses após a avaliação; modelos/metadados até 12 meses após a substituição. O descarte não está automatizado.

O [plano completo de governança](docs/governanca.md) detalha fontes legais, tratamento dos dados, retenção, vieses, causalidade, responsabilidades e resposta à degradação.

## Documentação detalhada

| Documento | Conteúdo |
| --- | --- |
| [Arquitetura](docs/arquitetura.md) | Dados, features, modelo, fluxo e mapa das evidências |
| [Observabilidade](docs/observabilidade.md) | Métricas de saúde, `make observe` e dashboard (Etapa 3) |
| [Etapa 1](docs/etapa-1.md) | Contrato, limpeza, limitações, baseline e execução preservada |
| [Data drift](docs/data-drift.md) | Cenários atuais, predições, métodos e relatórios estatísticos |
| [Comparativo](docs/comparativo-drift.md) | Perfil anterior, efeitos no modelo e importância das features |
| [Concept drift](docs/concept-drift.md) | Mecanismo sintético do alvo, segmentos e sensibilidade às seeds |
| [Governança](docs/governanca.md) | Privacidade, base legal, retenção, vieses e causalidade |
| [Reprodução](docs/reproducao.md) | Dependências, comandos, testes, auditorias e ressalvas |
| [Entrega](docs/entrega.md) | Vídeo STAR e checklist das pendências |
