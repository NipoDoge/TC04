# TC04 | Governança e confiabilidade de um modelo de crédito

Projeto acadêmico MLET/FIAP: validação de dados, detecção de drift e análise da confiabilidade de um modelo de crédito com o dataset **Give Me Some Credit** e XGBoost.

Os cenários são sintéticos, sem série temporal real ou comprovação de causa econômica. O projeto não realiza concessões reais de crédito.

## Situação da entrega

| Etapa | Situação |
| --- | --- |
| 1 — Qualidade e contratos | Executada: Pandera, bloqueio/quarentena e baseline treinado |
| 2 — Drift | Executada: data drift, concept drift, predições e relatórios Evidently |
| 3 — Observabilidade | Executada: logs reais por operação, recursos, histórico e dashboard com `make observe` |
| 4 — Governança e vídeo | Governança documentada, enquadramento acadêmico validado e dados revisados pelo grupo; gravação, revisão e publicação do vídeo pendentes |

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
| `demo-observability` | Executar demonstração no destino padrão, que deve estar inexistente; para repetir, seguir a seção de reprodução abaixo |

Adicionar `PYTHON=.venv/bin/python` para escolher o interpretador. `make` exibe ajuda.

Os artefatos já estão versionados. Simulação, inferência e relatórios exigem **destinos inexistentes**; para novas execuções, usar os argumentos `--output-dir` dos scripts. Preparação e geração dos lotes sobrescrevem suas saídas; o treino publica uma nova versão. [Reprodução e ressalvas](docs/reproducao.md).

## Observabilidade

Os scripts registram início e término reais em `reports/observability/executions.jsonl`, com IDs, duração, linhas processadas, tipo de erro, CPU e pico de memória do processo. O dashboard consolida o histórico, a última execução de cada etapa, espaço em disco e alertas de data drift e concept drift. As métricas do baseline aparecem separadas dos cenários de degradação.

A demonstração preservada contém **10 operações / 20 eventos**: duas falhas provocadas, um bloqueio de contrato e recuperação, com alertas de drift ativos ao final. [Painel da demonstração](artifacts/operational_demo/dashboard/index.html) e [resumo](artifacts/operational_demo/summary.json). O painel é atualizado sob demanda; etapas sem logs ficam `unknown` e registros vencidos ficam `stale`.

### Reproduzir a demonstração

Com as dependências instaladas, executar na raiz do repositório, escolhendo uma pasta de saída **que ainda não exista**:

```bash
.venv/bin/python -m src.demo_observability --output-dir artifacts/operational_demo_nova
```

O comando executa preparação, treino, ingestão, simulação, inferência e relatórios em uma pasta separada, preservando os dados, modelos e experimentos anteriores. O alvo `make demo-observability` usa o destino padrão `artifacts/operational_demo/`, já versionado neste repositório; por isso, a reexecução deve usar `--output-dir` como no exemplo. Em cada repetição, escolher outro nome de pasta. Se o destino já existir, o script encerra com `FileExistsError` para preservar seu conteúdo.

A nova execução gera:

- `artifacts/operational_demo_nova/dashboard/index.html`: painel da execução, com seus próprios modelos e relatórios.
- `artifacts/operational_demo_nova/summary.json`: resumo com identificador, resultados e localização do painel.
- Novos eventos no journal acumulativo `reports/observability/executions.jsonl`.

As duas falhas operacionais e a rejeição de contrato são provocadas para demonstrar bloqueio e recuperação. Ao concluir, o resultado esperado é **20 eventos de 10 operações**, ingestão final `ok` com 500 registros e estado consolidado `alert` pelos cenários de drift. Os incidentes anteriores permanecem no histórico.

### Atualizar e abrir os painéis

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

- **Servidor HTTP local** (executar na raiz do repositório):

```bash
.venv/bin/python -m http.server 8765 --bind 127.0.0.1
# painel principal: http://127.0.0.1:8765/reports/observability/index.html
# demonstração: http://127.0.0.1:8765/artifacts/operational_demo/dashboard/index.html
# nova execução: http://127.0.0.1:8765/artifacts/operational_demo_nova/dashboard/index.html
```

Manter a raiz do repositório como diretório servido permite acessar os relatórios em `reports/` e `artifacts/`, incluindo a análise de concept drift. Encerrar o servidor com `Ctrl+C`.

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

O dataset foi obtido no [Kaggle — Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/) em **28 de outubro de 2026**. Sua utilização neste projeto limita-se a fins acadêmicos, sem fins lucrativos.

Não foi informado um identificador de versão externa do dataset. A cópia de treino utilizada (`data/raw/cs-training.csv`) é identificada pelo SHA-256 registrado no [manifesto da Etapa 1](artifacts/stage1/20261114T201315-0300-5fbfe64d/manifest.json), no campo `input.sha256`.

Não foram identificados nome, CPF, telefone ou e-mail. O índice original é removido, embora isso não comprove anonimização.

A base legal proposta para a fintech hipotética é proteção do crédito. O enquadramento acadêmico validado pelo grupo descreve o Tech Challenge da Fase 4 do MLET/FIAP como estudo educacional de qualidade, drift e observabilidade, sem decisões reais de crédito. Abrange obtenção, preparação, treinamento, simulação e apresentação dos resultados à equipe, avaliadores e demais pessoas com acesso ao repositório e ao vídeo. O texto completo está em [Enquadramento acadêmico](docs/governanca.md#enquadramento-acadêmico).

A [revisão dos dados para publicação](docs/governanca.md#revisão-da-publicação) está concluída conforme confirmação do grupo. O documento mantém o escopo da revisão e as diretrizes para apresentação e novas publicações, priorizando resultados agregados e demonstrações sintéticas. A revisão da gravação integra a entrega do vídeo STAR. As políticas documentadas não comprovam conformidade operacional integral.

A retenção proposta é: dados brutos/derivados até 90 dias após a avaliação final; quarentena até 30 dias após a triagem; logs sem valores individuais por 180 dias após a execução; relatórios por 12 meses após a avaliação; modelos/metadados até 12 meses após a substituição. O descarte não está automatizado.

### Causalidade e limites das conclusões

Os experimentos permitem atribuir diferenças à **intervenção programada dentro da simulação**: no data drift, alteram-se utilização de crédito e atrasos de 90 dias, preservando os rótulos; no concept drift, altera-se o mecanismo do alvo, mantendo features, modelo e predições fixos. As métricas são comparadas ao controle do mesmo holdout. Os cenários reutilizam uma amostra aleatória e não representam uma série temporal real de concessões.

Inflação, redução de renda, endividamento e mudanças na política de aprovação são hipóteses de domínio a investigar. O projeto não estima seus efeitos econômicos. PSI e KS identificam mudanças de distribuição; sua detecção, a importância de uma feature ou a queda de AUC não estabelecem, isoladamente, a causa da degradação. Um diagnóstico operacional deve verificar qualidade, seleção da população, política de crédito e maturação dos rótulos; o alvo tem horizonte de dois anos.

### Riscos de viés e mitigação proposta

| Risco identificado | Mitigação a avaliar antes de uso operacional |
| --- | --- |
| Excluir renda/dependentes ausentes e valores extremos pode mudar a população atendida | Comparar incluídos e excluídos por idade e renda; avaliar imputação ajustada somente no treino e indicador de ausência como alternativa à exclusão |
| Desequilíbrio do alvo e limiar fixo podem distribuir erros de forma desigual | Avaliar AUC, recall, FPR, FNR e calibração por segmento, com tamanho amostral e incerteza; definir o limiar considerando inadimplência e recusa indevida |
| Idade, dependentes e atributos financeiros podem funcionar como proxies socioeconômicos | Justificar a necessidade das variáveis, comparar alternativas e submeter impactos à revisão de Risco e Privacidade |

O `scale_pos_weight` trata o desequilíbrio de classes no treinamento, mas não demonstra equidade. A análise existente aponta risco de seleção na limpeza; a auditoria de equidade e as comparações de alternativas acima são propostas. A ausência de atributos suficientes para avaliar todos os grupos protegidos limita as conclusões.

### Resposta à degradação

**Implementado:** bloqueio e quarentena por contrato, detecção estatística de data drift, alertas de desempenho no experimento de concept drift e registro de execuções reais no painel local. Nesse experimento, queda absoluta de AUC ≥ 0,05 ou recall ≥ 0,10 contra o controle do mesmo holdout gera alerta; esses limiares iniciais precisam de calibração para uma operação real.

**Fluxo operacional proposto:**

1. Registrar incidente com lote, horário e versões de código, dados e modelo. Engenharia investiga integridade e origem; Ciência de Dados examina distribuição, rótulos e desempenho; Risco avalia o impacto.
2. Corrigir e revalidar lotes bloqueados. Em degradação crítica, o responsável operacional decide sobre suspensão da decisão automática ou encaminhamento à análise humana.
3. Avaliar correção, recalibração ou retreinamento, comparando o candidato em dados independentes, inclusive por segmento. Exigir aprovação de Risco e do responsável técnico antes da publicação.
4. Acompanhar a nova versão, manter a anterior para rollback e registrar o encerramento e as medidas de prevenção.

O painel é atualizado sob demanda. Notificações externas, resposta a incidentes e promoção de novos modelos não estão automatizadas. As responsabilidades funcionais do plano estão descritas na documentação de governança.

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
