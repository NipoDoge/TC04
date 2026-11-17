# Dados, modelo e arquitetura

[Voltar ao README](../README.md)

## Escopo e evidências disponíveis

| Componente | Evidência no repositório | Situação |
| --- | --- | --- |
| Exploração e avaliação inicial de seleção | [Notebook](../notebooks/01_exploracao.ipynb) | Exploração auxiliar; saídas antigas removidas; leitura sem modificar dados |
| Limpeza e separação treino/validação/produção | [prepare.py](../src/prepare.py), relatório de limpeza e arquivos Parquet | Implementados; percentis ajustados somente no treino; divisão aleatória, sem simulação temporal |
| Contrato de dados | [contract.py](../src/contract.py) | Implementado em Pandera |
| Bloqueio e quarentena | [ingest.py](../src/ingest.py) | Implementados para falhas de validação do contrato |
| Lotes de demonstração | [Log de aprovação](../artifacts/stage1/20261114T201315-0300-5fbfe64d/06_ingest_good.txt), [log de bloqueio](../artifacts/stage1/20261114T201315-0300-5fbfe64d/07_ingest_bad.txt) e [falhas](../data/quarantine/bad_batch_errors.csv) | Executados: 500 linhas aprovadas e lote de 505 linhas em quarentena |
| Treinamento e avaliação | [Modelo](../models/runs/32542e11c8c8477ea23d42db115efd0e/model.joblib), [métricas](../models/runs/32542e11c8c8477ea23d42db115efd0e/metrics.json) e [manifesto da execução](../artifacts/stage1/20261114T201315-0300-5fbfe64d/manifest.json) | Baseline treinado, métricas conferidas após recarregar o modelo e publicação por versão |
| Simulação de data drift | [simulate_drift.py](../src/simulate_drift.py): controle e três períodos sintéticos pareados a partir do holdout | Implementada; não representa série temporal real |
| Predições da Etapa 2 | [predict.py](../src/predict.py): inferência com o modelo salvo na referência e nos quatro cenários | Implementadas; probabilidades e classes em arquivos separados |
| Detecção de data drift | [detect_drift.py](../src/detect_drift.py), [resumo estatístico](../reports/drift/summary.csv) e relatórios HTML do Evidently | Executada: utilização de crédito detectada em t1–t3; atrasos de 90 dias detectados em t2–t3 |
| Comparativo de intervenções | [compare_data_drift.py](../src/compare_data_drift.py) e [análise comparativa](../reports/drift_comparison/analysis.html) | Mesmo modelo/holdout; perfil anterior de renda/dívida preservado e comparado ao perfil atual de utilização/atrasos |
| Concept drift e comportamento do modelo | [concept_drift.py](../src/concept_drift.py), [análise](../artifacts/concept_drift/analysis.html) e [métricas](../artifacts/concept_drift/metrics.csv) | Executados: alteração condicional do alvo, features e modelo fixos, avaliação global e por segmento e sensibilidade a cinco seeds |
| Observabilidade centralizada | [`src/observability.py`](../src/observability.py), `make observe`, artefatos em [`reports/observability/`](../reports/observability/) e [métricas documentadas](observabilidade.md) | Executada (Opção A: dashboard [`index.html`](../reports/observability/index.html)) |
| Governança e causalidade | [Plano de governança](governanca.md) | Política proposta e hipóteses documentadas |
| Vídeo STAR | [Entrega e roteiro](entrega.md) | Gravação e link pendentes |

## Dados e finalidade

Os arquivos locais correspondem ao conjunto **Give Me Some Credit**, com 150.000 registros no `cs-training.csv`. A finalidade acadêmica é estimar o risco de inadimplência e estudar qualidade e confiabilidade. A origem de obtenção, a versão e os termos de uso devem ser registrados antes da entrega; a presença dos arquivos no Git não comprova autorização de redistribuição.

O alvo `SeriousDlqin2yrs` representa inadimplência grave no horizonte de dois anos. As dez variáveis preditoras são:

| Grupo | Colunas | Risco de privacidade e uso |
| --- | --- | --- |
| Perfil | `age`, `NumberOfDependents` | Combinações podem favorecer identificação e segmentação discriminatória |
| Situação financeira | `MonthlyIncome`, `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines` | Informações financeiras confidenciais; possíveis proxies socioeconômicos |
| Histórico de crédito | `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines` | Perfil financeiro e risco de exposição de comportamento |
| Desfecho | `SeriousDlqin2yrs` | Informação de inadimplência; usada no treino e na avaliação |

Não há colunas de nome, CPF, telefone ou e-mail no arquivo de treino inspecionado. A coluna `Unnamed: 0`, um índice do arquivo, é removida na preparação. **Essa remoção não comprova anonimização**: não há avaliação de reidentificação ou técnica de anonimização implementada.

Dados financeiros e idade podem ser dados pessoais quando vinculados a pessoa identificada ou identificável; não são automaticamente dados pessoais sensíveis na definição legal. Não foram identificadas colunas explícitas de saúde, biometria, religião ou origem racial. A distinção segue o [Glossário da ANPD](https://www.gov.br/anpd/pt-br/centrais-de-conteudo/materiais-educativos-e-publicacoes/glossario-anpd).

## Fluxo implementado

```text
data/raw/cs-training.csv
    -> src.prepare: limpeza fixa + divisões estratificadas (seed 42)
        -> treino (56%): ajusta os dois limites p99
        -> aplica os mesmos limites ao treino, validação (14%) e holdout (30%)
        -> data/reference/reference.parquet (somente treino)
        -> data/validation/validation.parquet
            -> src.train: valida contratos das duas partições
                -> treino XGBoost na referência e avaliação na validação
                -> models/.tmp-<execução>/{model.joblib,metrics.json}
                -> models/runs/<execução> (versão completa)
                -> models/current (apontador para a versão publicada)
        -> data/reference/outlier_limits.json + cleaning_report.json
        -> data/production/holdout.parquet
            -> src.make_bad_batch: lotes de demonstração
                -> data/incoming/{good,bad}_batch.parquet
                    -> src.ingest + contrato Pandera
                        -> válido: log e retorno 0
                        -> inválido: movimentação para data/quarantine,
                           CSV de falhas e retorno 1
                        -> falha operacional: log e retorno 2
            -> src.simulate_drift: controle e intervenções em utilização/atrasos de 90 dias
                -> data/production/scenarios/{t0_control,t1,t2,t3}.parquet
                -> data/production/scenarios/manifest.json

models/current/model.joblib + referência + quatro cenários
    -> src.predict: contrato + seleção ordenada de FEATURE_COLUMNS
        -> data/predictions/{reference,t0_control,t1,t2,t3}.parquet
        -> data/predictions/manifest.json

referência + quatro cenários (somente FEATURE_COLUMNS)
    -> src.detect_drift: Evidently DataDriftPreset + KS/PSI
        -> reports/drift/{t0_control,t1,t2,t3}.{html,json}
        -> reports/drift/summary.csv + manifest.json

perfil anterior arquivado + cenários/predições atuais + modelo fixo
    -> src.compare_data_drift
        -> reports/drift_comparison/{analysis.html,metrics.csv,feature_importance.csv}
        -> comparações nativas de classificação + importância conjunta + manifest.json

referência (ajusta apenas corte de renda) + holdout + modelo salvo
    -> src.concept_drift: altera Y condicionalmente a X; preserva X e modelo
        -> artifacts/concept_drift/{datasets,predictions}/
        -> métricas globais e por segmento + sensibilidade a cinco seeds
        -> relatórios de classificação do Evidently + prova de features estáveis
        -> analysis.html + interventions.parquet + manifest.json
```

[config.py](../src/config.py) centraliza alvo, seed, limiar de classificação, fatores dos períodos, critérios de drift e probabilidades do choque sintético no alvo. [artifacts.py](../src/artifacts.py) centraliza snapshots de leitura e hashes SHA-256 usados nos experimentos e relatórios. A ordem das features continua definida apenas em [contract.py](../src/contract.py).

Os percentuais são anteriores à filtragem de outliers; as proporções finais podem variar. Os scripts criam os diretórios de saída automaticamente. A preparação atual já foi executada; ao reproduzir, executar `python -m src.prepare` antes de treinar para manter as partições e os limites da mesma preparação. A ingestão aprovada somente valida o lote: não o encaminha automaticamente ao modelo. O treinamento valida referência e validação com o contrato e exige registros das duas classes em ambas antes de criar o modelo. Reprovação retorna 1; falha operacional retorna 2. Uma execução bloqueada preserva eventuais artefatos anteriores.

Modelo e métricas são gravados em um diretório temporário exclusivo dentro de `models/`. Somente após concluir ambas as gravações, a versão é movida para `models/runs/<execução>` e o link simbólico `models/current` é substituído atomicamente. Os caminhos publicados são `models/current/model.joblib` e `models/current/metrics.json`. Falhas de gravação ou publicação retornam 2 e preservam a versão anterior; temporários e versões não publicadas são removidos nas falhas tratadas. Versões publicadas anteriores permanecem disponíveis em `models/runs/`.

Para ler modelo e métricas da mesma execução enquanto houver novos treinamentos, resolver o apontador uma única vez com `run_dir = Path("models/current").resolve()` e abrir ambos a partir de `run_dir`. Todos os consumidores do projeto usam a estrutura publicada em `models/current` ou uma versão explícita de `models/runs`. A publicação por links simbólicos foi validada no macOS e requer suporte do filesystem a esses links.

O modelo é um `XGBClassifier` com 300 estimadores, profundidade 4, taxa de aprendizado 0,05 e ponderação da classe positiva. A avaliação calcula AUC ROC, recall com limiar 0,5 e KS como separação máxima entre TPR e FPR. **Esse KS de classificação não é um teste de drift entre datasets.** O limiar 0,5 é usado na avaliação; uma política de aprovação de crédito não foi implementada.
