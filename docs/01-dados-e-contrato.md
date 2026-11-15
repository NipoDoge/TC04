# Arquitetura, dados e contrato (Etapa 1)

Parte da documentação do [TC04](../README.md). Próximo: [experimentos de drift](02-experimentos-de-drift.md).

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
            -> src.simulate_drift: controle e três níveis de data drift pareados
               (renda, endividamento, utilização do rotativo e atrasos de 30–59 dias)
                -> data/production/scenarios/{t0_control,t1,t2,t3}.parquet
                -> data/production/scenarios/manifest.json

models/current/model.joblib + referência + quatro cenários
    -> src.predict: contrato + seleção ordenada de FEATURE_COLUMNS
        -> data/predictions/{reference,t0_control,t1,t2,t3}.parquet
        -> data/predictions/manifest.json

referência + quatro cenários (somente FEATURE_COLUMNS)
    -> src.detect_drift: Evidently DataDriftPreset (KS/PSI) + tamanho do efeito
        -> reports/drift/{t0_control,t1,t2,t3}.{html,json}
        -> reports/drift/summary.csv + manifest.json

reports/drift/summary.csv + data/predictions
    -> src.scenario_impact: PSI do score e métricas do modelo por cenário
        -> reports/scenario_impact.json

referência (ajusta apenas corte de renda) + holdout + modelo salvo
    -> src.concept_drift: altera Y condicionalmente a X; preserva X e modelo
        -> artifacts/concept_drift/{datasets,predictions}/
        -> métricas globais e por segmento + sensibilidade a cinco seeds
        -> relatórios de classificação do Evidently + prova de features estáveis
        -> analysis.html + interventions.parquet + manifest.json

data/production/holdout.parquet + calendário de níveis (src/config.py)
    -> src.simulate_batches: 8 lotes mensais + 1 entrega corrompida (seed 42),
       com as funções de intervenção de src.simulate_drift e src.concept_drift
        -> data/production/batches/*.parquet + manifest.json
            -> src.monitor, para cada lote na ordem de chegada:
                -> contrato (src.ingest): inválido vai para
                   data/quarantine/<execução>/ e o lote é interrompido
                -> escoragem com models/current
                -> src.batch_drift: detector de src.detect_drift (Evidently)
                   + tamanho do efeito (PSI por decis) e KS
                   -> reports/batches/<lote>.html
                -> desempenho com o rótulo observado
                -> src.alerts: regras de alerta
                -> um evento por etapa em logs/pipeline_events.jsonl
                   e métricas por lote no MLflow (mlflow.db)
            -> reports/monitoring_summary.json
    -> src.causal_analysis (último lote) -> reports/causal_analysis.json

resumos + log central
    -> src.dashboard -> reports/dashboard.html
```

[config.py](../src/config.py) centraliza alvo, seed, limiar de classificação, fatores dos níveis de data drift, critérios de drift, probabilidades do choque sintético no alvo e calendário dos lotes mensais, além dos caminhos e dos limiares de alerta do monitoramento. [artifacts.py](../src/artifacts.py) centraliza snapshots de leitura e hashes SHA-256 usados nos experimentos e relatórios. A ordem das features continua definida apenas em [contract.py](../src/contract.py).

Os percentuais são anteriores à filtragem de outliers; as proporções finais podem variar. Os scripts criam os diretórios de saída automaticamente. A preparação atual já foi executada; ao reproduzir, executar `python -m src.prepare` antes de treinar para manter as partições e os limites da mesma preparação. Na demonstração da Etapa 1 (`src.ingest` pela linha de comando), a ingestão aprovada somente valida o lote. No monitoramento (`src.monitor`), o lote aprovado segue para escoragem, drift e avaliação, e o rejeitado é interrompido antes de chegar ao modelo. O treinamento valida referência e validação com o contrato e exige registros das duas classes em ambas antes de criar o modelo. Reprovação retorna 1; falha operacional retorna 2. Uma execução bloqueada preserva eventuais artefatos anteriores.

Modelo e métricas são gravados em um diretório temporário exclusivo dentro de `models/`. Somente após concluir ambas as gravações, a versão é movida para `models/runs/<execução>` e o link simbólico `models/current` é substituído atomicamente. Os caminhos publicados são `models/current/model.joblib` e `models/current/metrics.json`. Falhas de gravação ou publicação retornam 2 e preservam a versão anterior; temporários e versões não publicadas são removidos nas falhas tratadas. Versões publicadas anteriores permanecem disponíveis em `models/runs/`.

Para ler modelo e métricas da mesma execução enquanto houver novos treinamentos, resolver o apontador uma única vez com `run_dir = Path("models/current").resolve()` e abrir ambos a partir de `run_dir`. Todos os consumidores do projeto usam a estrutura publicada em `models/current` ou uma versão explícita de `models/runs`. O monitoramento e a análise causal resolvem `models/current` uma vez, leem modelo e métricas do mesmo diretório e gravam a versão usada no resumo; o impacto dos cenários usa as predições publicadas e registra a versão e o hash do modelo que as gerou. A publicação por links simbólicos foi validada no macOS e requer suporte do filesystem a esses links.

O modelo é um `XGBClassifier` com 300 estimadores, profundidade 4, taxa de aprendizado 0,05 e ponderação da classe positiva. A avaliação calcula AUC ROC, recall com limiar 0,5 e KS como separação máxima entre TPR e FPR. **Esse KS de classificação não é um teste de drift entre datasets.** O limiar 0,5 é usado na avaliação; uma política de aprovação de crédito não foi implementada.

## Qualidade de dados e limites do contrato

O contrato exige pelo menos um registro; valores finitos em todas as colunas numéricas (rejeita `+inf` e `-inf`); alvo binário; idade inteira entre 18 e 100, inclusive; renda não nula e não negativa; demais medidas e contagens não negativas; colunas e tipos definidos; ausência de colunas extras e de linhas integralmente duplicadas. Os campos não marcados como anuláveis também rejeitam nulos por padrão. Não há coerção de tipos configurada. Lotes vazios e números infinitos são violações do contrato, com retorno 1 na ingestão e no treinamento.

A lista `FEATURE_COLUMNS`, definida no contrato, determina a ordem das entradas do modelo. O treinamento seleciona as mesmas features nessa ordem na referência e na validação, permitindo arquivos com colunas em outra ordem. A lista fica registrada também nos nomes das features do XGBoost serializado.

A preparação remove duplicatas, idades fora do intervalo, registros sem renda ou dependentes, rendas negativas e códigos 96/98 nas colunas de atraso. Separa 30% para holdout e divide os 70% restantes em treino/validação na proporção 80/20. Calcula os percentis 99 de `DebtRatio` e de utilização de crédito exclusivamente no treino, antes de qualquer filtro estatístico. Aplica os mesmos limites às três partições, sem recalculá-los na validação ou no holdout. Os limites são salvos em `data/reference/outlier_limits.json`, e o relatório registra exclusões e tamanhos por partição.

O [relatório da preparação executada](../data/reference/cleaning_report.json) registra 609 duplicatas, 14 exclusões por idade, 29.214 por renda/dependentes ausentes, nenhuma por renda negativa e 145 pelos códigos 96/98. Restaram 120.018 registros antes dos filtros estatísticos. Os limites calculados somente no treino foram `DebtRatio ≤ 672,76` e `RevolvingUtilizationOfUnsecuredLines ≤ 1,09213079212`; os valores completos estão em [outlier_limits.json](../data/reference/outlier_limits.json).

| Partição | Antes dos outliers | Removidos por DebtRatio | Removidos por utilização | Registros finais |
| --- | ---: | ---: | ---: | ---: |
| Treino / referência | 67.209 | 673 | 669 | 65.867 |
| Validação | 16.803 | 156 | 167 | 16.480 |
| Holdout | 36.006 | 366 | 343 | 35.297 |

As exclusões são contagens sequenciais, não estimativas independentes de prevalência. Ao final, restaram 117.644 registros nas três partições. A execução verificou contratos, limites aplicados e ausência de registros compartilhados entre partições.

Limitações que precisam orientar a operação:

- O contrato exige `SeriousDlqin2yrs`; atende lotes históricos rotulados. Inferência real exige contrato próprio sem o alvo, pois o desfecho futuro ainda não é conhecido. Os lotes mensais simulados já chegam rotulados, como se o desfecho tivesse maturado.
- O contrato não reproduz todos os filtros da limpeza: não bloqueia especificamente 96/98 nem aplica limites superiores de renda, dívida ou utilização.
- A remoção de outliers também filtra validação e holdout; as métricas descrevem a população retida. O relatório permite acompanhar as exclusões por partição. A simulação atual não reaplica esse filtro após as intervenções, preservando as mudanças introduzidas.
- O lote ruim contém idade 15, renda nula, dívida negativa, duplicatas e coluna ausente. Isso testa qualidade, não simula drift temporal. Os cenários pareados e os lotes mensais com drift passam no contrato, porque são dados válidos com outra distribuição; quem pega esse caso é a detecção de drift.
- A ingestão retorna 2 para falhas de leitura, validação inesperada, escrita ou movimentação. Se já houver lote ou relatório com o mesmo nome na quarentena, preserva a evidência anterior e o arquivo de entrada e retorna 2; é necessário resolver a colisão antes de repetir. No monitoramento, cada execução grava a quarentena em `data/quarantine/<execução>/`, o que evita a colisão; a demonstração da Etapa 1 mantém o comportamento descrito.

## Resultados e evidências da Etapa 1

Execução: `20261114T201315-0300-5fbfe64d`. O [manifesto](../artifacts/stage1/20261114T201315-0300-5fbfe64d/manifest.json) relaciona comandos, códigos de saída, hashes SHA-256, dependências e cópias das fontes e dos resultados em `artifacts/stage1/20261114T201315-0300-5fbfe64d/`. As cópias em `outputs/` preservam este resultado caso os arquivos de trabalho sejam regenerados.

| Partição | Registros | Inadimplentes | Taxa de inadimplência |
| --- | ---: | ---: | ---: |
| Treino / referência | 65.867 | 4.322 | 6,5617% |
| Validação | 16.480 | 1.089 | 6,6080% |
| Holdout | 35.297 | 2.335 | 6,6153% |

| Métrica na validação | Valor |
| --- | ---: |
| AUC ROC | 0,857208 |
| Recall no limiar 0,5 | 0,772268 |
| KS de classificação | 0,567594 |

As métricas foram recalculadas após recarregar o modelo salvo e coincidiram com o JSON publicado. O holdout ficou reservado para os lotes e etapas seguintes; estas métricas foram calculadas somente na validação. Elas descrevem a população retida pela limpeza e não demonstram desempenho temporal ou sob drift.

| Evidência | Resultado observado | Arquivo |
| --- | --- | --- |
| Dependências | Sem conflitos, retorno 0 | [Log](../artifacts/stage1/20261114T201315-0300-5fbfe64d/01_dependencies.txt) |
| Treinamento | Retorno 0, versão completa publicada | [Log](../artifacts/stage1/20261114T201315-0300-5fbfe64d/04_train.txt) |
| Lote válido | 500 linhas, retorno 0 | [Log](../artifacts/stage1/20261114T201315-0300-5fbfe64d/06_ingest_good.txt) |
| Lote inválido | 505 linhas, retorno 1, movido para quarentena | [Log](../artifacts/stage1/20261114T201315-0300-5fbfe64d/07_ingest_bad.txt) |
| Diagnóstico do contrato | 104 entradas de falhas: coluna ausente, idade, nulo, valor negativo e unicidade | [CSV](../data/quarantine/bad_batch_errors.csv) |

As 104 entradas do relatório não representam 104 clientes distintos: um registro pode produzir múltiplas entradas. Antes de repetir o lote inválido com o mesmo nome, arquivar a ocorrência ativa de quarentena ou usar outro nome; a ingestão retorna 2 em colisões e preserva a evidência anterior. As cópias desta execução em `artifacts/stage1/` devem ser mantidas inalteradas para conferência dos hashes.
