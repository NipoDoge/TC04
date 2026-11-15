# Etapa 1: qualidade, contratos e baseline

[Voltar ao README](../README.md)

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

- O contrato exige `SeriousDlqin2yrs`; atende lotes históricos rotulados. Inferência real exige contrato próprio sem o alvo, pois o desfecho futuro ainda não é conhecido.
- O contrato não reproduz todos os filtros da limpeza: não bloqueia especificamente 96/98 nem aplica limites superiores de renda, dívida ou utilização.
- A remoção de outliers também filtra validação e holdout; as métricas descrevem a população retida. O relatório permite acompanhar as exclusões por partição. A simulação atual não reaplica esse filtro após as intervenções, preservando as mudanças introduzidas.
- O lote ruim contém idade 15, renda nula, dívida negativa, duplicatas e coluna ausente. Isso testa qualidade, não simula drift temporal.
- A ingestão retorna 2 para falhas de leitura, validação inesperada, escrita ou movimentação. Se já houver lote ou relatório com o mesmo nome na quarentena, preserva a evidência anterior e o arquivo de entrada e retorna 2; é necessário resolver a colisão antes de repetir. O armazenamento de múltiplas ocorrências com identificadores únicos continua pendente.

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
