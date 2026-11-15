# Data drift e inferência

[Voltar ao README](../README.md)

## Cenário de controle e períodos sintéticos da Etapa 2

**O dataset não fornece uma série temporal real.** `t0`, `t1`, `t2` e `t3` são rótulos de cenários ordenados de um experimento, sem datas observadas, duração em meses ou alegação de evolução econômica real. Todos reutilizam os mesmos registros do holdout, na mesma ordem, para uma comparação pareada que mantém fixa a composição da amostra. Não representam safras independentes de clientes.

O script [simulate_drift.py](../src/simulate_drift.py) lê `data/production/holdout.parquet` e gera:

| Arquivo em `data/production/scenarios/` | Cenário | Fator de utilização de crédito | Chance de acrescentar um atraso de 90 dias |
| --- | --- | ---: | ---: |
| `t0_control.parquet` | Controle: cópia integral do holdout, sem alterações de valores | 1,00 | 0% |
| `t1.parquet` | Primeira intervenção sintética | 1,10 | 10% |
| `t2.parquet` | Segunda intervenção sintética | 1,30 | 25% |
| `t3.parquet` | Terceira intervenção sintética | 1,60 | 40% |

As features alteradas são `RevolvingUtilizationOfUnsecuredLines` e `NumberOfTimes90DaysLate`. O fator de utilização é aplicado diretamente ao holdout original, sem acumulação; valores zero permanecem zero. Para atrasos, sortear um `U ~ Uniforme(0,1)` por linha com seed 42 e acrescentar **um** atraso quando `U` estiver abaixo da probabilidade do período, inclusive para quem tinha zero atrasos. Os mesmos sorteios são reutilizados, formando grupos aninhados; a contagem permanece inteira e não usa o alvo ou os scores. Em t1/t2/t3, respectivamente 3.472/8.889/14.055 linhas recebem o acréscimo; 32.977 linhas têm utilização alterada em cada período. Renda, `DebtRatio`, demais features e o alvo histórico permanecem intactos. Esses parâmetros são testes de estresse, não estimativas econômicas ou evidências de desempenho futuro.

O controle não recebe intervenção; isso não garante ausência de diferenças estatísticas em relação ao treino. O holdout já passou pela limpeza da Etapa 1. Não há nova filtragem por p99 após as intervenções, para não remover a mudança introduzida. O script valida a entrada e todos os cenários com o contrato Pandera antes de publicar os arquivos. A referência de treino, a validação, o holdout e o modelo permanecem intactos; não há treinamento nessa operação.

O `manifest.json` registra entrada e hashes, identificação `credit_utilization_late90_v1`, seed, intervenções, quantidade de linhas alteradas, volumes e limitações. O processo usa todo o holdout sem reamostragem; a seed controla somente o choque de atrasos. Metadados ficam no manifesto, fora dos Parquet, para preservar o contrato estrito. O argumento `--seed` permite outra realização, usando um novo destino.

Na execução local, cada cenário contém 35.297 registros e passa pelo contrato. O [manifesto gerado](../data/production/scenarios/manifest.json) registra os artefatos. Referência, validação, holdout e modelo permanecem preservados. O controle coincide integralmente com o holdout após a leitura dos Parquet. Os cenários anteriores de renda/endividamento foram arquivados para comparação, conforme o [comparativo dos perfis](comparativo-drift.md).

Para gerar os cenários usando o holdout existente:

```bash
make simulate-drift PYTHON=.venv/bin/python
```

O alvo não depende de `prepare`, evitando regenerar a referência. Para outra entrada ou destino:

```bash
.venv/bin/python -m src.simulate_drift --holdout data/production/holdout.parquet --output-dir /tmp/tc04-scenarios
```

O destino deve ser inexistente; reexecuções não sobrescrevem cenários anteriores. Os quatro Parquet e o manifesto são publicados juntos após a gravação em diretório temporário. A inferência e a detecção estatística com Evidently estão descritas a seguir.

## Predições na referência e nos cenários

O script [predict.py](../src/predict.py) carrega `models/current/model.joblib` uma única vez, resolvendo o apontador para a versão publicada, e executa o mesmo modelo na referência e em `t0_control`, `t1`, `t2` e `t3`. Não executa preparação ou treinamento e não altera os arquivos de entrada. Cada dataset é validado com Pandera. As entradas do modelo são selecionadas exclusivamente com `df.loc[:, FEATURE_COLUMNS]`, na ordem do contrato e sem o alvo; essa ordem também é conferida contra os nomes de features do modelo salvo.

```bash
make predict PYTHON=.venv/bin/python
```

Os resultados ficam em `data/predictions/{reference,t0_control,t1,t2,t3}.parquet`. Cada arquivo contém:

| Coluna | Significado |
| --- | --- |
| `row_position` | Posição da linha no Parquet de origem, começando em zero; nenhuma linha é removida ou reordenada |
| `SeriousDlqin2yrs` | Alvo histórico, copiado para permitir análises posteriores; não é entrada do modelo |
| `probability_0` | Probabilidade estimada para a classe 0 |
| `probability_1` | Probabilidade estimada para a classe 1 (inadimplência grave) |
| `predicted_class` | 1 quando `probability_1 >= 0.5`, e 0 caso contrário, seguindo o limiar da avaliação do baseline |

O `manifest.json` de predições registra a versão resolvida e o hash SHA-256 do modelo, horário UTC, ordem de features, regra de classificação, caminhos/hashes das entradas e hashes/quantidades das saídas. Para associar resultados às features, usar a posição das linhas no arquivo de origem indicado no manifesto e conferir seu hash. As predições ficam separadas dos dados para preservar o contrato estrito. O conjunto completo é publicado após concluir todos os arquivos; destino existente causa erro, preservando resultados anteriores.

Para outro destino ou uma versão explícita de modelo:

```bash
.venv/bin/python -m src.predict --model models/runs/32542e11c8c8477ea23d42db115efd0e/model.joblib --output-dir /tmp/tc04-predictions
```

Os argumentos `--reference` e `--scenarios-dir` também permitem indicar outras entradas. As predições na referência são sobre dados usados no treinamento, sem representar avaliação independente. Nos períodos sintéticos, os rótulos históricos permanecem inalterados; resultados não comprovam concept drift nem desempenho futuro real.

Na execução local com o modelo `32542e11c8c8477ea23d42db115efd0e`, foram salvas 65.867 predições na referência e 35.297 em cada cenário, totalizando 207.055. O [manifesto de inferência](../data/predictions/manifest.json) registra a procedência. Após recarregar os cinco Parquet, probabilidades e classes coincidiram exatamente com uma nova inferência do modelo salvo; posições e rótulos também coincidiram com as entradas. Os hashes dos 12 arquivos de dados e modelo envolvidos permaneceram inalterados.

## Detecção de data drift com Evidently AI

O script [detect_drift.py](../src/detect_drift.py) integra o Evidently **0.7.23** usando a API `Dataset`, `DataDefinition`, `Report` e `DataDriftPreset`, conforme a [documentação oficial de relatórios](https://docs.evidentlyai.com/docs/library/report) e de [métodos de drift](https://docs.evidentlyai.com/metrics/customize_data_drift). A execução é local, com telemetria de uso desabilitada por `DO_NOT_TRACK=1`. Não requer conta, servidor ou envio dos dados à plataforma Evidently.

```bash
make detect-drift PYTHON=.venv/bin/python
```

Cada cenário é comparado com `data/reference/reference.parquet`. Entrada e referência passam pelo contrato Pandera; somente `FEATURE_COLUMNS` são entregues ao Evidently. O alvo, `row_position`, probabilidades e classes previstas não participam dessa comparação. Os arquivos de inferência continuam disponíveis para análises posteriores de scores e desempenho, separadas da detecção de data drift.

Os métodos e limiares são explícitos em `src/config.py`, sem depender da seleção automática por cardinalidade do Evidently:

| Grupo | Features | Método | Critério por feature |
| --- | --- | --- | --- |
| Medidas financeiras contínuas | `MonthlyIncome`, `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines` | KS de duas amostras | p-valor < 0,01 |
| Idade e contagens, tratadas como categorias | As outras sete features do contrato | PSI sobre frequências das categorias | PSI ≥ 0,20 |

O PSI categórico usa a união dos valores presentes nas duas amostras e o tratamento de frequências zero do Evidently 0.7.23; não discretiza as medidas financeiras contínuas. O KS evita a escolha de bins para essas medidas. O p-valor não mede a magnitude nem o impacto de negócio, e os testes podem ser sensíveis ao volume e a valores repetidos. Os limiares são escolhas iniciais do experimento, sem correção de múltiplas comparações; não são exigências do enunciado nem critérios universais. O critério global sinaliza drift quando **pelo menos 20% das dez features** apresentam drift; os resultados individuais continuam disponíveis mesmo abaixo desse percentual.

O diretório `reports/drift/` contém quatro relatórios HTML, quatro JSONs nativos do Evidently, `summary.csv` com as 40 comparações por feature e cenário e `manifest.json` com versões das bibliotecas, parâmetros, hashes das entradas/saídas e limitações. Nos relatórios, testes em falha indicam drift detectado, não falha operacional do script. Uma execução concluída retorna 0 mesmo com drift; erros de leitura, contrato ou geração propagam erro e não publicam um conjunto parcial. O destino deve ser inexistente. Para outro conjunto ou destino:

```bash
.venv/bin/python -m src.detect_drift --reference data/reference/reference.parquet --scenarios-dir data/production/scenarios --output-dir /tmp/tc04-drift
```

Resultados da execução sobre os artefatos preservados:

| Cenário | Features com drift | Proporção | Relatório |
| --- | --- | ---: | --- |
| Controle | Nenhuma | 0/10 | [t0_control.html](../reports/drift/t0_control.html) |
| t1 | `RevolvingUtilizationOfUnsecuredLines` | 1/10 | [t1.html](../reports/drift/t1.html) |
| t2 | Utilização de crédito e atrasos de 90 dias | 2/10 | [t2.html](../reports/drift/t2.html) |
| t3 | Utilização de crédito e atrasos de 90 dias | 2/10 | [t3.html](../reports/drift/t3.html) |

O KS da utilização tem p-valor aproximadamente 0,603 no controle e 3,23 × 10⁻²²⁷ em t1; em t2/t3 retorna 0 por precisão numérica. O PSI de atrasos de 90 dias é 0,002781 no controle, 0,129921 em t1, 0,550084 em t2 e 1,097602 em t3. O choque de t1 altera a distribuição dos atrasos, mas não supera o limiar PSI de 0,20. Assim, t1 não aciona o sinal global de 20% das features; t2 e t3 acionam. Os métodos e limiares foram mantidos para permitir comparação. Os números completos estão no [resumo estatístico](../reports/drift/summary.csv) e a procedência no [manifesto](../reports/drift/manifest.json).

A ausência de drift no controle é um resultado observado, sem garantia para outras amostras. A detecção nas duas features confirma mudanças de distribuição introduzidas no experimento; não comprova causa econômica ou concept drift. O [comparativo dos perfis](comparativo-drift.md) mede a sensibilidade do mesmo modelo aos choques com rótulos históricos preservados.
