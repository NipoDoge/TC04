# TC04 | Governança e confiabilidade de um modelo de crédito

Projeto acadêmico da Fase 4 do MLET/FIAP. O cenário é uma fintech cujo modelo de Credit Scoring pode sofrer degradação silenciosa diante de mudanças econômicas e do perfil dos clientes. O objetivo é validar dados, investigar mudanças de distribuição e definir como proteger dados e responder à degradação.

Este README documenta a **Etapa 1 executada**, a **simulação, as predições e os relatórios de data drift executados da Etapa 2** e a **proposta de governança da Etapa 4**. A observabilidade contínua da Etapa 3 ainda está pendente. O experimento implementado cobre data drift; não demonstra concept drift ou degradação real de desempenho. As políticas de operação futura são distinguidas dos controles executados; não constituem evidência de conformidade integral. O projeto não realiza concessões reais de crédito.

## 1. Escopo e evidências disponíveis

| Componente | Evidência no repositório | Situação |
| --- | --- | --- |
| Exploração e avaliação inicial de seleção | [Notebook](notebooks/01_exploracao.ipynb) | Exploração auxiliar; saídas antigas removidas; leitura sem modificar dados |
| Limpeza e separação treino/validação/produção | [prepare.py](src/prepare.py), relatório de limpeza e arquivos Parquet | Implementados; percentis ajustados somente no treino; divisão aleatória, sem simulação temporal |
| Contrato de dados | [contract.py](src/contract.py) | Implementado em Pandera |
| Bloqueio e quarentena | [ingest.py](src/ingest.py) | Implementados para falhas de validação do contrato |
| Lotes de demonstração | [Log de aprovação](artifacts/stage1/20261114T201315-0300-5fbfe64d/06_ingest_good.txt), [log de bloqueio](artifacts/stage1/20261114T201315-0300-5fbfe64d/07_ingest_bad.txt) e [falhas](data/quarantine/bad_batch_errors.csv) | Executados: 500 linhas aprovadas e lote de 505 linhas em quarentena |
| Treinamento e avaliação | [Modelo](models/runs/32542e11c8c8477ea23d42db115efd0e/model.joblib), [métricas](models/runs/32542e11c8c8477ea23d42db115efd0e/metrics.json) e [manifesto da execução](artifacts/stage1/20261114T201315-0300-5fbfe64d/manifest.json) | Baseline treinado, métricas conferidas após recarregar o modelo e publicação por versão |
| Simulação de data drift | [simulate_drift.py](src/simulate_drift.py): controle e três períodos sintéticos pareados a partir do holdout | Implementada; não representa série temporal real |
| Predições da Etapa 2 | [predict.py](src/predict.py): inferência com o modelo salvo na referência e nos quatro cenários | Implementadas; probabilidades e classes em arquivos separados |
| Detecção de data drift | [detect_drift.py](src/detect_drift.py), [resumo estatístico](reports/drift/summary.csv) e relatórios HTML do Evidently | Executada: controle sem detecção; renda e dívida com drift em t1, t2 e t3 |
| Observabilidade centralizada | Logs locais na ingestão; sem dashboard ou rastreamento identificado | Parcial |
| Governança e causalidade | Este README | Política proposta e hipóteses documentadas |
| Vídeo STAR | Seção 10 | Gravação e link pendentes |

## 2. Dados, modelo e arquitetura

### Dados e finalidade

Os arquivos locais correspondem ao conjunto **Give Me Some Credit**, com 150.000 registros no `cs-training.csv`. A finalidade acadêmica é estimar o risco de inadimplência e estudar qualidade e confiabilidade. A origem de obtenção, a versão e os termos de uso devem ser registrados antes da entrega; a presença dos arquivos no Git não comprova autorização de redistribuição.

O alvo `SeriousDlqin2yrs` representa inadimplência grave no horizonte de dois anos. As dez variáveis preditoras são:

| Grupo | Colunas | Risco de privacidade e uso |
| --- | --- | --- |
| Perfil | `age`, `NumberOfDependents` | Combinações podem favorecer identificação e segmentação discriminatória |
| Situação financeira | `MonthlyIncome`, `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines` | Informações financeiras confidenciais; possíveis proxies socioeconômicos |
| Histórico de crédito | `NumberOfTime30-59DaysPastDueNotWorse`, `NumberOfTimes90DaysLate`, `NumberOfTime60-89DaysPastDueNotWorse`, `NumberOfOpenCreditLinesAndLoans`, `NumberRealEstateLoansOrLines` | Perfil financeiro e risco de exposição de comportamento |
| Desfecho | `SeriousDlqin2yrs` | Informação de inadimplência; usada no treino e na avaliação |

Não há colunas de nome, CPF, telefone ou e-mail no arquivo de treino inspecionado. A coluna `Unnamed: 0`, um índice do arquivo, é removida na preparação. **Essa remoção não comprova anonimização**: não há avaliação de reidentificação ou técnica de anonimização implementada.

Dados financeiros e idade podem ser dados pessoais quando vinculados a pessoa identificada ou identificável; não são automaticamente dados pessoais sensíveis na definição legal. Não foram identificadas colunas explícitas de saúde, biometria, religião ou origem racial. A distinção segue o [Glossário da ANPD](https://www.gov.br/anpd/pt-br/documentos-e-publicacoes/glossario-anpd).

### Fluxo implementado

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
            -> src.simulate_drift: controle e intervenções pareadas em renda/dívida
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
```

[config.py](src/config.py) centraliza alvo, seed, limiar de classificação, fatores dos períodos e critérios de drift. [artifacts.py](src/artifacts.py) centraliza snapshots de leitura e hashes SHA-256 usados na simulação, na inferência e nos relatórios de drift. A ordem das features continua definida apenas em [contract.py](src/contract.py).

Os percentuais são anteriores à filtragem de outliers; as proporções finais podem variar. Os scripts criam os diretórios de saída automaticamente. A preparação atual já foi executada; ao reproduzir, executar `python -m src.prepare` antes de treinar para manter as partições e os limites da mesma preparação. A ingestão aprovada somente valida o lote: não o encaminha automaticamente ao modelo. O treinamento valida referência e validação com o contrato e exige registros das duas classes em ambas antes de criar o modelo. Reprovação retorna 1; falha operacional retorna 2. Uma execução bloqueada preserva eventuais artefatos anteriores.

Modelo e métricas são gravados em um diretório temporário exclusivo dentro de `models/`. Somente após concluir ambas as gravações, a versão é movida para `models/runs/<execução>` e o link simbólico `models/current` é substituído atomicamente. Os caminhos publicados são `models/current/model.joblib` e `models/current/metrics.json`. Falhas de gravação ou publicação retornam 2 e preservam a versão anterior; temporários e versões não publicadas são removidos nas falhas tratadas. Versões publicadas anteriores permanecem disponíveis em `models/runs/`.

Para ler modelo e métricas da mesma execução enquanto houver novos treinamentos, resolver o apontador uma única vez com `run_dir = Path("models/current").resolve()` e abrir ambos a partir de `run_dir`. Todos os consumidores do projeto usam a estrutura publicada em `models/current` ou uma versão explícita de `models/runs`. A publicação por links simbólicos foi validada no macOS e requer suporte do filesystem a esses links.

O modelo é um `XGBClassifier` com 300 estimadores, profundidade 4, taxa de aprendizado 0,05 e ponderação da classe positiva. A avaliação calcula AUC ROC, recall com limiar 0,5 e KS como separação máxima entre TPR e FPR. **Esse KS de classificação não é um teste de drift entre datasets.** O limiar 0,5 é usado na avaliação; uma política de aprovação de crédito não foi implementada.

## 3. Qualidade de dados e limites do contrato

O contrato exige pelo menos um registro; valores finitos em todas as colunas numéricas (rejeita `+inf` e `-inf`); alvo binário; idade inteira entre 18 e 100, inclusive; renda não nula e não negativa; demais medidas e contagens não negativas; colunas e tipos definidos; ausência de colunas extras e de linhas integralmente duplicadas. Os campos não marcados como anuláveis também rejeitam nulos por padrão. Não há coerção de tipos configurada. Lotes vazios e números infinitos são violações do contrato, com retorno 1 na ingestão e no treinamento.

A lista `FEATURE_COLUMNS`, definida no contrato, determina a ordem das entradas do modelo. O treinamento seleciona as mesmas features nessa ordem na referência e na validação, permitindo arquivos com colunas em outra ordem. A lista fica registrada também nos nomes das features do XGBoost serializado.

A preparação remove duplicatas, idades fora do intervalo, registros sem renda ou dependentes, rendas negativas e códigos 96/98 nas colunas de atraso. Separa 30% para holdout e divide os 70% restantes em treino/validação na proporção 80/20. Calcula os percentis 99 de `DebtRatio` e de utilização de crédito exclusivamente no treino, antes de qualquer filtro estatístico. Aplica os mesmos limites às três partições, sem recalculá-los na validação ou no holdout. Os limites são salvos em `data/reference/outlier_limits.json`, e o relatório registra exclusões e tamanhos por partição.

O [relatório da preparação executada](data/reference/cleaning_report.json) registra 609 duplicatas, 14 exclusões por idade, 29.214 por renda/dependentes ausentes, nenhuma por renda negativa e 145 pelos códigos 96/98. Restaram 120.018 registros antes dos filtros estatísticos. Os limites calculados somente no treino foram `DebtRatio ≤ 672,76` e `RevolvingUtilizationOfUnsecuredLines ≤ 1,09213079212`; os valores completos estão em [outlier_limits.json](data/reference/outlier_limits.json).

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

## 4. Plano de proteção de dados e base legal

### Finalidades e enquadramento

Na hipótese de operação real da fintech, a base legal proposta para avaliação de risco é **proteção do crédito, art. 7º, X, da LGPD**. O controlador deverá validar sua adequação à finalidade e aos dados tratados. Essa hipótese não autoriza automaticamente o reuso acadêmico ou a publicação da base. Finalidade acadêmica também exige observar os arts. 7º e 11, conforme art. 4º, II, b. Fonte: [texto atualizado da LGPD](https://www2.camara.leg.br/legin/fed/lei/2018/lei-13709-14-agosto-2018-787077-normaatualizada-pl.html).

Para este estudo, permanece pendente documentar a procedência, os termos de uso e o enquadramento do tratamento. O caráter público de uma fonte não afasta, por si só, os cuidados com dados pessoais. Fonte: [perguntas frequentes da ANPD](https://www.gov.br/anpd/pt-br/acesso-a-informacao/perguntas-frequentes).

### Privacy by Design e by Default

| Medida | Situação atual | Política proposta |
| --- | --- | --- |
| Minimização | Remoção do índice; contrato recusa colunas extras | Admitir somente atributos justificados pela finalidade; não incluir identificadores diretos nas features |
| Segregação | Diretórios de dados brutos, referência e holdout | Separar ambientes, identidades e permissões; manter chaves de associação fora da área analítica |
| Acesso | Sem controle de acesso implementado no projeto | Menor privilégio; dados detalhados restritos à equipe autorizada; revisão ao trocar integrantes |
| Armazenamento e transporte | Arquivos locais; sem configuração de criptografia no código | Armazenamento e backups criptografados; TLS nas transferências; segredos fora do Git |
| Publicação | CSV e Parquet estão versionados | Avaliar termos e risco; preferir amostras sintéticas e resultados agregados; tratar histórico Git se houver remoção necessária |
| Logs e relatórios | Log de lote e CSV de falhas | Não registrar linhas de clientes; restringir quarentena e sanitizar valores antes de compartilhar relatórios ou vídeo |
| Descarte | Sem rotina implementada | Aplicar a tabela de retenção, incluindo cópias, backups e histórico sob controle da equipe |

O CSV de falhas pode conter valores rejeitados. Quarentena é uma área restrita de diagnóstico, não um mecanismo de anonimização. A ausência de identificadores diretos também não autoriza divulgar registros financeiros individuais.

### Direitos e decisões automatizadas

Em uma operação real, disponibilizar canal de privacidade para acesso, correção e demais solicitações aplicáveis; manter registro de atendimento. Solicitações de revisão de decisões automatizadas devem receber procedimento documentado, com informação sobre critérios e parâmetros, respeitados os segredos comercial e industrial. Fonte: [direitos dos titulares, ANPD](https://www.gov.br/anpd/pt-br/assuntos/titular-de-dados-1/direito-dos-titulares).

Como política do projeto para uma implantação futura, decisões contestadas ou com alerta crítico serão encaminhadas a análise humana, com justificativa registrada. O protótipo atual não possui cadastro de titulares, canal de atendimento ou sistema de decisão de crédito.

### Retenção e descarte

Os prazos abaixo são **propostas internas para o ambiente acadêmico**, contados conforme cada linha, e não prazos legais universais. Na operação real, o controlador deverá definir uma matriz própria de obrigações e necessidades antes da coleta.

| Categoria | Retenção proposta | Destino e responsável |
| --- | --- | --- |
| Dados brutos e derivados locais | Até 90 dias após a avaliação final | Engenharia de Dados elimina cópias de trabalho e confirma tratamento das cópias compartilhadas |
| Lotes em quarentena e CSVs com valores | Até 30 dias após a triagem | Engenharia de Dados corrige/reprocessa ou elimina; guarda somente resumo agregado |
| Logs sem valores individuais | 180 dias após a execução | MLOps elimina após consolidar indicadores de auditoria |
| Relatórios agregados e documentação | 12 meses após a avaliação final | Responsável pelo projeto revisa necessidade de conservação |
| Modelos e metadados | Até 12 meses após a substituição | Ciência de Dados elimina versões sem necessidade de auditoria; acesso continua restrito |

Revisar mensalmente os vencimentos; registrar categoria, data, executor e resultado do descarte. Incluir backups no ciclo de expiração. Exceções exigem motivo, aprovador e nova data de revisão. Um arquivo excluído da pasta de trabalho pode continuar no Git ou em clones: a limpeza do histórico e o alcance das cópias precisam ser avaliados separadamente. Nada disso está automatizado hoje.

## 5. Vieses e uso responsável

O dataset de treino tem 10.026 positivos em 150.000 registros, uma taxa de 6,684%. O treinamento usa `scale_pos_weight` para lidar com o desequilíbrio; isso não demonstra equidade entre grupos.

A exclusão de registros sem renda ou dependentes pode mudar a população atendida. A análise do CSV bruto, reproduzível pelo notebook auxiliar, encontra taxas de inadimplência de aproximadamente 5,61% para renda ausente e 6,95% para renda preenchida, indicando seleção potencialmente relevante. Os filtros de percentil também podem remover clientes de maior risco. Esses resultados justificam investigação, sem comprovar discriminação ou causalidade.

Antes de recomendar uso operacional, a política proposta é:

1. Comparar proporções de exclusão, distribuição e inadimplência antes/depois da limpeza por faixas de idade e renda.
2. Avaliar AUC, recall, FPR, FNR e calibração por segmento, com tamanho amostral e incerteza; explicitar que o positivo representa inadimplência.
3. Comparar exclusão de renda ausente com imputação ajustada no treino e indicador de ausência, avaliando desempenho e seleção.
4. Revisar o uso de idade, dependentes e proxies socioeconômicos; justificar necessidade e alternativas.
5. Aprovar o limiar com base nos custos de inadimplência e de recusa indevida. Não converter automaticamente a probabilidade em aprovação/reprovação sem política validada.

Não há atributos explícitos suficientes para auditar todos os grupos protegidos. Sua ausência limita conclusões sobre equidade. Eventual coleta para auditoria exige finalidade, base legal e controles próprios; não deve ser feita indiscriminadamente.

## 6. Causalidade: diagnóstico e hipóteses

**Não há evidência de degradação temporal real ou de causa econômica no código atual.** O holdout é uma amostra aleatória; o gerador de lote ruim introduz violações de contrato. O script `src.simulate_drift` cria cenários sintéticos de data drift a partir do holdout e `src.detect_drift` identifica as alterações por testes estatísticos e gera relatórios do Evidently. Não há experimento causal ou simulação de concept drift implementados.

O mecanismo econômico a investigar é: mudança econômica pode afetar renda e endividamento, que podem alterar o risco de inadimplência; mudanças no público atendido e na política de aprovação também podem alterar a população observada. Essas relações são hipóteses de domínio, não conclusões obtidas neste projeto.

| Hipótese | Evidência necessária | Ação se corroborada |
| --- | --- | --- |
| Erro na coleta ou transformação | Falhas no contrato, mudança de unidade, versão da fonte e logs | Bloquear lote, corrigir origem e reprocessar |
| Data drift: mudança em P(X) | Comparação referência/análise, por exemplo em renda e endividamento, com testes e magnitude | Investigar origem, população e impacto sobre predições |
| Concept drift: mudança em P(Y\|X) | Rótulos maturados, queda de desempenho/calibração em condições comparáveis e análise de alternativas | Avaliar recalibração ou retreinamento |
| Seleção causada pela limpeza | Comparação entre incluídos/excluídos e alternativas de tratamento | Rever filtros e repetir avaliação por segmento |
| Efeito econômico sobre inadimplência | Dados temporais, fatores de confusão e desenho causal defensável | Submeter a hipótese a análise específica antes de atribuir causa |

PSI ou um teste KS de duas amostras pode apontar mudança de distribuição; não identifica sua causa nem comprova concept drift. Queda de performance também exige investigar qualidade, seleção, mudança de política e maturação do alvo.

A etapa experimental já registra intervenções em duas features e sua intensidade, compara cenários pareados e executa o mesmo modelo. A simulação usa todo o holdout sem aleatoriedade; a seed 42 se aplica à preparação e ao treinamento. Alterar apenas X mantendo rótulos antigos permite estudar sensibilidade das previsões, mas não demonstra desempenho real após uma mudança econômica. Um cenário de concept drift exige explicitar a alteração sintética do mecanismo do alvo ou obter novos desfechos observados.

Como o alvo tem horizonte de dois anos, produção real requer política de maturação e acompanhamento de rótulos. Sem desfechos disponíveis, monitorar dados e scores e declarar a limitação de avaliação do modelo. O dataset local não permite reconstruir uma série temporal real de concessões.

### Cenário de controle e períodos sintéticos da Etapa 2

**O dataset não fornece uma série temporal real.** `t0`, `t1`, `t2` e `t3` são rótulos de cenários ordenados de um experimento, sem datas observadas, duração em meses ou alegação de evolução econômica real. Todos reutilizam os mesmos registros do holdout, na mesma ordem, para uma comparação pareada que mantém fixa a composição da amostra. Não representam safras independentes de clientes.

O script [simulate_drift.py](src/simulate_drift.py) lê `data/production/holdout.parquet` e gera:

| Arquivo em `data/production/scenarios/` | Cenário | Fator de MonthlyIncome | Fator de DebtRatio |
| --- | --- | ---: | ---: |
| `t0_control.parquet` | Controle: cópia integral do holdout, sem alterações de valores | 1,00 | 1,00 |
| `t1.parquet` | Primeira intervenção sintética | 0,90 | 1,10 |
| `t2.parquet` | Segunda intervenção sintética | 0,75 | 1,30 |
| `t3.parquet` | Terceira intervenção sintética | 0,60 | 1,60 |

Cada fator é aplicado diretamente ao holdout original, sem acumular intervenções entre períodos. São parâmetros ilustrativos para reduzir renda e aumentar endividamento; não são estimativas econômicas nem recalculam contabilmente `DebtRatio` a partir da renda. Valores zero permanecem zero. Todas as demais features e o alvo histórico `SeriousDlqin2yrs` são preservados. Manter esses rótulos após alterar as features permite estudar sensibilidade, mas não comprova concept drift nem desempenho futuro real.

O controle não recebe intervenção; isso não garante ausência de diferenças estatísticas em relação ao treino. O holdout já passou pela limpeza da Etapa 1. Não há nova filtragem por p99 após as intervenções, para não remover a mudança introduzida. O script valida a entrada e todos os cenários com o contrato Pandera antes de publicar os arquivos. A referência de treino, a validação, o holdout e o modelo permanecem intactos; não há treinamento nessa operação.

O `manifest.json` registra o caminho e o hash SHA-256 da entrada, fatores, número de registros, hashes dos arquivos gerados e limitações do experimento. O processo usa todo o holdout e não contém amostragem ou aleatoriedade; não requer seed. Metadados ficam no manifesto, fora dos Parquet, para preservar o contrato estrito.

Na execução local, cada cenário contém 35.297 registros e passa pelo contrato. O [manifesto gerado](data/production/scenarios/manifest.json) registra os artefatos. A verificação SHA-256 antes/depois confirmou que os sete arquivos de referência, validação, holdout e modelo/métricas não foram alterados. O controle também coincidiu integralmente com o holdout após a leitura dos Parquet.

Para gerar os cenários usando o holdout existente:

```bash
make simulate-drift PYTHON=.venv/bin/python
```

O alvo não depende de `prepare`, evitando regenerar a referência. Para outra entrada ou destino:

```bash
.venv/bin/python -m src.simulate_drift --holdout data/production/holdout.parquet --output-dir /tmp/tc04-scenarios
```

O destino deve ser inexistente; reexecuções não sobrescrevem cenários anteriores. Os quatro Parquet e o manifesto são publicados juntos após a gravação em diretório temporário. A inferência e a detecção estatística com Evidently estão descritas a seguir.

### Predições na referência e nos cenários

O script [predict.py](src/predict.py) carrega `models/current/model.joblib` uma única vez, resolvendo o apontador para a versão publicada, e executa o mesmo modelo na referência e em `t0_control`, `t1`, `t2` e `t3`. Não executa preparação ou treinamento e não altera os arquivos de entrada. Cada dataset é validado com Pandera. As entradas do modelo são selecionadas exclusivamente com `df.loc[:, FEATURE_COLUMNS]`, na ordem do contrato e sem o alvo; essa ordem também é conferida contra os nomes de features do modelo salvo.

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

Na execução local com o modelo `32542e11c8c8477ea23d42db115efd0e`, foram salvas 65.867 predições na referência e 35.297 em cada cenário, totalizando 207.055. O [manifesto de inferência](data/predictions/manifest.json) registra a procedência. Após recarregar os cinco Parquet, probabilidades e classes coincidiram exatamente com uma nova inferência do modelo salvo; posições e rótulos também coincidiram com as entradas. Os hashes dos 12 arquivos de dados e modelo envolvidos permaneceram inalterados.

### Detecção de data drift com Evidently AI

O script [detect_drift.py](src/detect_drift.py) integra o Evidently **0.7.23** usando a API `Dataset`, `DataDefinition`, `Report` e `DataDriftPreset`, conforme a [documentação oficial de relatórios](https://docs.evidentlyai.com/docs/library/report) e de [métodos de drift](https://docs.evidentlyai.com/metrics/customize_data_drift). A execução é local, com telemetria de uso desabilitada por `DO_NOT_TRACK=1`. Não requer conta, servidor ou envio dos dados à plataforma Evidently.

```bash
make detect-drift PYTHON=.venv/bin/python
```

Cada cenário é comparado com `data/reference/reference.parquet`. Entrada e referência passam pelo contrato Pandera; somente `FEATURE_COLUMNS` são entregues ao Evidently. O alvo, `row_position`, probabilidades e classes previstas não participam dessa comparação. Os arquivos de inferência continuam disponíveis para análises posteriores de scores e desempenho, separadas da detecção de data drift.

Os métodos e limiares são explícitos em `src/config.py`, sem depender da seleção automática por cardinalidade do Evidently:

| Grupo | Features | Método | Critério por feature |
| --- | --- | --- | --- |
| Medidas financeiras contínuas | `MonthlyIncome`, `DebtRatio`, `RevolvingUtilizationOfUnsecuredLines` | KS de duas amostras | p-valor < 0,01 |
| Idade e contagens, tratadas como categorias | As outras sete features do contrato | PSI sobre frequências das categorias | PSI ≥ 0,20 |

O PSI categórico usa a união dos valores presentes nas duas amostras e o tratamento de frequências zero do Evidently 0.7.23; não discretiza renda ou dívida. O KS evita a escolha de bins para as medidas financeiras. O p-valor não mede a magnitude nem o impacto de negócio, e os testes podem ser sensíveis ao volume e a valores repetidos. Os limiares são escolhas iniciais do experimento, sem correção de múltiplas comparações; não são exigências do enunciado nem critérios universais. O critério global sinaliza drift quando **pelo menos 20% das dez features** apresentam drift; os resultados individuais continuam disponíveis mesmo abaixo desse percentual.

O diretório `reports/drift/` contém quatro relatórios HTML, quatro JSONs nativos do Evidently, `summary.csv` com as 40 comparações por feature e cenário e `manifest.json` com versões das bibliotecas, parâmetros, hashes das entradas/saídas e limitações. Nos relatórios, testes em falha indicam drift detectado, não falha operacional do script. Uma execução concluída retorna 0 mesmo com drift; erros de leitura, contrato ou geração propagam erro e não publicam um conjunto parcial. O destino deve ser inexistente. Para outro conjunto ou destino:

```bash
.venv/bin/python -m src.detect_drift --reference data/reference/reference.parquet --scenarios-dir data/production/scenarios --output-dir /tmp/tc04-drift
```

Resultados da execução sobre os artefatos preservados:

| Cenário | Features com drift | Proporção | Relatório |
| --- | --- | ---: | --- |
| Controle | Nenhuma | 0/10 | [t0_control.html](reports/drift/t0_control.html) |
| t1 | `MonthlyIncome`, `DebtRatio` | 2/10 | [t1.html](reports/drift/t1.html) |
| t2 | `MonthlyIncome`, `DebtRatio` | 2/10 | [t2.html](reports/drift/t2.html) |
| t3 | `MonthlyIncome`, `DebtRatio` | 2/10 | [t3.html](reports/drift/t3.html) |

No controle, os p-valores de renda e dívida foram aproximadamente 0,925 e 0,588; em t1, aproximadamente 1,01 × 10⁻¹²³ e 1,91 × 10⁻⁶⁰. Em t2 e t3, a biblioteca retornou 0 por precisão numérica para esses p-valores extremamente pequenos. Os números completos estão no [resumo estatístico](reports/drift/summary.csv) e a procedência no [manifesto](reports/drift/manifest.json). Os 54 arquivos existentes de dados, modelos e evidências permaneceram inalterados durante a geração, verificados por SHA-256.

A ausência de drift no controle é um resultado observado, sem garantia para outras amostras. A detecção em renda e dívida confirma mudanças de distribuição introduzidas no experimento; não comprova queda de desempenho, causa econômica ou concept drift. A menção a concept drift no contexto geral do enunciado requer um experimento adicional sobre `P(Y|X)`, ainda não implementado.

## 7. Monitoramento e resposta à degradação

Esta seção propõe a operação contínua e a resposta aos sinais; agendamento, centralização e acionamento operacional ainda não estão implementados. A detecção offline de data drift já está executada na seção 6. Os limiares são pontos iniciais e devem ser calibrados; não são exigências do enunciado.

| Sinal | Frequência proposta | Critério inicial | Resposta |
| --- | --- | --- | --- |
| Qualidade do contrato | Cada lote | Qualquer violação | Bloquear e colocar em quarentena; corrigir e revalidar |
| Status, volume, duração e erros | Cada execução | Falha ou lote vazio inesperado; duração acima de 2 vezes a mediana das últimas 20 execuções válidas | Abrir incidente e investigar dependências e origem |
| Drift de features | Cada janela de análise com pelo menos 500 registros | KS com p < 0,01 nas três medidas financeiras ou PSI ≥ 0,20 nas demais categorias; sinal global em ≥ 20% das features | Alertar; examinar magnitude, segmentos e qualidade antes de decidir |
| Performance com rótulos maturados | Cada janela elegível | Queda absoluta de AUC ≥ 0,05 ou recall ≥ 0,10 contra baseline, usando o mesmo limiar | Revisar incerteza, população e política; avaliar candidato |
| Equidade | Cada avaliação e troca de modelo | Aumento absoluto de FNR ou FPR ≥ 0,10 em segmento com amostra suficiente | Revisão conjunta de Ciência de Dados, Risco e Privacidade |

O baseline da Etapa 1 está preservado em `models/runs/32542e11c8c8477ea23d42db115efd0e`, com AUC 0,857208, recall no limiar 0,5 de 0,772268 e KS 0,567594 na validação. Esses valores são uma referência inicial; os alertas e o acompanhamento por janelas ainda não foram implementados. O volume mínimo acima é uma regra operacional inicial; não substitui avaliação de poder estatístico e incerteza por segmento.

Fluxo de resposta proposto:

1. Registrar incidente com horário, lote, versão do código, dados e modelo; manter amostra restrita somente se necessária.
2. Engenharia verifica integridade e pipeline; Ciência de Dados verifica distribuição, rótulos e desempenho; Risco avalia impacto nas decisões.
3. Em falha de qualidade, interromper o lote. Em degradação crítica, suspender a decisão automática ou encaminhar casos à análise humana, mediante responsável operacional.
4. Avaliar correção, recalibração ou retreinamento. Drift isolado não deve promover automaticamente um novo modelo.
5. Comparar candidato com versão vigente em dados independentes, incluindo segmentos e calibração; registrar aprovação de Risco e responsável técnico.
6. Publicar versão com monitoramento reforçado; manter versão anterior para rollback e registrar encerramento e prevenção da recorrência.

Em suspeita de exposição de dados, conter acessos, preservar evidências restritas e acionar Segurança e o encarregado. A avaliação de impacto e eventuais comunicações deve seguir a regulamentação aplicável. Não há resposta a incidentes implementada no protótipo.

## 8. Responsabilidades e rastreabilidade

Os papéis abaixo deverão ser atribuídos nominalmente pelo grupo antes da entrega; não há pessoas ou contatos presumidos.

| Papel | Responsabilidade proposta |
| --- | --- |
| Responsável pelo projeto / Risco | Aprovar finalidade, política de decisão, critérios de promoção e suspensão |
| Engenharia de Dados | Contratos, qualidade, linhagem, quarentena e descarte |
| Ciência de Dados | Modelo, análise de drift, hipóteses causais e vieses |
| MLOps / Segurança | Execução, acessos, logs, versionamento, incidentes e rollback |
| Privacidade / encarregado, na operação real | Validar plano de privacidade, apoiar direitos e avaliar incidentes |

Para cada execução, a política é guardar identificador, horário, commit, ambiente e dependências, hash e origem dos dados, seed quando houver aleatoriedade, versão do contrato, exclusões, métricas, parâmetros e artefato do modelo. A execução preservada da Etapa 1 já inclui manifesto com horários, commit-base, hashes do CSV de entrada, fontes e saídas, versões das dependências, seed, comandos e códigos de saída. Inclui cópias dos scripts e testes usados, pois havia alterações locais sobre o commit-base; o manifesto não atribui esses ajustes ao commit-base. Simulação e inferência já geram manifestos próprios de entradas, saídas e parâmetros; a inferência também registra modelo e horário. A coleta de todos os campos dessa política em cada etapa ainda não está automatizada. Em mudanças operacionais de modelo, comparação, aprovação e motivo continuam sendo políticas propostas.

## 9. Reprodução e ressalvas técnicas

Executar a partir da raiz, em ambiente virtual com **Python 3.12**, após instalar as dependências de [requirements.txt](requirements.txt). As versões fixadas exigem Python 3.12 ou superior. O arquivo contém oito dependências diretas do pipeline e dos testes:

| Dependência | Uso |
| --- | --- |
| NumPy | Validação de finitude, probabilidades e operações numéricas |
| pandas | Leitura, transformação e escrita de datasets |
| Pandera com extra `pandas` | Contrato de qualidade |
| PyArrow | Engine de leitura e escrita de Parquet pelo pandas |
| scikit-learn | Separação estratificada e métricas de classificação |
| XGBoost | Treinamento e execução do classificador salvo |
| joblib | Persistência e leitura do modelo |
| Evidently AI | Testes estatísticos e relatórios HTML/JSON de data drift |

O Evidently 0.7.23 falha ao importar com pandas 3.1.0 por usar uma função interna removida. A integração fixa **pandas 2.3.3**, adiciona o Evidently e mantém as versões das outras seis dependências diretas do pipeline. A preparação, as predições e o baseline foram reproduzidos com os mesmos valores após esse ajuste. Parquet regenerado pode ter hash binário diferente devido à versão do pandas nos metadados; comparar o conteúdo e guardar o novo hash no manifesto. Os artefatos preservados da execução com pandas 3.1.0 continuam inalterados.

Dependências transitivas, como SciPy, Pydantic e Plotly, continuam necessárias às bibliotecas e são resolvidas pelo pip; não são listadas como dependências diretas. O pacote de compatibilidade `pathlib` foi removido porque o código usa a biblioteca padrão. Ferramentas de servidor Jupyter, Prometheus e demais pacotes sem uso direto foram retirados da lista de execução; bibliotecas trazidas transitivamente pelo Evidently permanecem necessárias à sua instalação. No Linux, o próprio XGBoost declara a dependência transitiva de NCCL; não há configuração de GPU no projeto. A limpeza do arquivo não desinstala pacotes de ambientes virtuais já existentes.

O arquivo fixa as versões diretas; não é um lockfile completo de dependências transitivas. O ambiente integral da execução histórica permanece registrado em `artifacts/stage1/` para auditoria.

O notebook é auxiliar e não faz parte da execução do pipeline. Para executá-lo em um editor compatível com Jupyter, instalar opcionalmente `python -m pip install -r requirements-notebooks.txt`, que adiciona o IPython kernel às dependências do projeto. A raiz é localizada a partir da pasta do kernel; execute o notebook dentro do projeto. Suas saídas antigas foram removidas para evitar resultados e caminhos de outras máquinas.

No macOS, instalar também o runtime OpenMP exigido pelo XGBoost:

```bash
brew install libomp
```

A execução definitiva da Etapa 1 foi realizada no projeto em **14/11/2026, às 20:13, no fuso America/Sao_Paulo**, usando macOS ARM64 e Python 3.12.14. O `pip check` não encontrou conflitos, os 32 testes passaram e os artefatos estão preservados. As versões completas do ambiente constam em [environment.txt](artifacts/stage1/20261114T201315-0300-5fbfe64d/environment.txt).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m src.prepare
python -m src.train
python -m src.make_bad_batch
python -m src.ingest data/incoming/good_batch.parquet
python -m src.ingest data/incoming/bad_batch.parquet
```

O resultado esperado da última chamada é código de saída 1, lote movido para quarentena e arquivo `bad_batch_errors.csv`. A chamada do lote válido deve retornar 0, se os tipos e valores satisfizerem o contrato. Erros operacionais retornam 2, incluindo arquivo ausente/ilegível ou falha ao registrar a quarentena. O código 1 somente é retornado após salvar o relatório e movimentar o lote rejeitado.

Com o ambiente ativado, `make` exibe os alvos sem executar o pipeline. A sequência explícita é `make prepare`, `make train`, `make simulate-drift`, `make predict` e `make detect-drift`; `train` usa as partições existentes e não chama `prepare`. Para indicar o interpretador, usar `PYTHON=.venv/bin/python`. Simulação, inferência e detecção exigem destinos inexistentes; para novas execuções sem substituir as evidências atuais, usar os argumentos `--output-dir` descritos na seção 6. A detecção usa diretamente a referência e os cenários, sem depender das predições. Se regenerar partições ou trocar o modelo, gerar novos cenários, predições e relatórios antes de compará-los; manifestos antigos identificam as versões anteriores pelos hashes.

Os alvos `make demo-bad` e `make demo-good` demonstram o contrato da Etapa 1. `demo-bad` gera os dois lotes e somente passa se a ingestão retornar 1, o lote sair da entrada e os arquivos de quarentena e relatório existirem e não estiverem vazios. Deve preceder `demo-good`. O gerador valida o holdout e exige pelo menos 500 registros; sua importação não lê nem grava arquivos. Para gerar lotes em outro local, usar `python -m src.make_bad_batch --output-dir /tmp/tc04-batches`. Cada script cria os diretórios necessários às suas saídas. Reexecutar preparação ou geração dos lotes sobrescreve suas saídas; o treinamento cria uma nova versão e troca o apontador publicado. A ingestão rejeita colisões na quarentena. Preservar as evidências e resolver colisões antes de repetir a demonstração.

Os testes são executados com `make test` ou `python -m unittest discover -s tests -v`. Verificam limites calculados somente no treino, invariância a alterações na validação/holdout, aplicação de limites fixos, separação das partições e criação das saídas. Cobrem bloqueio do treinamento, códigos de saída da ingestão, erros operacionais, colisões na quarentena e falso sucesso no Makefile. Incluem rejeição de lotes vazios e valores infinitos e invariância à ordem das colunas. Os testes de publicação simulam falhas e conferem preservação da versão anterior e limpeza de temporários. Simulação e inferência têm testes de pareamento, preservação de entradas, hashes, ordem das features, exclusão do alvo, limiar 0,5, probabilidades inválidas e publicação sem resultados parciais. Também há cobertura para importação do gerador sem efeitos no filesystem, lotes de demonstração e execução explícita do Makefile. Os testes de drift usam o Evidently real para conferir ausência de detecção em features idênticas, exclusão do alvo, detecção de alterações conhecidas, hashes dos relatórios e ausência de publicação parcial quando um cenário é inválido.

O [log histórico dos 32 testes da Etapa 1](artifacts/stage1/20261114T201315-0300-5fbfe64d/02_tests.txt) e as cópias de código e requisitos em `artifacts/stage1/` são evidências imutáveis daquela execução, não a implementação atual. Não devem ser usados para instalar ou executar o pipeline atual.

Antes da integração do Evidently, os 42 testes passaram no ambiente do projeto e em um ambiente isolado contendo somente o fechamento transitivo das sete dependências então existentes, reutilizando as versões instaladas. Com Evidently 0.7.23 e pandas 2.3.3, a suíte contém **46 testes**, todos aprovados. A reprodução em diretórios temporários conferiu os mesmos valores nas três partições, relatórios de limpeza, lotes da Etapa 1, quatro cenários, 207.055 predições e métricas/probabilidades do baseline retreinado. Todas as células do notebook foram executadas a partir da raiz e de `notebooks/`. Os 54 arquivos existentes de dados, modelos e evidências permaneceram inalterados, verificados por SHA-256. `pip check` não apontou conflitos.

### Resultados e evidências da Etapa 1

Execução: `20261114T201315-0300-5fbfe64d`. O [manifesto](artifacts/stage1/20261114T201315-0300-5fbfe64d/manifest.json) relaciona comandos, códigos de saída, hashes SHA-256, dependências e cópias das fontes e dos resultados em `artifacts/stage1/20261114T201315-0300-5fbfe64d/`. As cópias em `outputs/` preservam este resultado caso os arquivos de trabalho sejam regenerados.

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
| Dependências | Sem conflitos, retorno 0 | [Log](artifacts/stage1/20261114T201315-0300-5fbfe64d/01_dependencies.txt) |
| Treinamento | Retorno 0, versão completa publicada | [Log](artifacts/stage1/20261114T201315-0300-5fbfe64d/04_train.txt) |
| Lote válido | 500 linhas, retorno 0 | [Log](artifacts/stage1/20261114T201315-0300-5fbfe64d/06_ingest_good.txt) |
| Lote inválido | 505 linhas, retorno 1, movido para quarentena | [Log](artifacts/stage1/20261114T201315-0300-5fbfe64d/07_ingest_bad.txt) |
| Diagnóstico do contrato | 104 entradas de falhas: coluna ausente, idade, nulo, valor negativo e unicidade | [CSV](data/quarantine/bad_batch_errors.csv) |

As 104 entradas do relatório não representam 104 clientes distintos: um registro pode produzir múltiplas entradas. Antes de repetir o lote inválido com o mesmo nome, arquivar a ocorrência ativa de quarentena ou usar outro nome; a ingestão retorna 2 em colisões e preserva a evidência anterior. As cópias desta execução em `artifacts/stage1/` devem ser mantidas inalteradas para conferência dos hashes.

## 10. Vídeo STAR e checklist de entrega

**Link do vídeo: pendente de gravação e publicação pelo grupo.** Substituir esta indicação por um link acessível antes da entrega. Duração máxima: 5 minutos.

| Tempo | STAR | Conteúdo sugerido |
| --- | --- | --- |
| 0:00–0:40 | Situation | Cenário da fintech, risco de degradação silenciosa e impacto sobre crédito |
| 0:40–1:10 | Task | Validar dados, detectar drift, observar o pipeline e definir governança |
| 1:10–3:40 | Action | Mostrar contrato, bloqueio/quarentena, modelo, simulação, predições e relatórios de data drift; após implementação, demonstrar observabilidade; explicar privacidade e retenção |
| 3:40–5:00 | Result | Mostrar métricas e alertas efetivamente gerados, interpretação, limitações causais, mitigação de vieses e resposta proposta |

### Antes da entrega

- [x] Documentar arquitetura e distinguir controles implementados de políticas propostas.
- [x] Explicar tratamento de dados, base legal proposta, minimização e retenção.
- [x] Documentar riscos de viés, hipóteses causais e plano de resposta.
- [ ] Registrar origem, versão, termos de uso e enquadramento dos dados acadêmicos.
- [ ] Atribuir responsáveis e validar as políticas propostas.
- [x] Executar a demonstração e guardar evidências de contrato, quarentena, modelo e métricas.
- [x] Gerar cenários sintéticos de data drift e predições com o modelo salvo.
- [x] Executar a detecção estatística e gerar os relatórios HTML de data drift da Etapa 2.
- [ ] Complementar a menção a concept drift no contexto geral com experimento próprio sobre `P(Y|X)`.
- [ ] Implementar a observabilidade da Etapa 3.
- [ ] Atualizar a análise causal com resultados observados, sem transformar correlação em causa.
- [ ] Revisar os dados e valores expostos no repositório, relatórios e vídeo.
- [ ] Publicar o vídeo STAR e inserir o link nesta seção.

A Etapa 1 e o fluxo de data drift da Etapa 2 possuem execução e evidências preservadas; a documentação de governança cobre a proposta teórica da Etapa 4. O experimento de concept drift citado no contexto geral, a Etapa 3, as pendências de governança e o vídeo ainda precisam ser concluídos para a entrega completa do projeto.
