# Reprodução, dependências e validação

[Voltar ao README](../README.md)

Executar a partir da raiz, em ambiente virtual com **Python 3.12**, após instalar as dependências de [requirements.txt](../requirements.txt). As versões fixadas exigem Python 3.12 ou superior. O arquivo contém oito dependências diretas do pipeline e dos testes:

| Dependência | Uso |
| --- | --- |
| NumPy | Validação de finitude, probabilidades e operações numéricas |
| pandas | Leitura, transformação e escrita de datasets |
| Pandera com extra `pandas` | Contrato de qualidade |
| PyArrow | Engine de leitura e escrita de Parquet pelo pandas |
| scikit-learn | Separação estratificada e métricas de classificação |
| XGBoost | Treinamento e execução do classificador salvo |
| joblib | Persistência e leitura do modelo |
| Evidently AI | Testes estatísticos de data drift e relatórios HTML/JSON de drift e desempenho de classificação |

O Evidently 0.7.23 falha ao importar com pandas 3.1.0 por usar uma função interna removida. A integração fixa **pandas 2.3.3**, adiciona o Evidently e mantém as versões das outras seis dependências diretas do pipeline. A preparação, as predições e o baseline foram reproduzidos com os mesmos valores após esse ajuste. Parquet regenerado pode ter hash binário diferente devido à versão do pandas nos metadados; comparar o conteúdo e guardar o novo hash no manifesto. Os artefatos preservados da execução com pandas 3.1.0 continuam inalterados.

Dependências transitivas, como SciPy, Pydantic e Plotly, continuam necessárias às bibliotecas e são resolvidas pelo pip; não são listadas como dependências diretas. O pacote de compatibilidade `pathlib` foi removido porque o código usa a biblioteca padrão. Ferramentas de servidor Jupyter, Prometheus e demais pacotes sem uso direto foram retirados da lista de execução; bibliotecas trazidas transitivamente pelo Evidently permanecem necessárias à sua instalação. No Linux, o próprio XGBoost declara a dependência transitiva de NCCL; não há configuração de GPU no projeto. A limpeza do arquivo não desinstala pacotes de ambientes virtuais já existentes.

O arquivo fixa as versões diretas; não é um lockfile completo de dependências transitivas. O ambiente integral da execução histórica permanece registrado em `artifacts/stage1/` para auditoria.

O notebook é auxiliar e não faz parte da execução do pipeline. Para executá-lo em um editor compatível com Jupyter, instalar opcionalmente `python -m pip install -r requirements-notebooks.txt`, que adiciona o IPython kernel às dependências do projeto. A raiz é localizada a partir da pasta do kernel; execute o notebook dentro do projeto. Suas saídas antigas foram removidas para evitar resultados e caminhos de outras máquinas.

No macOS, instalar também o runtime OpenMP exigido pelo XGBoost:

```bash
brew install libomp
```

A execução definitiva da Etapa 1 foi realizada no projeto em **14/11/2026, às 20:13, no fuso America/Sao_Paulo**, usando macOS ARM64 e Python 3.12.14. O `pip check` não encontrou conflitos, os 32 testes passaram e os artefatos estão preservados. As versões completas do ambiente constam em [environment.txt](../artifacts/stage1/20261114T201315-0300-5fbfe64d/environment.txt).

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

Com o ambiente ativado, `make` exibe os alvos sem executar o pipeline. A sequência explícita é `make prepare`, `make train`, `make simulate-drift`, `make predict` e `make detect-drift`; `train` usa as partições existentes e não chama `prepare`. Para indicar o interpretador, usar `PYTHON=.venv/bin/python`. Simulação, inferência e detecção exigem destinos inexistentes; para novas execuções sem substituir as evidências atuais, usar os argumentos `--output-dir` descritos nos guias de [data drift](data-drift.md), [comparativo](comparativo-drift.md) e [concept drift](concept-drift.md). A detecção usa diretamente a referência e os cenários, sem depender das predições. Se regenerar partições ou trocar o modelo, gerar novos cenários, predições e relatórios antes de compará-los; manifestos antigos identificam as versões anteriores pelos hashes.

`make compare-data-drift` compara os artefatos ativos com o perfil de renda/dívida arquivado, sem regerar os cenários nem treinar o modelo. Exige os mesmos hashes de modelo/holdout e destino inexistente. O perfil antigo não é uma opção do gerador ativo: sua versão de código está identificada pelo commit no arquivo histórico. O comparativo é um relatório adicional, separado dos relatórios de detecção por cenário.

`make concept-drift` executa o experimento separado com a referência, o holdout e o modelo existentes. Não depende dos cenários t1–t3 de data drift. O destino também deve ser inexistente e pode ser alterado com `python -m src.concept_drift --output-dir ...`; esse experimento não sobrescreve os dados ou relatórios das etapas anteriores.

Os alvos `make demo-bad` e `make demo-good` demonstram o contrato da Etapa 1. `demo-bad` gera os dois lotes e somente passa se a ingestão retornar 1, o lote sair da entrada e os arquivos de quarentena e relatório existirem e não estiverem vazios. Deve preceder `demo-good`. O gerador valida o holdout e exige pelo menos 500 registros; sua importação não lê nem grava arquivos. Para gerar lotes em outro local, usar `python -m src.make_bad_batch --output-dir /tmp/tc04-batches`. Cada script cria os diretórios necessários às suas saídas. Reexecutar preparação ou geração dos lotes sobrescreve suas saídas; o treinamento cria uma nova versão e troca o apontador publicado. A ingestão rejeita colisões na quarentena. Preservar as evidências e resolver colisões antes de repetir a demonstração.

Os testes são executados com `make test` ou `python -m unittest discover -s tests -v`. Verificam limites calculados somente no treino, invariância a alterações na validação/holdout, aplicação de limites fixos, separação das partições e criação das saídas. Cobrem bloqueio do treinamento, códigos de saída da ingestão, erros operacionais, colisões na quarentena e falso sucesso no Makefile. Incluem rejeição de lotes vazios e valores infinitos e invariância à ordem das colunas. Os testes de publicação simulam falhas e conferem preservação da versão anterior e limpeza de temporários. Simulação e inferência têm testes de pareamento, preservação de entradas, hashes, ordem das features, exclusão do alvo, limiar 0,5, probabilidades inválidas e publicação sem resultados parciais. Também há cobertura para importação do gerador sem efeitos no filesystem, lotes de demonstração e execução explícita do Makefile. Os testes de drift usam o Evidently real para conferir ausência de detecção em features idênticas, exclusão do alvo, detecção de alterações conhecidas, hashes dos relatórios e ausência de publicação parcial quando um cenário é inválido.

O [log histórico dos 32 testes da Etapa 1](../artifacts/stage1/20261114T201315-0300-5fbfe64d/02_tests.txt) e as cópias de código e requisitos em `artifacts/stage1/` são evidências imutáveis daquela execução, não a implementação atual. Não devem ser usados para instalar ou executar o pipeline atual.

Antes da integração do Evidently, os 42 testes passaram no ambiente do projeto e em um ambiente isolado contendo somente o fechamento transitivo das sete dependências então existentes, reutilizando as versões instaladas. Após a integração do Evidently 0.7.23 com pandas 2.3.3, passaram 46 testes; com o experimento de concept drift passaram 51. Após substituir as features de data drift e adicionar o comparativo, a suíte passou a conter **54 testes**, todos aprovados naquela validação. A cobertura adicional verifica sorteios reproduzíveis, grupos aninhados, incremento inteiro de atrasos a partir de zero, invariância das demais colunas, comparação dos dois perfis e rejeição de predições/snapshots adulterados. Os testes de concept drift continuam verificando intervenção condicional, métricas conhecidas, hashes, concordância com Evidently e ausência de publicação parcial.

A reprodução anterior em diretórios temporários conferiu os mesmos valores nas três partições, relatórios de limpeza, lotes da Etapa 1, quatro cenários de data drift, 207.055 predições e métricas/probabilidades do baseline retreinado. Todas as células do notebook foram executadas a partir da raiz e de `notebooks/`, preservando os 54 arquivos então existentes de dados, modelos e evidências. O experimento de concept drift também preservou os arquivos anteriores, incluindo os relatórios de data drift, com conferência SHA-256. `pip check` não apontou conflitos; não foram necessárias novas dependências para o concept drift.

A [auditoria consolidada das Etapas 1 e 2](../artifacts/audit_stage1_stage2/audit.json) é uma evidência histórica anterior à troca das features de data drift: registra 51 testes no ambiente então vigente e em um ambiente isolado, reprodução dos experimentos, integridade e links daquele estado. Seus hashes e conclusões se referem ao commit e às fontes indicados, não certificam automaticamente os novos artefatos. Os resultados de renda/dívida foram preservados em `artifacts/data_drift_income_debt/`; os resultados atuais e o comparativo têm manifestos próprios. Foram atualizados dois endereços da ANPD e removidos metadados antigos do notebook. As fontes em `artifacts/stage1/` também são evidências históricas, sem uso no pipeline ativo. Os commits antigos com mensagens não semânticas foram preservados. A verificação de HTML foi estrutural e de consistência com os JSONs, sem inspeção visual em navegador, indisponível no ambiente. A execução validada cobre macOS ARM64/Python 3.12; outros sistemas não foram executados.

## Testes adaptados da entrega final

Foram preservados os 54 testes existentes e acrescentados **11 casos**, totalizando **65 testes**, além de ampliar as verificações de métricas em um caso existente. A origem é a branch `nipodoge/feat/fase4-entrega-final`, commit [`4f50a83`](https://github.com/NipoDoge/TC04/tree/4f50a83bb68a3f705fd03eb470236a5475d4d902). A adaptação reaproveita os cenários de integração de `tests/test_monitor.py`, o caso de renda/endividamento de `tests/test_detect_drift.py` e o gerador sintético de `tests/support.py`; não é uma cópia dos 138 testes daquela branch.

| Arquivo | Cobertura incorporada |
| --- | --- |
| [test_pipeline_integration.py](../tests/test_pipeline_integration.py) — 8 casos | Preparação, treino e recarga do XGBoost real; métricas da validação vinculadas à versão publicada; contrato e quarentena; rejeição antes de chamar o modelo; contagem e predições de todos os cenários; relatórios do Evidently real; isolamento entre data drift e concept drift; procedência entre etapas; preservação das entradas e recuperação de falha na geração de relatórios |
| [test_detect_drift.py](../tests/test_detect_drift.py) — 1 caso novo | Detectar choques conhecidos de renda e endividamento sem substituir os testes das features do perfil ativo |
| [test_simulate_drift.py](../tests/test_simulate_drift.py) — 2 casos novos | A seed altera somente o choque de atrasos; inverter o alvo não altera as intervenções nas features |
| [test_concept_drift.py](../tests/test_concept_drift.py) — caso ampliado | Precisão, prevalência, taxa de positivos previstos, score médio e KS, inclusive a ausência de KS definido em amostra de uma classe |

[support.py](../tests/support.py) gera dados financeiros fictícios com seed fixa. A integração usa 3.000 registros antes da preparação, partições separadas e diretórios temporários com limpeza automática. As saídas de simulação, inferência, drift e concept drift ficam nesses temporários. A rejeição pela ingestão é executada com o diretório corrente temporariamente ajustado para que a quarentena também fique isolada. Não são necessários novos pacotes, registros do dataset versionado ou downloads.

O perfil desta branch continua sendo utilização de crédito e atrasos de 90 dias. O teste alternativo de renda/endividamento verifica a capacidade geral do detector, sem mudar a configuração dos experimentos. As validações foram aplicadas às funções existentes em `src/`; não foi necessário alterar sua implementação.

Na adaptação original não foram incorporados os testes de dashboard, MLflow, logs centrais, alertas, PSI por decis, lotes mensais ou decomposição causal. Posteriormente, a instrumentação e a cobertura próprias de observabilidade foram adicionadas, conforme a seção abaixo. Os testes de isolamento e desempenho foram adaptados às APIs de simulação e concept drift disponíveis; isso não implementa um orquestrador que conecte automaticamente a ingestão à inferência.

Para executar tudo, incluindo o comparativo histórico já existente:

```bash
make test PYTHON=.venv/bin/python
.venv/bin/python -m pip check
```

Para executar somente a nova integração:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_pipeline_integration.py' -v
```

Resultado da validação: **65 testes aprovados em 22,859 segundos**, `pip check` sem conflitos e **142 arquivos versionados protegidos sem alteração de SHA-256**, incluindo código de produção, dados, modelos, relatórios e evidências anteriores. A recarga do modelo versionado reproduziu AUC **0,8572078813**, recall **0,7722681359** e KS **0,5675944208**, iguais às métricas salvas. Não houve alteração de `src/`, requisitos ou Makefile. A execução usou o ambiente virtual existente; não foi criada uma instalação isolada de dependências.

As evidências desta validação ficam em [test_adaptation](../artifacts/test_adaptation/validation.json), com o resultado da suíte, dependências e hashes das fontes de teste. O [log completo](../artifacts/test_adaptation/tests.txt) registra os casos executados. A verificação abrange comportamento e consistência de relatórios HTML/JSON; não certifica sua aparência visual nem a execução em outros sistemas operacionais.

## Validação da observabilidade operacional

A primeira validação da instrumentação operacional continha **78 testes**: os 65 anteriores e 13 casos em `tests/test_observability.py`. A execução passou em **22,607 segundos**. O alvo `make test` configura um journal temporário com limpeza automática; não mistura eventos de teste com os logs operacionais. Para executar diretamente o unittest, definir `TC04_EXECUTION_LOG` para um destino temporário.

`make demo-observability PYTHON=.venv/bin/python` executou preparação, treino, contrato, simulação, inferência e relatórios reais em uma pasta separada. Foram preservados 20 eventos de dez chamadas, incluindo duas falhas operacionais provocadas, uma rejeição e recuperação. O painel final apresenta todas as sete etapas com duração e volume, dois alertas globais de data drift e três cenários com degradação de desempenho. O estado consolidado `alert` é esperado; o histórico de incidentes permanece disponível. [Evidências e comandos](observabilidade.md#demonstração-reproduzível).

A preparação agora aceita `--raw` e `--output-root` para permitir execuções isoladas. O treino e os experimentos mantêm os contratos e formatos de retorno; a instrumentação acrescenta arquivos de eventos. Não foram adicionadas dependências. As evidências históricas de testes anteriores continuam referindo-se às versões originais.

## Testes remotos de observabilidade

Origem: branch `nipodoge/feat/fase4-entrega-final`, commit [`4f50a83`](https://github.com/NipoDoge/TC04/tree/4f50a83bb68a3f705fd03eb470236a5475d4d902), conferido com `git fetch nipodoge feat/fase4-entrega-final`. A adaptação preservou os 78 testes locais e acrescentou **36 casos**, totalizando **114 testes**; um teste de demonstração anterior foi ampliado. Não foi feito merge da implementação remota.

| Origem remota | Cobertura adaptada |
| --- | --- |
| `tests/test_observability.py` | Cinco casos de eventos/IDs/duração/leitura e três de ingestão, agregação de regras e privacidade |
| `tests/test_dashboard.py` | Nove casos de seções, números, HTML independente de recursos externos, nomes acessíveis das tabelas, isolamento por run, escaping, ausência de sucessos/fontes, links e CLI |
| `tests/test_alerts.py` | Quatro casos de saúde, limite de queda de AUC/recall e AUC indefinida |
| `tests/test_monitor.py` | Dois casos de métricas, um de falha operacional de drift e ampliação da integração existente para correlação de eventos e alertas |
| Regressões locais adicionais | Doze casos de exceção original, fusos, metadados/JSON/UTF-8 inválidos, manifesto corrompido, proteção do journal e preservação do painel em falhas de renderização/gravação temporária |

São 24 novos casos adaptados das fontes remotas e 12 regressões adicionais. A implementação local mantém início/término por chamada, estados `ok`/`fail`, duração em milissegundos, recall indefinido como `null` quando não há positivos e bloqueio em falha de gravação inicial do log. Na origem, alguns desses contratos são diferentes. Os testes de MLflow, cenários mensais, gráficos SVG, regras adicionais de alertas e decomposição causal não foram incorporados porque esses módulos não fazem parte da implementação atual. Nenhum teste foi marcado como skip ou expected failure.

A integração acrescentou suporte a escalares NumPy/Path no JSONL e trouxe o resumo agregado das violações para a ingestão, preservando os retornos 0/1/2. As regressões identificadas foram corrigidas no logger, coletor, publicação do painel e comparação dos limiares de desempenho. Não foram adicionadas dependências.

Executar: `make test PYTHON=.venv/bin/python`. Para somente a observabilidade: `python -m unittest discover -s tests -p 'test_observability*.py' -v`; esses testes configuram journals temporários por caso. A validação final e os hashes estão em [validation.json](../artifacts/observability_test_adaptation/validation.json), com [log da suíte](../artifacts/observability_test_adaptation/tests.txt) e [reprodução isolada com dados do projeto](../artifacts/observability_test_adaptation/reproduction.json). A checagem cobre comportamento, sintaxe, dependências e consistência estrutural dos relatórios; não certifica ausência de qualquer defeito nem aparência visual em navegador ou execução em outros sistemas operacionais.

Resultado final: **114 testes aprovados em 21,766 segundos**, `pip check` sem conflitos, módulos compilados e `git diff --check` sem erros. Os **192 arquivos protegidos** (incluindo dados, modelos, relatórios e journal operacional anteriores, exceto os snapshots regeneráveis do painel) mantiveram seus hashes. A reprodução completa em diretório temporário repetiu as métricas do baseline publicado e confirmou 20 eventos, recuperação das falhas, dois cenários de data drift global e três alertas de desempenho.
