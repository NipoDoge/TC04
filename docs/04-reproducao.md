# Reprodução, testes e ressalvas técnicas

Parte da documentação do [TC04](../README.md). Anterior: [monitoramento e observabilidade](03-monitoramento-e-observabilidade.md).

## Ambiente e dependências

Executar a partir da raiz, em ambiente virtual com **Python 3.12**, após instalar as dependências de [requirements.txt](../requirements.txt). As versões fixadas exigem Python 3.12 ou superior. O arquivo contém dez dependências diretas do pipeline e dos testes:

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
| SciPy | Teste KS de duas amostras no tamanho do efeito dos cenários e dos lotes |
| MLflow | Rastreamento do baseline e do monitoramento |

O Evidently 0.7.23 falha ao importar com pandas 3.1.0 por usar uma função interna removida. A integração fixa **pandas 2.3.3**, adiciona o Evidently e mantém as versões das outras seis dependências diretas do pipeline. A preparação, as predições e o baseline foram reproduzidos com os mesmos valores após esse ajuste. Parquet regenerado pode ter hash binário diferente devido à versão do pandas nos metadados; comparar o conteúdo e guardar o novo hash no manifesto. Os artefatos preservados da execução com pandas 3.1.0 continuam inalterados.

Dependências transitivas, como Pydantic e Plotly, continuam necessárias às bibliotecas e são resolvidas pelo pip; não são listadas como dependências diretas. O pacote de compatibilidade `pathlib` foi removido porque o código usa a biblioteca padrão. Ferramentas de servidor Jupyter, Prometheus e demais pacotes sem uso direto foram retirados da lista de execução; bibliotecas trazidas transitivamente pelo Evidently permanecem necessárias à sua instalação. No Linux, o próprio XGBoost declara a dependência transitiva de NCCL; não há configuração de GPU no projeto. A limpeza do arquivo não desinstala pacotes de ambientes virtuais já existentes.

O arquivo fixa as versões diretas; não é um lockfile completo de dependências transitivas. O ambiente integral da execução histórica permanece registrado em `artifacts/stage1/` para auditoria, e o ambiente completo em que o monitoramento foi executado, com as transitivas, está em [requirements-lock.txt](../requirements-lock.txt).

O notebook é auxiliar e não faz parte da execução do pipeline. Para executá-lo em um editor compatível com Jupyter, instalar opcionalmente `python -m pip install -r requirements-notebooks.txt`, que adiciona o IPython kernel às dependências do projeto. A raiz é localizada a partir da pasta do kernel; execute o notebook dentro do projeto. Suas saídas antigas foram removidas para evitar resultados e caminhos de outras máquinas.

No macOS, instalar também o runtime OpenMP exigido pelo XGBoost:

```bash
brew install libomp
```

A execução definitiva da Etapa 1 foi realizada no projeto em **14/11/2026, às 20:13, no fuso America/Sao_Paulo**, usando macOS ARM64 e Python 3.12.14. O `pip check` não encontrou conflitos, os 32 testes passaram e os artefatos estão preservados. As versões completas do ambiente constam em [environment.txt](../artifacts/stage1/20261114T201315-0300-5fbfe64d/environment.txt).

O monitoramento por lote e a observabilidade versionados são da execução `mon-20261115T024544Z-cd2bd7` (horário UTC), na mesma máquina, com o ambiente do `requirements-lock.txt`. Em 15/11/2026, em um clone limpo do repositório, `make all` (preparação, treino, impacto dos cenários, lotes mensais, monitoramento, análise causal, dashboard e testes) rodou sem erro em cerca de um minuto e meio. O modelo retreinado nessa conferência produziu **predições idênticas** às do modelo publicado na Etapa 1 em todo o holdout, e os números de drift, desempenho e alertas coincidiram com os versionados. Os relatórios versionados foram gerados com o modelo publicado (`models/runs/32542e11c8c8477ea23d42db115efd0e`), e não com o retreinado.

## Comandos

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
python -m src.scenario_impact     # impacto dos cenários (lê reports/drift e data/predictions)
python -m src.simulate_batches    # lotes mensais em data/production/batches/
python -m src.monitor             # contrato, escoragem, drift, alertas, MLflow
python -m src.causal_analysis     # cenários contrafactuais
python -m src.dashboard           # reports/dashboard.html
python -m unittest discover -s tests
```

O resultado esperado da chamada com `bad_batch.parquet` é código de saída 1, lote movido para quarentena e arquivo `bad_batch_errors.csv`. A chamada do lote válido deve retornar 0, se os tipos e valores satisfizerem o contrato. Erros operacionais retornam 2, incluindo arquivo ausente/ilegível ou falha ao registrar a quarentena. O código 1 somente é retornado após salvar o relatório e movimentar o lote rejeitado.

Com o ambiente ativado, `make` exibe os alvos sem executar o pipeline. A sequência explícita é `make prepare`, `make train`, `make simulate-drift`, `make predict` e `make detect-drift`; `train` usa as partições existentes e não chama `prepare`. Para indicar o interpretador, usar `PYTHON=.venv/bin/python`. Simulação, inferência e detecção exigem destinos inexistentes; para novas execuções sem substituir as evidências atuais, usar os argumentos `--output-dir` descritos em [experimentos de drift](02-experimentos-de-drift.md). A detecção usa diretamente a referência e os cenários, sem depender das predições. Se regenerar partições ou trocar o modelo, gerar novos cenários, predições e relatórios antes de compará-los; manifestos antigos identificam as versões anteriores pelos hashes.

Os alvos de monitoramento e observabilidade usam o modelo já publicado em `models/current`:

| Alvo | O que faz |
| --- | --- |
| `make scenario-impact` | Impacto dos cenários pareados no modelo, lido de `reports/drift/` e `data/predictions/` |
| `make simulate-batches` | Recria os lotes mensais e a entrega corrompida |
| `make monitor` | Recria os lotes e roda contrato, escoragem, drift, desempenho e alertas. Recria antes porque a ingestão move a entrega corrompida para a quarentena |
| `make causal`, `make dashboard` | Análise causal e dashboard |
| `make report` | `scenario-impact`, `monitor`, `causal` e `dashboard`, nessa ordem |
| `make all` | Fluxo completo a partir do CSV bruto: `prepare`, `train`, `report` e `test`. O treino publica uma nova versão em `models/runs/` |
| `make mlflow-ui` | Interface do MLflow sobre `mlflow.db` |

`make all` não executa `demo-bad`, `simulate-drift`, `predict`, `detect-drift` nem `concept-drift`: as evidências desses passos já estão versionadas, e os cinco recusam sobrescrever o destino existente. O bloqueio pelo contrato é exercitado de novo dentro do monitoramento, com a entrega corrompida de maio. Cada execução do monitor acrescenta eventos ao log central e uma execução ao MLflow; o dashboard mostra a execução do resumo mais recente. Reexecutar `make report` reescreve o dashboard, os resumos e os dois relatórios versionados de `reports/batches/` com novo identificador de execução, novos horários e durações; os números de drift, desempenho e alertas se repetem, e `reports/drift/` não é tocado. Ficam fora do Git, e são recriados por `make report`: `logs/`, `mlflow.db`, `mlruns/`, os lotes simulados e escorados, a quarentena do monitoramento e seis dos oito relatórios por lote. O `mlflow.db` guarda o caminho absoluto de `mlruns/`: se a pasta do projeto for movida, apagar os dois e rodar de novo.

As evidências da Etapa 2 formam uma cadeia, e cada elo lê o anterior: `simulate-drift` (cenários), `predict` (predições), `detect-drift` (relatórios e `summary.csv`) e `scenario-impact` (resumo de impacto). Para regenerá-las no lugar, remover ou mover os três diretórios de destino e rodar, nesta ordem:

```bash
rm -r data/production/scenarios data/predictions reports/drift
make simulate-drift predict detect-drift scenario-impact
```

`scenario-impact` não recalcula nada: se `reports/drift/summary.csv` ou `data/predictions/` faltarem, falha com erro. O resumo de impacto identifica o modelo que gerou `data/predictions/`. Depois de um novo treino, `models/current` passa a apontar para outra versão; como o treino é determinístico, as predições coincidem, mas quem quiser os dois resumos apontando para a mesma versão deve regenerar a cadeia acima.

`make concept-drift` executa o experimento separado com a referência, o holdout e o modelo existentes. Não depende dos cenários t1–t3 de data drift. O destino também deve ser inexistente e pode ser alterado com `python -m src.concept_drift --output-dir ...`; esse experimento não sobrescreve os dados ou relatórios das etapas anteriores.

Os alvos `make demo-bad` e `make demo-good` demonstram o contrato da Etapa 1. `demo-bad` gera os dois lotes e somente passa se a ingestão retornar 1, o lote sair da entrada e os arquivos de quarentena e relatório existirem e não estiverem vazios. Deve preceder `demo-good`. O gerador valida o holdout e exige pelo menos 500 registros; sua importação não lê nem grava arquivos. Para gerar lotes em outro local, usar `python -m src.make_bad_batch --output-dir /tmp/tc04-batches`. Cada script cria os diretórios necessários às suas saídas. Reexecutar preparação ou geração dos lotes sobrescreve suas saídas; o treinamento cria uma nova versão e troca o apontador publicado. A ingestão rejeita colisões na quarentena. Preservar as evidências e resolver colisões antes de repetir a demonstração.

## Testes

Os testes são executados com `make test` ou `python -m unittest discover -s tests -v`. Verificam limites calculados somente no treino, invariância a alterações na validação/holdout, aplicação de limites fixos, separação das partições e criação das saídas. Cobrem bloqueio do treinamento, códigos de saída da ingestão, erros operacionais, colisões na quarentena e falso sucesso no Makefile. Incluem rejeição de lotes vazios e valores infinitos e invariância à ordem das colunas. Os testes de publicação simulam falhas e conferem preservação da versão anterior e limpeza de temporários. Simulação e inferência têm testes de pareamento, preservação de entradas, hashes, ordem das features, exclusão do alvo, limiar 0,5, probabilidades inválidas e publicação sem resultados parciais. Também há cobertura para importação do gerador sem efeitos no filesystem, lotes de demonstração e execução explícita do Makefile. Os testes de drift usam o Evidently real para conferir ausência de detecção em features idênticas, exclusão do alvo, detecção de alterações conhecidas, hashes dos relatórios e ausência de publicação parcial quando um cenário é inválido.

O [log histórico dos 32 testes da Etapa 1](../artifacts/stage1/20261114T201315-0300-5fbfe64d/02_tests.txt) e as cópias de código e requisitos em `artifacts/stage1/` são evidências imutáveis daquela execução, não a implementação atual. Não devem ser usados para instalar ou executar o pipeline atual.

Antes da integração do Evidently, os 42 testes passaram no ambiente do projeto e em um ambiente isolado contendo somente o fechamento transitivo das sete dependências então existentes, reutilizando as versões instaladas. Após a integração do Evidently 0.7.23 com pandas 2.3.3, passaram 46 testes; com o experimento de concept drift a suíte contém **51 testes**, todos aprovados. Os cinco testes adicionais verificam a intervenção condicional e cumulativa, seeds, invariância de features/scores, métricas em exemplos conhecidos, segmentos de uma classe, concordância com métricas nativas do Evidently, hashes e publicação sem resultados parciais.

A reprodução anterior em diretórios temporários conferiu os mesmos valores nas três partições, relatórios de limpeza, lotes da Etapa 1, quatro cenários de data drift, 207.055 predições e métricas/probabilidades do baseline retreinado. Todas as células do notebook foram executadas a partir da raiz e de `notebooks/`, preservando os 54 arquivos então existentes de dados, modelos e evidências. O experimento de concept drift também preservou os arquivos anteriores, incluindo os relatórios de data drift, com conferência SHA-256. `pip check` não apontou conflitos; não foram necessárias novas dependências para o concept drift.

Com o monitoramento, a observabilidade e a unificação dos experimentos, a suíte tem **138 testes**: 54 nos arquivos das Etapas 1 e 2 (os 51 anteriores e três novos, para a regra de atraso, a intervenção em uma variável por vez e o tamanho do efeito no resumo) e 84 do monitoramento. Estes cobrem o PSI por decis (amostras iguais, deslocamento, baixa cardinalidade, faixas vazias e a limitação das faixas das pontas), os lotes mensais (níveis de controle não alteram nada, determinismo por seed, os lotes usam exatamente as intervenções dos experimentos pareados, o rótulo só muda de bom para mau e só no segmento de menor renda, o calendário vem da configuração, lotes com drift passam no contrato e a entrega corrompida reprova), cada regra de alerta nos seus limites, o log central (esquema do evento, falha registrada e relançada, log indisponível não derruba o pipeline) e o dashboard (seções, escape de texto, execução sem lotes válidos). Os testes de integração treinam um modelo em dados sintéticos pequenos e rodam, em diretório temporário, o monitoramento completo com o Evidently de verdade e a cadeia `simulate_drift`, `predict`, `detect_drift` e `scenario_impact`; conferem que o tamanho do efeito do resumo é o do lote escorado, que o veredito do Evidently chega ao resumo, que as evidências de entrada não são alteradas e que resumos e log não carregam valores de clientes.

## Auditoria das Etapas 1 e 2 e o que mudou depois dela

A [auditoria consolidada das Etapas 1 e 2](../artifacts/audit_stage1_stage2/audit.json) registra a nova execução dos 51 testes no ambiente atual e em um ambiente isolado, a reprodução dos dois experimentos, a integridade dos manifestos e a conferência dos links. Foram atualizados dois endereços da ANPD que redirecionavam e removidos metadados antigos de Python 3.14.4 do notebook. Não foi identificado fluxo legado ativo; as fontes em `artifacts/stage1/` são evidências históricas. Os commits anteriores à padronização ainda têm mensagens não semânticas, sem reescrita do histórico. A verificação de HTML foi estrutural e de consistência com os JSONs, sem inspeção visual em navegador, indisponível no ambiente de auditoria. A validação de execução cobre macOS ARM64/Python 3.12; outros sistemas não foram executados.

**A auditoria descreve o commit `a7b5a47` e é anterior ao monitoramento e à unificação dos experimentos; não foi refeita.** Em relação ao estado auditado, mudaram:

- **Fontes**: `src/config.py`, `src/simulate_drift.py`, `src/detect_drift.py` e `src/concept_drift.py` (níveis com quatro variáveis, tamanho do efeito e funções de intervenção compartilhadas); `src/prepare.py`, `src/train.py` e `src/ingest.py` (eventos no log central); `tests/test_simulate_drift.py` e `tests/test_detect_drift.py`; `Makefile`, `requirements.txt`, `.gitignore` e este conjunto de documentos. Os hashes de fontes gravados em `active_source_hashes` não correspondem mais a esses arquivos.
- **Evidências regeneradas**: os cenários de `data/production/scenarios/`, as predições de `data/predictions/` e os relatórios de `reports/drift/`. Os hashes e os resultados de data drift citados na auditoria (duas features com drift em `t1`, `t2` e `t3`) são os da versão anterior, recuperável no histórico do Git (commit `d953260`).
- **Sem alteração**: os dados brutos, a referência, a validação, o holdout, o modelo publicado, as evidências de `artifacts/stage1/` e o experimento de `artifacts/concept_drift/`, que foi reexecutado depois da unificação e reproduziu os mesmos resultados.
