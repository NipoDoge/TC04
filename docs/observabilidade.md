# Observabilidade — logs reais e saúde operacional (Etapa 3)

[Voltar ao README](../README.md) · [Arquitetura](arquitetura.md) · [Data drift](data-drift.md)

## Execuções e painel

`src/execution_log.py` instrumenta as operações reais de preparação, ingestão, treino, simulação, inferência, detecção de data drift e experimento de concept drift. Cada chamada grava um evento de início e um de término em `reports/observability/executions.jsonl`. O journal é acumulativo: atualizar o painel não o sobrescreve.

`make observe PYTHON=.venv/bin/python` lê esse journal e os resultados dos experimentos e gera:

| Arquivo | Conteúdo |
| --- | --- |
| `executions.jsonl` | Fonte acumulativa: início/término reais, gravados pelos scripts |
| `events.jsonl` | Cópia dos eventos selecionados na última coleta, regenerável |
| `metrics.json` | Último resultado de cada etapa, contagens históricas, infraestrutura e alertas |
| [`index.html`](../reports/observability/index.html) | Painel com histórico filtrável, durações, volumes, recursos e relatórios |

Abra o HTML no navegador. A coleta não executa nem retreina o pipeline. Os registros passam a existir a partir da instrumentação; artefatos anteriores não são convertidos em execuções fictícias.

## Contrato dos eventos

- `run_id`: agrupa etapas de uma execução; definir `TC04_RUN_ID` para compartilhar o identificador entre processos. Sem essa variável, cada chamada recebe seu próprio identificador.
- `execution_id`: identificador único de cada chamada, compartilhado pelo início e término.
- `source`, `event`, `status`: etapa, início/término e resultado (`running`, `ok`, `rejected`, `fail` ou `alert`).
- `started_at_utc`, `ts`: horário de início e horário do evento, em UTC.
- `duration_ms`: tempo real decorrido, medido com relógio monotônico até a conclusão da função; não inclui importação do interpretador nem a escrita do evento final.
- `volume`, `volume_unit`: quantidade de linhas da operação. Na ingestão é o lote lido; na preparação, a soma das partições finais; no treino, as linhas de treino (validação em `validation_rows`); em simulação, drift e concept drift, a soma dos cenários; na inferência, referência mais cenários. São processamentos, não clientes distintos. Em falha antes da leitura, o volume é desconhecido (`null`).
- `error_type`: classe do erro, sem texto da exceção ou valores individuais. Na ingestão, os códigos 0/1/2 são preservados e registrados em `exit_code`.
- `violations`: resumo por coluna/regra e quantidade de linhas afetadas, sem valores rejeitados; duplicatas de linha inteira são consolidadas entre colunas.
- `contract_failure_count`: número de ocorrências de validação, quando disponíveis; não representa necessariamente clientes distintos.
- `alert_count`: cenários com alerta global de data drift ou queda de desempenho, conforme a etapa.
- `cpu_seconds`: CPU de usuário e sistema consumida pelo processo durante a operação.
- `infrastructure`: pico de memória residente do processo em MB, CPUs disponíveis e espaço livre/total do disco do diretório corrente. O pico RSS é acumulado desde o início do processo, não memória exclusiva da etapa. CPU e RSS não incluem processos filhos.

Não são gravadas linhas de clientes, conteúdo da quarentena ou mensagens de exceção no journal. Os logs textuais legados de console continuam separados. Escritas concorrentes usam bloqueio de arquivo POSIX, flush e fsync. Falha na gravação inicial impede executar a operação sem registro; falha na gravação final é propagada e pode ocorrer depois da publicação dos artefatos. Quando a operação também falha, sua exceção original é preservada e a falha secundária do log é sinalizada no console e em uma nota da exceção. Nesse caso, permanece um início sem término, sinalizado no painel como execução não concluída.

Para escolher outro journal:

```bash
export TC04_EXECUTION_LOG=/tmp/tc04/executions.jsonl
export TC04_RUN_ID=execucao-manual-001
make prepare PYTHON=.venv/bin/python
make train PYTHON=.venv/bin/python
make observe PYTHON=.venv/bin/python
```

O caminho padrão é relativo ao diretório corrente. Ao mudar de diretório entre chamadas, configure um caminho absoluto. Não use os nomes regeneráveis `events.jsonl`, `metrics.json` ou `index.html` como journal.

## Interpretação da saúde

O painel usa a última chamada iniciada de cada etapa, identificada por horário e IDs. Uma rejeição antiga não mantém a ingestão em falha depois de uma aprovação: o incidente continua no histórico. A quarentena é contabilizada separadamente, em arquivos `.parquet` (lotes), sem contar o CSV de erros como outro lote.

| Estado | Significado |
| --- | --- |
| `ok` | Última operação concluída com sucesso |
| `rejected` | Última operação bloqueada pelo contrato |
| `fail` | Falha operacional ou erro na leitura/validação de uma fonte de monitoramento |
| `alert` | Operação concluída com drift ou degradação detectados |
| `running` | Início sem término observado |
| `unknown` | Nenhum registro real para a etapa |
| `stale` | Execução sem término há mais de cinco minutos ou último registro concluído com mais de 24 horas |

A saúde consolidada fica `fail` diante de falha/rejeição atual, registro vencido ou journal inválido; `incomplete` diante de etapas ou artefatos ausentes/execução em andamento; `alert` diante de drift, degradação ou pouco disco; `ok` quando todas as etapas e fontes estão disponíveis e não há sinais ativos. As contagens históricas incluem falhas já recuperadas.

O limite de disco é 5% livre. O prazo dos registros concluídos é configurável com `--stale-after-seconds`. Esses limites são critérios iniciais de demonstração, não SLAs validados. O painel mostra as métricas publicadas do baseline separadas do desempenho nos cenários de concept drift. O experimento continua sintético e offline; não é avaliação contínua de decisões reais de crédito.

## Demonstração reproduzível

```bash
make demo-observability PYTHON=.venv/bin/python
make observe PYTHON=.venv/bin/python
```

A demonstração exige que `artifacts/operational_demo/` ainda não exista. Prepara dados, treina outro baseline e gera cenários/relatórios nessa pasta, preservando as evidências anteriores. Para repetir:

```bash
.venv/bin/python -m src.demo_observability --output-dir /tmp/tc04-demo-nova
```

São executados dez procedimentos reais: uma preparação com fonte ausente, preparação válida, treino, ingestão de arquivo ausente, rejeição do lote inválido, aprovação de 500 registros, simulação, inferência, detecção de data drift e concept drift. As duas falhas e a rejeição são intencionais. A aprovação posterior demonstra recuperação; os alertas de drift permanecem ativos.

A execução preservada em 17/11/2026 gerou **20 eventos**, com **5 sucessos, 2 falhas operacionais, 1 rejeição e 2 operações com alertas**. Todas as sete etapas têm duração e volume medidos. Foram processadas 117.644 linhas nas partições finais, 65.867 no treino e 207.055 predições; as três operações por cenários contabilizam 141.188 linhas cada. O estado final é **`alert`**, com dois cenários de data drift global e três de degradação sob concept drift; as falhas provocadas foram recuperadas.

Evidências: [resumo](../artifacts/operational_demo/summary.json), [painel da execução](../artifacts/operational_demo/dashboard/index.html) e [journal central](../reports/observability/executions.jsonl). O painel da demonstração filtra seu `run_id` e usa seus próprios modelos e relatórios; o painel principal consolida o journal e os artefatos padrão. Para selecionar uma execução manualmente, usar `python -m src.observability --run-id <identificador>` com os argumentos de fontes correspondentes.

## Validação e limites

`make test PYTHON=.venv/bin/python` usa um journal temporário, para não misturar os testes com as evidências operacionais. A suíte contém 114 testes, incluindo 49 de observabilidade (13 originais e 36 da adaptação remota e regressões): instrumentação, privacidade dos eventos, falha de leitura, recuperação, concorrência, fontes inválidas, registros vencidos, preservação do journal e demonstração integrada com XGBoost/Pandera/Evidently reais.

O monitoramento é local e atualizado sob demanda. Não há serviço contínuo, agendamento, notificação externa, rotação automática do journal ou medição de disponibilidade de infraestrutura ao longo do tempo. A instrumentação registra recursos por operação e no momento da coleta; retenção e controles de acesso seguem o plano de [governança](governanca.md). A implementação usa recursos POSIX e foi executada no macOS; Windows não foi validado.

## Integração dos testes da branch remota

Os testes de `nipodoge/feat/fase4-entrega-final`, commit `4f50a83bb68a3f705fd03eb470236a5475d4d902`, foram adaptados às APIs locais em [test_observability_remote.py](../tests/test_observability_remote.py), preservando os testes anteriores. A cobertura inclui schema JSONL, NumPy/Path, correlação de IDs, agregação de falhas sem valores, quarentena configurável, isolamento por execução, HTML independente de recursos externos, acessibilidade das tabelas, links relativos e limiares de desempenho. Um caso integrado anterior foi ampliado para conferir os pares início/término de todas as etapas e a contagem de alertas.

Foram corrigidos a ordenação por instantes com fusos distintos, o tratamento de fontes/linhas inválidas (incluindo NaN e UTF-8 inválido), a aceitação de manifestos auxiliares corrompidos, o arredondamento no limite dos alertas e a ocultação da exceção original por falha de log. O painel apresenta os estados das fontes separadamente. Serialização, renderização e gravação temporária precedem a substituição dos arquivos publicados; falhas nessas fases preservam o painel anterior. A substituição final é feita por arquivo, não constitui uma transação entre os três arquivos.

O contrato local de bloquear operações sem registro inicial foi mantido; o teste remoto que exige continuar sem log não foi transplantado. Também não foram introduzidos MLflow, lotes mensais, gráficos SVG, alertas de score/latência/volume ou decomposição causal para satisfazer testes de módulos ausentes. [Rastreabilidade e resultados](reproducao.md#testes-remotos-de-observabilidade).
