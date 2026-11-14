# TC04 | Governança e confiabilidade de um modelo de crédito

Projeto acadêmico da Fase 4 do MLET/FIAP. O cenário é uma fintech cujo modelo de Credit Scoring pode sofrer degradação silenciosa diante de mudanças econômicas e do perfil dos clientes. O objetivo é validar dados, investigar mudanças de distribuição e definir como proteger dados e responder à degradação.

Este README consolida a **proposta de governança da Etapa 4**, com base no enunciado do Tech Challenge, páginas 6 e 7, e no código deste repositório. As políticas abaixo orientam uma operação futura; não constituem evidência de controles já implantados nem de conformidade integral. O projeto não realiza concessões reais de crédito.

## 1. Escopo e evidências disponíveis

| Componente | Evidência no repositório | Situação |
| --- | --- | --- |
| Exploração e avaliação inicial de seleção | [Notebook](notebooks/01_exploracao.ipynb) | Código e algumas saídas disponíveis |
| Limpeza e separação referência/produção | [prepare.py](src/prepare.py), [relatório de limpeza](data/reference/cleaning_report.json) e arquivos Parquet | Implementados; divisão aleatória, sem simulação temporal |
| Contrato de dados | [contract.py](src/contract.py) | Implementado em Pandera |
| Bloqueio e quarentena | [ingest.py](src/ingest.py) | Implementados para falhas de validação do contrato |
| Lotes de demonstração | [make_bad_batch.py](src/make_bad_batch.py) | Gerador implementado; execução deve produzir as evidências |
| Treinamento e avaliação | [train.py](src/train.py) | Script implementado; modelo e métricas gerados não constam no repositório inspecionado |
| Simulação e detecção de drift | Sem scripts ou relatórios identificados | Pendentes |
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
    -> src.prepare: limpeza + divisão estratificada (seed 42)
        -> data/reference/reference.parquet (70%)
            -> src.train: divisão interna 80/20, treino XGBoost e avaliação
                -> models/model.joblib + models/metrics.json
        -> data/production/holdout.parquet (30%)
            -> src.make_bad_batch: lotes de demonstração
                -> data/incoming/{good,bad}_batch.parquet
                    -> src.ingest + contrato Pandera
                        -> válido: log e retorno 0
                        -> inválido: movimentação para data/quarantine,
                           CSV de falhas e retorno 1
```

As saídas de treinamento e demonstração são produzidas quando os scripts são executados. A ingestão aprovada somente valida o lote: não o encaminha automaticamente ao modelo. Também não há integração obrigando o treinamento a passar pelo contrato.

O modelo é um `XGBClassifier` com 300 estimadores, profundidade 4, taxa de aprendizado 0,05 e ponderação da classe positiva. A avaliação calcula AUC ROC, recall com limiar 0,5 e KS como separação máxima entre TPR e FPR. **Esse KS de classificação não é um teste de drift entre datasets.** O limiar 0,5 é usado na avaliação; uma política de aprovação de crédito não foi implementada.

## 3. Qualidade de dados e limites do contrato

O contrato exige alvo binário; idade inteira entre 18 e 100, inclusive; renda não nula e não negativa; demais medidas e contagens não negativas; colunas e tipos definidos; ausência de colunas extras e de linhas integralmente duplicadas. Os campos não marcados como anuláveis também rejeitam nulos por padrão. Não há coerção de tipos configurada.

A preparação remove duplicatas, idades fora do intervalo, registros sem renda ou dependentes, rendas negativas e códigos 96/98 nas colunas de atraso. Em seguida, remove valores acima do percentil 99 de `DebtRatio` e de utilização de crédito, sequencialmente.

O relatório existente registra 609 duplicatas, 14 exclusões por idade, 29.214 por renda/dependentes ausentes, nenhuma por renda negativa, 145 pelos códigos 96/98 e 1.201 e 1.189 pelos dois filtros de outliers. São contagens sequenciais, não estimativas independentes de prevalência.

Limitações que precisam orientar a operação:

- O contrato exige `SeriousDlqin2yrs`; atende lotes históricos rotulados. Inferência real exige contrato próprio sem o alvo, pois o desfecho futuro ainda não é conhecido.
- O contrato não reproduz todos os filtros da limpeza: não bloqueia especificamente 96/98 nem aplica limites superiores de renda, dívida ou utilização.
- A limpeza e seus percentis são calculados antes da separação referência/holdout. Para avaliação independente, ajustar as regras estatísticas somente no treino e aplicá-las às demais bases.
- O lote ruim contém idade 15, renda nula, dívida negativa, duplicatas e coluna ausente. Isso testa qualidade, não simula drift temporal.
- O código trata `SchemaErrors`; erros de leitura, escrita e movimentação não têm tratamento operacional dedicado.

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

A exclusão de registros sem renda ou dependentes pode mudar a população atendida. O notebook registra taxas de inadimplência de aproximadamente 5,61% para renda ausente e 6,95% para renda preenchida, indicando seleção potencialmente relevante. Os filtros de percentil também podem remover clientes de maior risco. Esses resultados justificam investigação, sem comprovar discriminação ou causalidade.

Antes de recomendar uso operacional, a política proposta é:

1. Comparar proporções de exclusão, distribuição e inadimplência antes/depois da limpeza por faixas de idade e renda.
2. Avaliar AUC, recall, FPR, FNR e calibração por segmento, com tamanho amostral e incerteza; explicitar que o positivo representa inadimplência.
3. Comparar exclusão de renda ausente com imputação ajustada no treino e indicador de ausência, avaliando desempenho e seleção.
4. Revisar o uso de idade, dependentes e proxies socioeconômicos; justificar necessidade e alternativas.
5. Aprovar o limiar com base nos custos de inadimplência e de recusa indevida. Não converter automaticamente a probabilidade em aprovação/reprovação sem política validada.

Não há atributos explícitos suficientes para auditar todos os grupos protegidos. Sua ausência limita conclusões sobre equidade. Eventual coleta para auditoria exige finalidade, base legal e controles próprios; não deve ser feita indiscriminadamente.

## 6. Causalidade: diagnóstico e hipóteses

**Não há evidência de degradação temporal ou de causa econômica no código atual.** O holdout é uma amostra aleatória, e o gerador de lote ruim introduz violações de contrato. Não foram encontrados experimento causal, simulação de concept drift ou relatório estatístico de drift.

O mecanismo econômico a investigar é: mudança econômica pode afetar renda e endividamento, que podem alterar o risco de inadimplência; mudanças no público atendido e na política de aprovação também podem alterar a população observada. Essas relações são hipóteses de domínio, não conclusões obtidas neste projeto.

| Hipótese | Evidência necessária | Ação se corroborada |
| --- | --- | --- |
| Erro na coleta ou transformação | Falhas no contrato, mudança de unidade, versão da fonte e logs | Bloquear lote, corrigir origem e reprocessar |
| Data drift: mudança em P(X) | Comparação referência/análise, por exemplo em renda e endividamento, com testes e magnitude | Investigar origem, população e impacto sobre predições |
| Concept drift: mudança em P(Y\|X) | Rótulos maturados, queda de desempenho/calibração em condições comparáveis e análise de alternativas | Avaliar recalibração ou retreinamento |
| Seleção causada pela limpeza | Comparação entre incluídos/excluídos e alternativas de tratamento | Rever filtros e repetir avaliação por segmento |
| Efeito econômico sobre inadimplência | Dados temporais, fatores de confusão e desenho causal defensável | Submeter a hipótese a análise específica antes de atribuir causa |

PSI ou um teste KS de duas amostras pode apontar mudança de distribuição; não identifica sua causa nem comprova concept drift. Queda de performance também exige investigar qualidade, seleção, mudança de política e maturação do alvo.

Para a etapa experimental, registrar as intervenções em pelo menos duas features, sua intensidade e seed; comparar cenários controle e alterado e executar o mesmo modelo. Alterar apenas X mantendo rótulos antigos permite estudar sensibilidade das previsões, mas não demonstra desempenho real após uma mudança econômica. Um cenário de concept drift exige explicitar a alteração sintética do mecanismo do alvo ou obter novos desfechos observados.

Como o alvo tem horizonte de dois anos, produção real requer política de maturação e acompanhamento de rótulos. Sem desfechos disponíveis, monitorar dados e scores e declarar a limitação de avaliação do modelo. O dataset local não permite reconstruir uma série temporal real de concessões.

## 7. Monitoramento e resposta à degradação

Esta seção prescreve controles ainda não automatizados. Os limiares são pontos iniciais para simulação e devem ser calibrados; não são resultados medidos ou exigências do enunciado.

| Sinal | Frequência proposta | Critério inicial | Resposta |
| --- | --- | --- | --- |
| Qualidade do contrato | Cada lote | Qualquer violação | Bloquear e colocar em quarentena; corrigir e revalidar |
| Status, volume, duração e erros | Cada execução | Falha ou lote vazio inesperado; duração acima de 2 vezes a mediana das últimas 20 execuções válidas | Abrir incidente e investigar dependências e origem |
| Drift de features | Cada janela de análise com pelo menos 500 registros | PSI ≥ 0,20 em feature monitorada, com bins fixados na referência | Alertar; examinar magnitude, segmentos e qualidade antes de decidir |
| Performance com rótulos maturados | Cada janela elegível | Queda absoluta de AUC ≥ 0,05 ou recall ≥ 0,10 contra baseline, usando o mesmo limiar | Revisar incerteza, população e política; avaliar candidato |
| Equidade | Cada avaliação e troca de modelo | Aumento absoluto de FNR ou FPR ≥ 0,10 em segmento com amostra suficiente | Revisão conjunta de Ciência de Dados, Risco e Privacidade |

O baseline de comparação deverá ser versionado quando houver avaliação executada. Não há valor de AUC, recall ou KS verificado nesta proposta. O volume mínimo acima é uma regra operacional inicial; não substitui avaliação de poder estatístico e incerteza por segmento.

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

Para cada execução, a política é guardar identificador, horário, commit, ambiente e dependências, hash e origem dos dados, seed, versão do contrato, exclusões, métricas, parâmetros e artefato do modelo. Em mudanças de modelo, acrescentar comparação com a versão anterior, aprovadores e motivo. O código já fixa seed 42 na preparação/treino e gera relatório de limpeza e métricas; o registro completo e o versionamento de artefatos estão pendentes.

## 9. Reprodução e ressalvas técnicas

Executar a partir da raiz, em ambiente virtual com **Python 3.12**, após instalar as dependências de [requirements.txt](requirements.txt). As versões fixadas exigem Python 3.12 ou superior. A dependência NVIDIA NCCL é instalada somente no Linux; no macOS, o XGBoost utiliza CPU.

No macOS, instalar também o runtime OpenMP exigido pelo XGBoost:

```bash
brew install libomp
```

A instalação foi validada localmente em macOS ARM64 com Python 3.12.14: `pip check` sem conflitos, imports das bibliotecas principais, validação dos dois Parquets pelo contrato, rejeição de idade inválida e treino reduzido do XGBoost em CPU com persistência via Joblib. O treinamento completo do baseline e a demonstração de ingestão/quarentena continuam pendentes.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
mkdir -p data/reference data/production data/incoming data/quarantine models
python -m src.prepare
python -m src.train
python -m src.make_bad_batch
python -m src.ingest data/incoming/good_batch.parquet
python -m src.ingest data/incoming/bad_batch.parquet
```

O resultado esperado da última chamada é código de saída 1, lote movido para quarentena e arquivo `bad_batch_errors.csv`. A chamada do lote válido deve retornar 0, se os tipos e valores satisfizerem o contrato. Essas expectativas precisam ser demonstradas em execução antes de serem apresentadas como resultados.

Com o ambiente ativado e os diretórios criados, também é possível usar `make prepare`, `make train`, `make demo-bad` e `make demo-good`. O alvo `demo-bad` gera os dois lotes, portanto deve preceder `demo-good`. O gerador de lotes exige pelo menos 500 registros no holdout. Reexecutar scripts sobrescreve artefatos, e a ingestão inválida movimenta o arquivo de origem: preservar evidências antes de repetir a demonstração.

## 10. Vídeo STAR e checklist de entrega

**Link do vídeo: pendente de gravação e publicação pelo grupo.** Substituir esta indicação por um link acessível antes da entrega. Duração máxima: 5 minutos.

| Tempo | STAR | Conteúdo sugerido |
| --- | --- | --- |
| 0:00–0:40 | Situation | Cenário da fintech, risco de degradação silenciosa e impacto sobre crédito |
| 0:40–1:10 | Task | Validar dados, detectar drift, observar o pipeline e definir governança |
| 1:10–3:40 | Action | Mostrar contrato, bloqueio/quarentena e modelo; após implementação, demonstrar simulação, relatório de drift e observabilidade; explicar privacidade e retenção |
| 3:40–5:00 | Result | Mostrar métricas e alertas efetivamente gerados, interpretação, limitações causais, mitigação de vieses e resposta proposta |

### Antes da entrega

- [x] Documentar arquitetura e distinguir controles implementados de políticas propostas.
- [x] Explicar tratamento de dados, base legal proposta, minimização e retenção.
- [x] Documentar riscos de viés, hipóteses causais e plano de resposta.
- [ ] Registrar origem, versão, termos de uso e enquadramento dos dados acadêmicos.
- [ ] Atribuir responsáveis e validar as políticas propostas.
- [ ] Executar a demonstração e guardar evidências de contrato, quarentena, modelo e métricas.
- [ ] Completar as Etapas 2 e 3 com simulação, relatório estatístico e observabilidade.
- [ ] Atualizar a análise causal com resultados observados, sem transformar correlação em causa.
- [ ] Revisar os dados e valores expostos no repositório, relatórios e vídeo.
- [ ] Publicar o vídeo STAR e inserir o link nesta seção.

A documentação cobre a proposta teórica da Etapa 4. O link do vídeo e as evidências pendentes precisam ser acrescentados para concluir a entrega do projeto.
