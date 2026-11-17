# Governança, privacidade e causalidade

[Voltar ao README](../README.md)

## Plano de proteção de dados e base legal

### Finalidades e enquadramento

Na hipótese de operação real da fintech, a base legal proposta para avaliação de risco é **proteção do crédito, art. 7º, X, da LGPD**. O controlador deverá validar sua adequação à finalidade e aos dados tratados. Essa hipótese não autoriza automaticamente o reuso acadêmico ou a publicação da base. Finalidade acadêmica também exige observar os arts. 7º e 11, conforme art. 4º, II, b. Fonte: [texto atualizado da LGPD](https://www2.camara.leg.br/legin/fed/lei/2018/lei-13709-14-agosto-2018-787077-normaatualizada-pl.html).

A fonte, a data de obtenção e a finalidade acadêmica estão registradas no [README — Governança e privacidade](../README.md#governança-e-privacidade). O mesmo registro informa a ausência de identificador de versão externa e aponta o SHA-256 que identifica a cópia de treino utilizada.

O grupo confirmou a validação do enquadramento acadêmico e a conclusão da revisão dos dados. As seções abaixo registram esse estado e mantêm as diretrizes para apresentação e novas publicações.

### Enquadramento acadêmico

**Situação: validado pelo grupo.**

O TC04 é um projeto acadêmico da Fase 4 do MLET/FIAP, desenvolvido para estudar a sustentação e a confiabilidade de um classificador de risco de crédito. Sua finalidade é educacional, sem exploração comercial ou concessão real de crédito. O estudo utiliza o conjunto Give Me Some Credit, com fonte, data de obtenção e identificação da cópia de treino registradas no README.

As atividades abrangem obtenção e armazenamento local dos dados, exploração, limpeza, separação de partições, treinamento e avaliação do baseline, validação de lotes, simulações de data drift e concept drift, geração de predições, relatórios e logs e apresentação dos resultados no repositório e no vídeo STAR. Os cenários de degradação são sintéticos; as conclusões se limitam às intervenções realizadas no experimento.

O público previsto é composto pelos integrantes do grupo, docentes e avaliadores, além de outras pessoas que tenham acesso ao repositório GitHub e ao vídeo conforme a configuração de compartilhamento adotada. A apresentação deve priorizar resultados agregados e exemplos sintéticos. Registros financeiros individuais e valores em quarentena integram a revisão da publicação descrita a seguir.

O projeto remove o índice original das features, valida o contrato de dados e mantém o journal operacional com contagens e métricas, sem valores individuais. A ausência de identificadores diretos não comprova anonimização. A retenção proposta está documentada neste plano; sua execução ainda depende da equipe. O enquadramento descreve o contexto acadêmico e não representa aprovação para uso operacional do modelo.

A validação registrada refere-se ao contexto acadêmico deste projeto. A base proposta para uma operação real da fintech permanece uma hipótese do plano de implantação futura.

### Revisão da publicação

**Situação: revisão dos dados concluída conforme confirmação do grupo.** A tabela mantém o escopo da revisão e as diretrizes para apresentação e novas publicações. A inspeção da gravação integra a preparação do vídeo STAR, ainda pendente.

| Categoria e localização | Aspectos abrangidos pela revisão | Diretrizes para apresentação e novas publicações |
| --- | --- | --- |
| Dados brutos em `data/raw/` | Necessidade de compartilhar o conjunto completo e exposição de registros financeiros individuais | Priorizar referência à fonte e resultados agregados; registrar a justificativa e o alcance de acesso se os arquivos forem mantidos |
| Partições e cenários em `data/` e `artifacts/` | Presença de atributos financeiros e possibilidade de relacionar registros entre arquivos | Avaliar cada conjunto necessário à reprodução; usar exemplos sintéticos para demonstrações de linhas individuais |
| Quarentenas e `*_errors.csv` | Valores rejeitados, índices e amostras de registros | Apresentar o resumo agregado por regra; evitar exibir os valores individuais no vídeo e registrar o tratamento dos arquivos versionados |
| Predições e intervenções em `data/predictions/` e `artifacts/` | Scores, rótulos e posições que permitem associação ao dataset de origem | Priorizar métricas e distribuições agregadas na apresentação; justificar a manutenção dos arquivos por registro |
| Modelos e manifestos em `models/` e `artifacts/` | Metadados, caminhos de ambiente e necessidade dos artefatos para reprodução | Manter a rastreabilidade necessária e revisar metadados expostos antes da apresentação |
| HTML, JSON, CSV de métricas e notebook | Amostras individuais, tabelas incorporadas e saídas antigas | Conferir o conteúdo efetivo; apresentar agregados e remover ou substituir exposições desnecessárias que a revisão identificar |
| Journal e logs de console | Mensagens de exceção e outros campos que possam conter valores individuais | Priorizar o journal estruturado; revisar os logs textuais separadamente, pois não têm o mesmo contrato de campos |

Em novas revisões, registrar, por categoria, os caminhos inspecionados, a decisão de **manter, ajustar, substituir por material sintético ou restringir**, a justificativa e a data. Qualquer ajuste deve ser conferido após sua execução. Quando houver remoção de conteúdo já compartilhado, incluir o histórico Git e as cópias conhecidas na avaliação do alcance da medida.

**Vídeo STAR:** mostrar painéis, métricas e exemplos sintéticos; conferir o conteúdo visível, os arquivos abertos e a configuração de acesso ao vídeo antes da publicação.

### Privacy by Design e by Default

| Medida | Situação atual | Política proposta |
| --- | --- | --- |
| Minimização | Remoção do índice; contrato recusa colunas extras | Admitir somente atributos justificados pela finalidade; não incluir identificadores diretos nas features |
| Segregação | Diretórios de dados brutos, referência e holdout | Separar ambientes, identidades e permissões; manter chaves de associação fora da área analítica |
| Acesso | Sem controle de acesso implementado no projeto | Menor privilégio; dados detalhados restritos à equipe autorizada; revisão ao trocar integrantes |
| Armazenamento e transporte | Arquivos locais; sem configuração de criptografia no código | Armazenamento e backups criptografados; TLS nas transferências; segredos fora do Git |
| Publicação | CSV e Parquet estão versionados; revisão dos dados concluída pelo grupo | Em novas publicações, avaliar exposição de registros individuais; preferir amostras sintéticas e resultados agregados; tratar histórico Git se houver remoção necessária |
| Logs e relatórios | Log de lote e CSV de falhas | Não registrar linhas de clientes; restringir quarentena e sanitizar valores antes de compartilhar relatórios ou vídeo |
| Descarte | Sem rotina implementada | Aplicar a tabela de retenção, incluindo cópias, backups e histórico sob controle da equipe |

O CSV de falhas pode conter valores rejeitados. Quarentena é uma área restrita de diagnóstico, não um mecanismo de anonimização. A ausência de identificadores diretos também não autoriza divulgar registros financeiros individuais.

### Direitos e decisões automatizadas

Em uma operação real, disponibilizar canal de privacidade para acesso, correção e demais solicitações aplicáveis; manter registro de atendimento. Solicitações de revisão de decisões automatizadas devem receber procedimento documentado, com informação sobre critérios e parâmetros, respeitados os segredos comercial e industrial. Fonte: [direitos dos titulares, ANPD](https://www.gov.br/anpd/pt-br/assuntos/titular-de-dados/direito-dos-titulares).

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

## Vieses e uso responsável

O dataset de treino tem 10.026 positivos em 150.000 registros, uma taxa de 6,684%. O treinamento usa `scale_pos_weight` para lidar com o desequilíbrio; isso não demonstra equidade entre grupos.

A exclusão de registros sem renda ou dependentes pode mudar a população atendida. A análise do CSV bruto, reproduzível pelo notebook auxiliar, encontra taxas de inadimplência de aproximadamente 5,61% para renda ausente e 6,95% para renda preenchida, indicando seleção potencialmente relevante. Os filtros de percentil também podem remover clientes de maior risco. Esses resultados justificam investigação, sem comprovar discriminação ou causalidade.

Antes de recomendar uso operacional, a política proposta é:

1. Comparar proporções de exclusão, distribuição e inadimplência antes/depois da limpeza por faixas de idade e renda.
2. Avaliar AUC, recall, FPR, FNR e calibração por segmento, com tamanho amostral e incerteza; explicitar que o positivo representa inadimplência.
3. Comparar exclusão de renda ausente com imputação ajustada no treino e indicador de ausência, avaliando desempenho e seleção.
4. Revisar o uso de idade, dependentes e proxies socioeconômicos; justificar necessidade e alternativas.
5. Aprovar o limiar com base nos custos de inadimplência e de recusa indevida. Não converter automaticamente a probabilidade em aprovação/reprovação sem política validada.

Não há atributos explícitos suficientes para auditar todos os grupos protegidos. Sua ausência limita conclusões sobre equidade. Eventual coleta para auditoria exige finalidade, base legal e controles próprios; não deve ser feita indiscriminadamente.

## Causalidade: diagnóstico e hipóteses

**Não há evidência de degradação temporal real ou de causa econômica no código atual.** O holdout é uma amostra aleatória; o gerador de lote ruim introduz violações de contrato. O script `src.simulate_drift` cria cenários sintéticos de data drift e `src.detect_drift` identifica as alterações por testes estatísticos. Separadamente, `src.concept_drift` muda o mecanismo do alvo com as features fixas e mede a degradação sintética do modelo. Isso permite atribuir diferenças de métricas à intervenção programada dentro do experimento, sem identificar causas econômicas no mundo real.

O mecanismo econômico a investigar é: mudança econômica pode afetar renda e endividamento, que podem alterar o risco de inadimplência; mudanças no público atendido e na política de aprovação também podem alterar a população observada. Essas relações são hipóteses de domínio, não conclusões obtidas neste projeto.

| Hipótese | Evidência necessária | Ação se corroborada |
| --- | --- | --- |
| Erro na coleta ou transformação | Falhas no contrato, mudança de unidade, versão da fonte e logs | Bloquear lote, corrigir origem e reprocessar |
| Data drift: mudança em P(X) | Comparação referência/análise, por exemplo em utilização de crédito e atrasos de 90 dias, com testes e magnitude | Investigar origem, população e impacto sobre predições |
| Concept drift: mudança em P(Y\|X) | Rótulos maturados, queda de desempenho/calibração em condições comparáveis e análise de alternativas | Avaliar recalibração ou retreinamento |
| Seleção causada pela limpeza | Comparação entre incluídos/excluídos e alternativas de tratamento | Rever filtros e repetir avaliação por segmento |
| Efeito econômico sobre inadimplência | Dados temporais, fatores de confusão e desenho causal defensável | Submeter a hipótese a análise específica antes de atribuir causa |

PSI ou um teste KS de duas amostras pode apontar mudança de distribuição; não identifica sua causa nem comprova concept drift. Queda de performance também exige investigar qualidade, seleção, mudança de política e maturação do alvo.

O experimento de data drift registra intervenções em duas features, compara cenários pareados e executa o mesmo modelo. Usa todo o holdout; sorteios com seed 42 determinam quais registros recebem um atraso adicional, sem alterar a composição da amostra. O experimento separado de concept drift usa sorteios reproduzíveis para alterar o mecanismo do alvo. A seed 42 também se aplica à preparação e ao treinamento. Alterar apenas X mantendo rótulos antigos permite estudar sensibilidade das previsões, mas não demonstra desempenho real após uma mudança econômica; a alteração de `P(Y|X)` está explicitada no [experimento de concept drift](concept-drift.md).

Como o alvo tem horizonte de dois anos, produção real requer política de maturação e acompanhamento de rótulos. Sem desfechos disponíveis, monitorar dados e scores e declarar a limitação de avaliação do modelo. O dataset local não permite reconstruir uma série temporal real de concessões.

## Monitoramento e resposta à degradação

Esta seção propõe a operação contínua e a resposta aos sinais; o agendamento e o acionamento externo ainda não estão implementados. A centralização local de eventos reais, métricas e alertas está implementada em `src/execution_log.py` e `src/observability.py`. A [detecção offline de data drift](data-drift.md) e o [experimento de concept drift](concept-drift.md) já estão executados. Os limiares são pontos iniciais e devem ser calibrados; não são exigências do enunciado.

| Sinal | Frequência proposta | Critério inicial | Resposta |
| --- | --- | --- | --- |
| Qualidade do contrato | Cada lote | Qualquer violação | Bloquear e colocar em quarentena; corrigir e revalidar |
| Status, volume, duração e erros | Cada execução | Falha ou lote vazio inesperado; duração acima de 2 vezes a mediana das últimas 20 execuções válidas | Abrir incidente e investigar dependências e origem |
| Drift de features | Cada janela de análise com pelo menos 500 registros | KS com p < 0,01 nas três medidas financeiras ou PSI ≥ 0,20 nas demais categorias; sinal global em ≥ 20% das features | Alertar; examinar magnitude, segmentos e qualidade antes de decidir |
| Performance com rótulos maturados | Cada janela elegível | Queda absoluta de AUC ≥ 0,05 ou recall ≥ 0,10 contra baseline, usando o mesmo limiar | Revisar incerteza, população e política; avaliar candidato |
| Equidade | Cada avaliação e troca de modelo | Aumento absoluto de FNR ou FPR ≥ 0,10 em segmento com amostra suficiente | Revisão conjunta de Ciência de Dados, Risco e Privacidade |

O baseline da Etapa 1 está preservado em `models/runs/32542e11c8c8477ea23d42db115efd0e`, com AUC 0,857208, recall no limiar 0,5 de 0,772268 e KS 0,567594 na validação. O experimento de concept drift já calcula sinais offline de queda de AUC/recall contra o controle do holdout; os alertas já são consolidados no painel local; notificações externas e acompanhamento contínuo por janelas ainda não foram implementados. O volume mínimo acima é uma regra operacional inicial; não substitui avaliação de poder estatístico e incerteza por segmento.

Fluxo de resposta proposto:

1. Registrar incidente com horário, lote, versão do código, dados e modelo; manter amostra restrita somente se necessária.
2. Engenharia verifica integridade e pipeline; Ciência de Dados verifica distribuição, rótulos e desempenho; Risco avalia impacto nas decisões.
3. Em falha de qualidade, interromper o lote. Em degradação crítica, suspender a decisão automática ou encaminhar casos à análise humana, mediante responsável operacional.
4. Avaliar correção, recalibração ou retreinamento. Drift isolado não deve promover automaticamente um novo modelo.
5. Comparar candidato com versão vigente em dados independentes, incluindo segmentos e calibração; registrar aprovação de Risco e responsável técnico.
6. Publicar versão com monitoramento reforçado; manter versão anterior para rollback e registrar encerramento e prevenção da recorrência.

Em suspeita de exposição de dados, conter acessos, preservar evidências restritas e acionar Segurança e o encarregado. A avaliação de impacto e eventuais comunicações deve seguir a regulamentação aplicável. Não há resposta a incidentes implementada no protótipo.

## Responsabilidades e rastreabilidade

A tabela organiza as responsabilidades do plano de governança por função, incluindo os papéis previstos para uma implantação futura.

| Papel | Responsabilidade proposta |
| --- | --- |
| Responsável pelo projeto / Risco | Aprovar finalidade, política de decisão, critérios de promoção e suspensão |
| Engenharia de Dados | Contratos, qualidade, linhagem, quarentena e descarte |
| Ciência de Dados | Modelo, análise de drift, hipóteses causais e vieses |
| MLOps / Segurança | Execução, acessos, logs, versionamento, incidentes e rollback |
| Privacidade / encarregado, na operação real | Validar plano de privacidade, apoiar direitos e avaliar incidentes |

Para cada execução, a política é guardar identificador, horário, commit, ambiente e dependências, hash e origem dos dados, seed quando houver aleatoriedade, versão do contrato, exclusões, métricas, parâmetros e artefato do modelo. A execução preservada da Etapa 1 já inclui manifesto com horários, commit-base, hashes do CSV de entrada, fontes e saídas, versões das dependências, seed, comandos e códigos de saída. Inclui cópias dos scripts e testes usados, pois havia alterações locais sobre o commit-base; o manifesto não atribui esses ajustes ao commit-base. Simulação e inferência já geram manifestos próprios de entradas, saídas e parâmetros; a inferência também registra modelo e horário. A coleta de todos os campos dessa política em cada etapa ainda não está automatizada. Em mudanças operacionais de modelo, comparação, aprovação e motivo continuam sendo políticas propostas.
