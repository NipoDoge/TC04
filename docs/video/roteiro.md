# Roteiro do vídeo STAR

Vídeo: [tc04-star.mp4](tc04-star.mp4), 4:08 de duração (limite de 5 minutos), 1920×1080. Legendas: [tc04-star.srt](tc04-star.srt); as mesmas frases estão gravadas na imagem.

Os quadros usam as capturas de `docs/img/` e números agregados dos relatórios versionados; nenhum dado individual de cliente é exibido.

| Bloco | Intervalo | Duração |
| --- | --- | ---: |
| Abertura | 0:00 a 0:07 | 8 s |
| Situation (situação) | 0:07 a 0:38 | 31 s |
| Task (tarefa) | 0:38 a 1:05 | 27 s |
| Action (ação) | 1:05 a 2:54 | 109 s |
| Result (resultado) | 2:54 a 4:08 | 74 s |

## Abertura

| Início | Na tela | Narração |
| ---: | --- | --- |
| 0:00 | Cartela de título | Este é o nosso trabalho da fase 4: drift, observabilidade e governança de um modelo de crédito. |

## Situation (situação)

| Início | Na tela | Narração |
| ---: | --- | --- |
| 0:07 | Quadro da situação: contexto, base de dados e métricas do baseline | Uma fintech usa um modelo de pontuação de crédito para estimar o risco de inadimplência de cada cliente. |
| 0:14 | (mesmo quadro) | O risco é a degradação silenciosa: a economia e o perfil dos clientes mudam, e o modelo piora sem que ninguém perceba. |
| 0:21 | (mesmo quadro) | Trabalhamos com uma base pública de crédito, com 150 mil registros, e um modelo XGBoost como referência. |
| 0:29 | (mesmo quadro) | Na validação, ele tem AUC de 0,857 e identifica 77% dos inadimplentes. |

## Task (tarefa)

| Início | Na tela | Narração |
| ---: | --- | --- |
| 0:38 | As quatro etapas da tarefa | Nossa tarefa tinha quatro partes. |
| 0:41 | (mesmo quadro) | Primeiro: validar os dados que chegam, com um contrato de dados. |
| 0:45 | (mesmo quadro) | Segundo: simular e detectar drift, separando a mudança nos dados da mudança na relação com a inadimplência. |
| 0:52 | (mesmo quadro) | Terceiro: observar o processamento, com registros, métricas e alertas centralizados. |
| 0:57 | (mesmo quadro) | E quarto: documentar a governança, com proteção de dados, vieses, causalidade e resposta à degradação. |

## Action (ação)

| Início | Na tela | Narração |
| ---: | --- | --- |
| 1:05 | Regras do contrato e trecho de src/contract.py | Começamos pelo contrato de dados, escrito em Pandera: colunas e tipos definidos, faixas válidas, sem valores nulos e sem linhas duplicadas. |
| 1:14 | Dashboard: sinais por lote, com a entrega corrompida de maio em destaque | Na simulação, uma entrega corrompida chega em maio. O contrato bloqueia as 4.612 linhas e manda o lote para a quarentena. |
| 1:23 | (mesmo quadro) | Os lotes com drift passam pelo contrato: são dados válidos, que apenas mudaram de distribuição. |
| 1:29 | Níveis de data drift e de concept drift e o calendário dos lotes mensais | Para o drift, definimos os níveis de intensidade uma única vez e usamos os mesmos em todos os experimentos. |
| 1:36 | (mesmo quadro) | No data drift, a renda cai, o endividamento e o uso do rotativo sobem, e quem passa a usar todo o limite ganha um atraso. |
| 1:44 | (mesmo quadro) | No concept drift, bons pagadores de baixa renda passam a inadimplir, sem nenhuma mudança nas variáveis. |
| 1:50 | (mesmo quadro) | Aplicamos cada nível à base inteira, um mecanismo por vez, e depois a oito lotes mensais, que combinam os dois. |
| 1:58 | Relatório do Evidently do lote de agosto | A detecção usa o Evidently: o teste KS, para as medidas contínuas, e o PSI, para as contagens. |
| 2:05 | Dashboard: cenários pareados, veredito do Evidently e PSI por variável | Com muitos dados, o teste acusa até mudanças mínimas. Por isso gravamos também o tamanho do efeito. |
| 2:11 | (mesmo quadro) | No cenário mais fraco, o teste já aponta drift, mas o PSI da renda é de apenas 0,03. No mais forte, chega a 0,60. |
| 2:21 | Dashboard: saúde do pipeline por lote | No monitoramento, cada lote passa por contrato, escoragem, drift, avaliação e regras de alerta. |
| 2:28 | Evento do log central e métricas no MLflow | Cada etapa grava um evento em um log central, e as métricas de cada lote vão para o MLflow. |
| 2:34 | Visão geral do dashboard e lista das seções | Tudo aparece em um painel único, em um só arquivo, que abre direto no navegador. |
| 2:39 | Quadro de governança: proteção de dados, uso responsável e resposta | Na governança, propomos a base legal de proteção do crédito, a minimização dos dados e prazos de retenção. |
| 2:46 | (mesmo quadro) | Relatórios e logs guardam apenas valores agregados, e documentamos riscos de viés e um plano de resposta. |

## Result (resultado)

| Início | Na tela | Narração |
| ---: | --- | --- |
| 2:54 | Dashboard: AUC e PSI por lote, com maio em destaque | Em oito meses simulados, a AUC caiu de 0,85 para 0,62. |
| 3:01 | (mesmo quadro) | O ponto mais importante é maio: a AUC cai para 0,77 e nenhum PSI das entradas passa do limite de aviso. |
| 3:09 | (mesmo quadro) | É o efeito do concept drift. Só o rótulo, que no caso real leva até dois anos para maturar, revela essa queda. |
| 3:17 | Dashboard: cenários pareados, colunas de AUC e de clientes sinalizados | O data drift se comporta de outro jeito: quase não muda a ordenação dos clientes, mas muda as decisões. |
| 3:23 | (mesmo quadro) | No nível mais forte, o modelo passa a sinalizar cerca de 38% dos clientes, contra 25%, com a mesma inadimplência. |
| 3:33 | Dashboard: experimento contrafactual | O experimento contrafactual confirma: o concept drift tira 0,18 da AUC; o data drift, só 0,01. |
| 3:42 | Dashboard: tabela de alertas | O monitoramento gerou 21 alertas, do bloqueio pelo contrato às quedas de desempenho. |
| 3:48 | Quadro de limites: tudo é simulação | Tudo isso é simulação: mostra que o monitoramento funciona, não o desempenho de uma operação real. |
| 3:54 | Quadro de conclusão | A conclusão é que vigiar as entradas não basta. É preciso acompanhar os rótulos, as decisões e o desempenho por segmento. |
| 4:02 | (mesmo quadro) | O código, os relatórios e a documentação estão no repositório. Obrigado. |
