# Roteiro do vídeo STAR

Vídeo: [tc04-star.mp4](tc04-star.mp4), 4:49 de duração (limite de 5 minutos), 1920×1080. Legendas: [tc04-star.srt](tc04-star.srt); as mesmas frases estão gravadas na imagem.

A narração é sintetizada (voz Luciana, pt-BR, do macOS) e os integrantes não aparecem. Sem poder ouvir o áudio, a pronúncia foi conferida passando cada frase por um reconhecedor de fala e comparando com o roteiro; a grafia usada só para a fala foi ajustada onde ele não entendia. Os quadros usam as capturas de `docs/img/` e números agregados dos relatórios versionados; nenhum dado individual de cliente é exibido.

| Bloco | Intervalo | Duração |
| --- | --- | ---: |
| Abertura | 0:00 a 0:08 | 9 s |
| Situation (situação) | 0:08 a 0:43 | 35 s |
| Task (tarefa) | 0:43 a 1:14 | 31 s |
| Action (ação) | 1:14 a 3:24 | 129 s |
| Result (resultado) | 3:24 a 4:49 | 86 s |

## Abertura

| Início | Na tela | Narração |
| ---: | --- | --- |
| 0:00 | Cartela de título | Este é o nosso trabalho da fase 4: drift, observabilidade e governança de um modelo de crédito. |

## Situation (situação)

| Início | Na tela | Narração |
| ---: | --- | --- |
| 0:08 | Quadro da situação: contexto, base de dados e métricas do baseline | Uma fintech usa um modelo de pontuação de crédito para estimar o risco de inadimplência de cada cliente. |
| 0:16 | (mesmo quadro) | O risco é a degradação silenciosa: a economia e o perfil dos clientes mudam, e o modelo piora sem que ninguém perceba. |
| 0:24 | (mesmo quadro) | Trabalhamos com uma base pública de crédito, com 150 mil registros, e um modelo XGBoost como referência. |
| 0:33 | (mesmo quadro) | Na validação, ele tem AUC de 0,857 e identifica 77% dos inadimplentes. |

## Task (tarefa)

| Início | Na tela | Narração |
| ---: | --- | --- |
| 0:43 | As quatro etapas da tarefa | Nossa tarefa tinha quatro partes. |
| 0:46 | (mesmo quadro) | Primeiro: validar os dados que chegam, com um contrato de dados. |
| 0:51 | (mesmo quadro) | Segundo: simular e detectar drift, separando a mudança nos dados da mudança na relação com a inadimplência. |
| 0:59 | (mesmo quadro) | Terceiro: observar o processamento, com registros, métricas e alertas centralizados. |
| 1:06 | (mesmo quadro) | E quarto: documentar a governança, com proteção de dados, vieses, causalidade e resposta à degradação. |

## Action (ação)

| Início | Na tela | Narração |
| ---: | --- | --- |
| 1:14 | Regras do contrato e trecho de src/contract.py | Começamos pelo contrato de dados, escrito em Pandera: colunas e tipos definidos, faixas válidas, sem valores nulos e sem linhas duplicadas. |
| 1:26 | Dashboard: sinais por lote, com a entrega corrompida de maio em destaque | Na simulação, uma entrega corrompida chega em maio. O contrato bloqueia as 4.612 linhas e manda o lote para a quarentena. |
| 1:36 | (mesmo quadro) | Os lotes com drift passam pelo contrato: são dados válidos, que apenas mudaram de distribuição. |
| 1:43 | Níveis de data drift e de concept drift e o calendário dos lotes mensais | Para o drift, definimos os níveis de intensidade uma única vez e usamos os mesmos em todos os experimentos. |
| 1:51 | (mesmo quadro) | No data drift, a renda cai, o endividamento e o uso do rotativo sobem, e quem passa a usar todo o limite ganha um atraso. |
| 2:00 | (mesmo quadro) | No concept drift, bons pagadores de baixa renda passam a inadimplir, sem nenhuma mudança nas variáveis. |
| 2:08 | (mesmo quadro) | Aplicamos cada nível à base inteira, um mecanismo por vez, e depois a oito lotes mensais, que combinam os dois. |
| 2:17 | Relatório do Evidently do lote de agosto | A detecção usa o Evidently: o teste KS, para as medidas contínuas, e o PSI, para as contagens. |
| 2:26 | Dashboard: cenários pareados, veredito do Evidently e PSI por variável | Com muitos dados, o teste acusa até mudanças mínimas. Por isso gravamos também o tamanho do efeito. |
| 2:33 | (mesmo quadro) | No cenário mais fraco, o teste já aponta drift, mas o PSI da renda é de apenas 0,03. No mais forte, chega a 0,60. |
| 2:45 | Dashboard: saúde do pipeline por lote | No monitoramento, cada lote passa por contrato, escoragem, drift, avaliação e regras de alerta. |
| 2:53 | Evento do log central e métricas no MLflow | Cada etapa grava um evento em um log central, e as métricas de cada lote vão para o MLflow. |
| 3:01 | Visão geral do dashboard e lista das seções | Tudo aparece em um painel único, em um só arquivo, que abre direto no navegador. |
| 3:07 | Quadro de governança: proteção de dados, uso responsável e resposta | Na governança, propomos a base legal de proteção do crédito, a minimização dos dados e prazos de retenção. |
| 3:15 | (mesmo quadro) | Relatórios e logs guardam apenas valores agregados, e documentamos riscos de viés e um plano de resposta. |

## Result (resultado)

| Início | Na tela | Narração |
| ---: | --- | --- |
| 3:24 | Dashboard: AUC e PSI por lote, com maio em destaque | Em oito meses simulados, a AUC caiu de 0,85 para 0,62. |
| 3:31 | (mesmo quadro) | O ponto mais importante é maio: a AUC cai para 0,77 e nenhum PSI das entradas passa do limite de aviso. |
| 3:41 | (mesmo quadro) | É o efeito do concept drift. Só o rótulo, que no caso real leva até dois anos para maturar, revela essa queda. |
| 3:50 | Dashboard: cenários pareados, colunas de AUC e de clientes sinalizados | O data drift se comporta de outro jeito: quase não muda a ordenação dos clientes, mas muda as decisões. |
| 3:58 | (mesmo quadro) | No nível mais forte, o modelo passa a sinalizar cerca de 38% dos clientes, contra 25%, com a mesma inadimplência. |
| 4:09 | Dashboard: experimento contrafactual | O experimento contrafactual confirma: o concept drift tira 0,18 da AUC; o data drift, só 0,01. |
| 4:19 | Dashboard: tabela de alertas | O monitoramento gerou 21 alertas, do bloqueio pelo contrato às quedas de desempenho. |
| 4:26 | Quadro de limites: tudo é simulação | Tudo isso é simulação: mostra que o monitoramento funciona, não o desempenho de uma operação real. |
| 4:34 | Quadro de conclusão | A conclusão é que vigiar as entradas não basta. É preciso acompanhar os rótulos, as decisões e o desempenho por segmento. |
| 4:42 | (mesmo quadro) | O código, os relatórios e a documentação estão no repositório. Obrigado. |

## Como regenerar

O vídeo é montado por [gera_video.py](gera_video.py): cada frase vira um quadro (HTML renderizado pelo Chrome) e um trecho de fala (comando `say`), e o ffmpeg junta tudo. Para trocar uma frase, editar a lista `SCRIPT` e rodar:

```bash
python docs/video/gera_video.py . /tmp/tc04-video tudo
cp /tmp/tc04-video/tc04-star.mp4 /tmp/tc04-video/tc04-star.srt /tmp/tc04-video/roteiro.md docs/video/
```

Requer macOS com a voz Luciana, Google Chrome, Pillow, NumPy e ffmpeg. Na lista, o quinto campo de cada frase é a grafia usada só para a fala (siglas e termos em inglês escritos como se pronunciam). Para gravar a narração com voz própria, tirar o áudio com `ffmpeg -i tc04-star.mp4 -an -c:v copy sem-audio.mp4` e narrar sobre as legendas.
