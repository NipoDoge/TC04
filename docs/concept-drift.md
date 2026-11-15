# Concept drift e comportamento do modelo

[Voltar ao README](../README.md)

[concept_drift.py](../src/concept_drift.py) mantém **as mesmas features, os mesmos clientes, a mesma ordem, o mesmo modelo e o limiar 0,5** em quatro cenários. Altera somente o mecanismo do alvo histórico. A distinção entre mudanças em `P(X)` e em `P(Y|X)` segue a formulação de [Tan, Lee e Salehi (2019)](https://arxiv.org/abs/1909.11251); aqui a mudança condicional é conhecida por construção, não inferida somente pela mudança da prevalência.

O segmento afetado é `MonthlyIncome ≤ 3500`, corte ajustado no primeiro quartil **somente da referência de treino**. Inclui 9.147 registros do holdout (aproximadamente 25,9%; os empates no corte explicam a diferença para 25%). A hipótese ilustrativa é um choque adicional de inadimplência nesse segmento. Não há estimativa de inflação, desemprego ou efeito causal econômico.

Para cada linha, sortear `U ~ Uniforme(0,1)`, independente do modelo, com seed 42 e reutilizar o mesmo `U` nos quatro cenários. Com `S(X)` indicando o segmento e `Y0` o rótulo histórico:

```text
Yq = Y0 OU (S(X) E U < q)
Pq(Y=1|X) = p0(X) + (1 - p0(X)) * q * S(X)
```

`p0(X)` é a probabilidade histórica desconhecida, não o score do modelo. Apenas rótulos 0 no segmento podem virar 1; positivos históricos e rótulos fora do segmento permanecem intactos. Os valores de `q` são 0, 0,10, 0,25 e 0,50, definidos em `src/config.py`. São chances adicionais para os negativos elegíveis, não taxas finais de inadimplência. O uso dos mesmos sorteios torna as alterações cumulativas entre cenários. O mecanismo não usa scores nem ajusta parâmetros para obter determinada queda de métrica.

```bash
make concept-drift PYTHON=.venv/bin/python
```

O comando usa o holdout original, não os cenários de data drift, e publica um conjunto completo em `artifacts/concept_drift/`: datasets, predições, auditoria por linha em `interventions.parquet`, métricas globais e por segmento, sensibilidade a cinco seeds, quatro relatórios de classificação HTML/JSON do Evidently, comparação de features e manifesto com parâmetros, versões e hashes. Os [relatórios de classificação](https://docs.evidentlyai.com/metrics/preset_classification) usam o controle do mesmo holdout como referência, com gráficos ROC/PR e matrizes de confusão; a validação do treino não é usada como baseline desse experimento. Não há treinamento, recalibração ou mudança de limiar.

Resultados com seed 42, todos com 35.297 registros:

| Cenário | q | Rótulos alterados | Prevalência | AUC ROC | Recall | Precisão | Falsos negativos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| c0_control | 0 | 0 | 6,62% | 0,843257 | 74,09% | 19,74% | 605 |
| c1 | 0,10 | 815 | 8,92% | 0,768076 | 62,22% | 22,37% | 1.190 |
| c2 | 0,25 | 2.062 | 12,46% | 0,710069 | 52,49% | 26,34% | 2.089 |
| c3 | 0,50 | 4.123 | 18,30% | 0,669626 | 45,35% | 33,42% | 3.529 |

**Comportamento observado:** a probabilidade média prevista permanece 0,326790 e a proporção de classes positivas previstas permanece 24,83% em todos os cenários. As probabilidades e classes são idênticas linha a linha; o modelo não tem sinal de entrada que revele o choque. Mesmo assim, em c3 a AUC cai 0,173631 e o recall cai 28,74 pontos percentuais. O Brier piora de 0,147595 para 0,182354 e o log loss de 0,459707 para 0,553312. O KS de classificação cai de 0,533130 para 0,258752. As três intensidades ativam os critérios experimentais de queda absoluta de AUC ≥ 0,05 ou recall ≥ 0,10 em relação ao controle.

A precisão aumenta porque, dentro da mesma lista de previsões positivas, passam a existir mais positivos nos novos rótulos. O F1 também aumenta e a average precision de c3 fica próxima à do controle; essas métricas respondem à prevalência e não demonstram recuperação da discriminação. A ponderação de classes do baseline não garante probabilidades calibradas. Os resultados completos estão em [metrics.csv](../artifacts/concept_drift/metrics.csv).

No segmento afetado, a AUC cai de 0,841789 para 0,548125 e o recall de 81,58% para 37,49%; a prevalência sobe de 8,60% para 53,68%. Fora dele, rótulos e todas as métricas permanecem iguais. Isso localiza o efeito na intervenção programada, sem transformá-la em evidência de vulnerabilidade real de um grupo. Ver [métricas por segmento](../artifacts/concept_drift/segment_metrics.csv).

A sensibilidade usa as seeds 42–46 com os mesmos parâmetros e modelo. Em c3, a AUC varia de 0,669626 a 0,673019 e o recall de 44,75% a 45,84%, mantendo a degradação nas cinco realizações. Esses mínimos/máximos **não são intervalos de confiança** e não validam a hipótese econômica. Ver [sensitivity.csv](../artifacts/concept_drift/sensitivity.csv).

A igualdade exata das features é verificada em todos os períodos. A [comparação Evidently entre c3 e o controle](../artifacts/concept_drift/feature_drift.html) confirma zero features com drift. Monitorar apenas distribuições de entrada e scores não detectaria esse experimento: é necessário acompanhar rótulos maturados, desempenho e erros por segmento. A [análise consolidada](../artifacts/concept_drift/analysis.html) reúne interpretação e links; o [manifesto](../artifacts/concept_drift/manifest.json) registra a procedência. Os dados, modelo e relatórios anteriores permanecem intactos.

Para outra execução, escolher destino inexistente:

```bash
.venv/bin/python -m src.concept_drift --seed 43 --output-dir /tmp/tc04-concept-drift
```

Os argumentos `--reference`, `--holdout` e `--model` permitem indicar versões explícitas das entradas. Uma falha não publica resultados parciais; um destino existente é preservado. O experimento é um teste de estresse sintético, sem uma série temporal real e sem comprovar degradação do modelo em concessões reais. Em produção, o alvo de dois anos exige política de maturação. Antes de qualquer atualização operacional, investigar origem dos rótulos e comparar candidatos em dados independentes; nenhum modelo novo é promovido aqui.
