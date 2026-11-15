# Comparativo dos perfis de data drift

[Voltar ao README](../README.md)

O perfil anterior, com renda ×0,90/0,75/0,60 e dívida ×1,10/1,30/1,60, foi preservado em [artifacts/data_drift_income_debt](../artifacts/data_drift_income_debt/archive.json), com seus 21 arquivos de cenários, predições e relatórios e o commit de origem. Os manifestos históricos mantêm os caminhos da execução antiga; `archive.json` mapeia cada caminho original para o snapshot e seu hash. Esse material é evidência histórica; o gerador ativo produz somente utilização de crédito/atrasos de 90 dias.

[compare_data_drift.py](../src/compare_data_drift.py) exige o mesmo modelo e holdout, confere hashes, ordem, rótulos e predições, recalcula os resultados salvos e avalia ambos os perfis no limiar 0,5. As features fora da dupla de intervenção devem permanecer idênticas. A avaliação inclui AUC, recall, precisão, F1, average precision, KS de classificação, Brier, log loss, matriz de confusão, scores e classes alteradas, além de importância por ganho e permutação com dez repetições.

```bash
make compare-data-drift PYTHON=.venv/bin/python
```

Os resultados ficam em [reports/drift_comparison/analysis.html](../reports/drift_comparison/analysis.html), com métricas, importância individual e conjunta, 80 resultados estatísticos de drift dos dois perfis e comparações de classificação do Evidently em t3. A referência de treino não é usada para medir desempenho: as métricas são do holdout pareado. Para outro destino, usar `python -m src.compare_data_drift --output-dir /tmp/tc04-comparison`.

| Perfil | Cenário | AUC | Recall | Precisão | Previsões positivas | Classes alteradas contra controle |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Controle comum | t0 | 0,843257 | 74,09% | 19,74% | 24,83% | 0 |
| Renda/dívida anterior | t1 | 0,843497 | 75,20% | 19,34% | 25,73% | 608 |
| Renda/dívida anterior | t2 | 0,844089 | 76,62% | 18,58% | 27,28% | 1.210 |
| Renda/dívida anterior | t3 | 0,842989 | 77,94% | 17,75% | 29,05% | 1.950 |
| Utilização/atrasos atual | t1 | 0,826368 | 78,67% | 15,53% | 33,52% | 3.200 |
| Utilização/atrasos atual | t2 | 0,810124 | 84,03% | 12,19% | 45,61% | 7.462 |
| Utilização/atrasos atual | t3 | 0,800850 | 88,27% | 10,29% | 56,74% | 11.377 |

Em t3, os falsos positivos são 8.435 no perfil anterior e 17.965 no atual; a taxa de falsos positivos passa de 25,59% para 54,50%. O aumento de recall ocorre junto com mais previsões positivas e menor precisão. O Brier piora de 0,168535 para 0,316147 e o log loss de 0,514014 para 0,885914 entre os dois perfis de t3. A variação média absoluta do score contra o controle é 0,044117 no anterior e 0,193918 no atual. O modelo é mais sensível aos choques atuais escolhidos, com pior discriminação e mais falsos positivos contra os rótulos históricos.

No holdout original, utilização e atrasos de 90 dias ocupam a 1ª e a 2ª posição por importância de permutação: quedas médias de AUC de 0,065950 e 0,032620. Dívida e renda ocupam a 5ª e a 8ª posição, com 0,008679 e 0,003704. A permutação conjunta, preservando a relação interna de cada dupla e usando as mesmas dez ordens, reduz a AUC em média 0,123676 na dupla atual e 0,012697 na anterior. Ganhos normalizados de 28,29% e 21,71% são indicadores internos das árvores, não proporções do risco explicado. Ver [feature_importance.csv](../reports/drift_comparison/feature_importance.csv) e [joint_importance.csv](../reports/drift_comparison/joint_importance.csv).

**Limites da comparação:** o modelo e a amostra são fixos, mas as doses e unidades das intervenções são diferentes. Nomes t1/t2/t3 não significam choques economicamente equivalentes. Portanto, a diferença dos efeitos não pode ser atribuída somente ao ranking de importância. Com os rótulos antigos, trata-se de sensibilidade sintética, não uma estimativa de desfechos após uma mudança real. Os desvios padrão das permutações não são intervalos de confiança. O [experimento separado de concept drift](concept-drift.md) mantém seu mecanismo de renda, seus dados e seus resultados anteriores.
