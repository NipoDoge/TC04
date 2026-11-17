# Entrega, vídeo STAR e checklist

[Voltar ao README](../README.md)

**Link do vídeo: pendente de gravação e publicação pelo grupo.** Substituir esta indicação por um link acessível antes da entrega. Duração máxima: 5 minutos.

| Tempo | STAR | Conteúdo sugerido |
| --- | --- | --- |
| 0:00–0:40 | Situation | Cenário da fintech, risco de degradação silenciosa e impacto sobre crédito |
| 0:40–1:10 | Task | Validar dados, detectar drift, observar o pipeline e definir governança |
| 1:10–3:40 | Action | Mostrar contrato, bloqueio/quarentena, modelo, simulação, predições, relatórios de data drift e degradação sob concept drift; após implementação, demonstrar observabilidade; explicar privacidade e retenção |
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
- [x] Executar experimento próprio de concept drift em `P(Y|X)` e analisar o modelo com features fixas.
- [x] Implementar a observabilidade da Etapa 3.
- [x] Atualizar a interpretação dos experimentos com resultados observados e distinguir intervenção sintética de causa econômica real.
- [ ] Revisar os dados e valores expostos no repositório, relatórios e vídeo.
- [ ] Publicar o vídeo STAR e inserir o link nesta seção.

A Etapa 1, os experimentos de data drift e concept drift da Etapa 2 e a observabilidade da Etapa 3 (Opção A: `make observe` + [dashboard](../reports/observability/index.html); [métricas documentadas](observabilidade.md)) possuem execução e evidências preservadas; o [plano de governança](governanca.md) cobre a proposta teórica da Etapa 4. As pendências de governança (procedência/responsáveis) e o vídeo ainda precisam ser concluídos para a entrega completa do projeto.
