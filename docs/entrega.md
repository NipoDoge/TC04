# Entrega, vídeo STAR e checklist

[Voltar ao README](../README.md)

**Link do vídeo: pendente de gravação e publicação pelo grupo.** Substituir esta indicação por um link acessível antes da entrega. Duração máxima: 5 minutos.

[Roteiro detalhado em Word](roteiro-video-star.docx): passo a passo com tempos, falas sugeridas, telas, destaques visuais e checklist de gravação.

| Tempo | STAR | Conteúdo sugerido |
| --- | --- | --- |
| 0:00–0:40 | Situation | Cenário da fintech, risco de degradação silenciosa e impacto sobre crédito |
| 0:40–1:10 | Task | Validar dados, detectar drift, observar o pipeline e definir governança |
| 1:10–3:40 | Action | Mostrar contrato, bloqueio/quarentena, modelo, simulação, predições, relatórios de data drift e degradação sob concept drift; demonstrar logs reais, duração, recursos, falhas e recuperação no painel; explicar privacidade e retenção |
| 3:40–5:00 | Result | Mostrar métricas e alertas efetivamente gerados, interpretação, limitações causais, mitigação de vieses e resposta proposta |

### Antes da entrega

- [x] Documentar arquitetura e distinguir controles implementados de políticas propostas.
- [x] Explicar tratamento de dados, base legal proposta, minimização e retenção.
- [x] Documentar riscos de viés, hipóteses causais e plano de resposta.
- [x] Registrar fonte e data de obtenção no [README](../README.md#governança-e-privacidade).
- [x] Identificar a cópia de treino pelo SHA-256 do manifesto da Etapa 1 e explicitar que a versão externa não foi informada.
- [x] Documentar o [enquadramento acadêmico](governanca.md#enquadramento-acadêmico), com finalidade, atividades, público e limites do estudo.
- [x] Registrar a validação do enquadramento acadêmico pelo grupo.
- [x] Documentar o escopo e as diretrizes da [revisão da publicação](governanca.md#revisão-da-publicação).
- [x] Documentar as responsabilidades funcionais do plano de governança.
- [x] Executar a demonstração e guardar evidências de contrato, quarentena, modelo e métricas.
- [x] Gerar cenários sintéticos de data drift e predições com o modelo salvo.
- [x] Executar a detecção estatística e gerar os relatórios HTML de data drift da Etapa 2.
- [x] Executar experimento próprio de concept drift em `P(Y|X)` e analisar o modelo com features fixas.
- [x] Implementar a observabilidade da Etapa 3 com logs reais, recursos, histórico e alertas de desempenho.
- [x] Demonstrar falhas operacionais, bloqueio por contrato e recuperação em execução isolada.
- [x] Atualizar a interpretação dos experimentos com resultados observados e distinguir intervenção sintética de causa econômica real.
- [x] Registrar a conclusão da revisão dos dados e materiais do projeto, conforme confirmação do grupo.
- [ ] Gravar, revisar e publicar o vídeo STAR e inserir o link nesta seção.

A Etapa 1, os experimentos de data drift e concept drift da Etapa 2 e a observabilidade da Etapa 3 (logs reais por execução + `make observe` + [dashboard](../reports/observability/index.html); [métricas documentadas](observabilidade.md)) possuem execução e evidências preservadas; o [plano de governança](governanca.md) cobre a proposta teórica da Etapa 4. A fonte, a data de obtenção e a identificação da cópia de treino estão documentadas. O grupo confirmou a validação do enquadramento acadêmico e a conclusão da revisão dos dados; as responsabilidades funcionais estão descritas no plano. A pendência deste checklist é gravar, revisar e publicar o vídeo STAR, inserindo seu link acessível na documentação.
