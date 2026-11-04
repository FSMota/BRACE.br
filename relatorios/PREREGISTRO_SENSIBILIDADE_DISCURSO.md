# Pré-registro — sensibilidade: checagens de discurso (G1 e Lupa) fora do teste

> Escrito em 18/10/2026, **antes** de recalcular qualquer métrica sem essas
> instâncias. Commitar este arquivo e os scripts antes de rodar. Correções vão
> num adendo datado, no fim.

**Origem.** Revisando a Metodologia (18/10/2026), o pesquisador perguntou como
as falsas do G1 foram extraídas. A conferência mostrou que as 57 falsas do G1
Fato ou Fake (24) e da Lupa (33) **não são checagens de boatos**, e sim de
**discursos de políticos** (entrevistas, debates e sabatinas de 2026), em que
cada frase recebe um veredito próprio:
- G1: 13 das 24 vêm de artigos "Veja o que é #FATO e o que é #FAKE na
  entrevista de…"; parte dessas frases é **#FATO** (verdadeira) e entrou como
  FALSO;
- Lupa: vereditos graduados ("erra", "exagera", "omite contexto"), e várias
  entradas são texto do jornalista ("O candidato errou ao afirmar que…").

Todas foram extraídas pela via `llm` (esses portais não marcam a mensagem
num bloco), vêm de só 20 artigos do G1 e 14 da Lupa raspados, e estão todas no
`teste` (07/04/2026 a 25/08/2026): **52 no pareado**, 5 fora. Nenhuma está no
treino, na validação, nem no protótipo de recuperação.

**Decisão do pesquisador (18/10/2026):** marcar, não apagar. As instâncias
ficam no corpus como amostra dos outros portais (registro de auditoria e da
dificuldade de coletar portais confiáveis por paywall e bloqueio de robôs),
marcadas como `checagem_de_discurso` e fora do pareado na versão publicada.

---

## 1. Desenho fixado

| item | valor |
|---|---|
| marcação | `scripts/marcar_checagem_discurso.py` grava `relatorios/marcacoes/checagem_de_discurso.json`: as 57 falsas com `fonte` em {G1 Fato ou Fake, Lupa}, com id, fonte, título da checagem, parte e se está no pareado. A integração ao `fundir_corpus.py` fica para a versão publicada (não muda o corpus usado nos experimentos) |
| medida | `scripts/sensibilidade_discurso.py`: para os 13 modelos da matriz, nos dois braços, as métricas no `teste` **com** e **sem** as 52 instâncias pareadas, a partir das predições já gravadas em `relatorios/matriz/` (nenhum modelo é retreinado: nenhuma instância marcada está no treino ou na validação) |
| métricas | nossa base → nosso teste e Fake.br → nosso teste (acurácia balanceada); diferença com − sem; as afirmações A2, A6, A7 e A8 da matriz recontadas sem as instâncias (A1, A3 e A5 dependem do Fake.br ou de ambas; A3 é recontada também) |
| fora do escopo | RQ2 (só vias html e texto), controle temporal (2014–2020), protótipo de recuperação: não contêm as instâncias. RQ3 uso A: as instâncias são 52 de 2.361 itens; não é recalculada aqui |

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| D1 | maior variação, em módulo, de qualquer métrica de teste | **≤ 0,5 pp** |
| D2 | variação do modelo de referência (P + LR, bruto), nosso teste | **entre −0,3 e +0,3 pp** |
| D3 | leitura de alguma afirmação da matriz muda (adendo 3, 11 válidos) | **nenhuma** |

**Por que:** 52 de 5.162 itens do teste (1,0%), e de uma só classe; a direção
depende de como o modelo classifica frases de político (registro formal, que o
modelo tende a chamar de VERDADEIRO), o que, se for o caso, faria a acurácia
**subir** um pouco sem elas.

## 3. Leituras fixadas antes do resultado

| maior variação | leitura |
|---|---|
| **≤ 0,5 pp** e nenhuma afirmação muda | as checagens de discurso não afetam as conclusões; ficam marcadas e documentadas |
| **> 0,5 pp** ou alguma afirmação muda | os números do TCC passam a ser relatados também sem elas, e a mudança é discutida no texto |

## 4. Previsões do pesquisador

> Preencher antes de rodar, em uma linha cada, com a data.

Registradas em _18_/_10_/2026, antes da rodada.

- **D1** (quanto você acha que os números mudam sem essas 52 instâncias?): ___0,2 - 0,5 pp___
- **D2** (o modelo de referência vai melhor ou pior sem elas?): ___pior mas com diferença baixa <= 0,04 pp___

---

## Adendos

### Adendo 1 — 18/10/2026, resultado da rodada

**Execução.** Pré-registro, marcação e scripts commitados antes (5c8f8c0);
rodada única, em segundos. Saída: `relatorios/resultado_sensibilidade_discurso.json`.
Instâncias marcadas no teste da matriz: 52.

**Resultado.** Todas as 52 variações (13 modelos × 2 braços × 2 métricas de
teste) ficam entre −0,14 e +0,44 pp. A maior é Ch + NB, bruto, nosso teste
(89,95% → 90,39%). O modelo de referência (P + LR, bruto) vai de 91,19% para
91,40% (+0,21 pp) no teste interno e de 78,07% para 78,20% na direção Fake.br →
nossa base. Quase todas as variações são **positivas**: sem as checagens de
discurso, os modelos acertam um pouco mais, isto é, eles tendiam a errar essas
instâncias, frases de político em registro formal rotuladas como FALSO.

**Afirmações da matriz (11 modelos válidos), com → sem:** A2 11 → 11, A3 11 → 11,
A6 5 → 5, A7 8 → 8, A8 9 → 9. Nenhuma leitura muda.

**Leitura fixada aplicável (§3): maior variação ≤ 0,5 pp e nenhuma afirmação
muda → "as checagens de discurso não afetam as conclusões".** Ficam marcadas e
documentadas; os números do TCC continuam os da base com elas, e a versão
publicada as tira do pareado.

**Confronto com as previsões:**

| | resultado | §2 | pesquisador |
|---|---|---|---|
| D1 maior variação | 0,44 pp | ≤ 0,5 ✓ | 0,2–0,5 ✓ |
| D2 modelo de referência | +0,21 pp (melhor sem elas) | −0,3 a +0,3 ✓ | "pior, ≤ 0,04 pp" ✗ (a direção é a oposta) |
| D3 leitura das afirmações | nenhuma mudou | nenhuma ✓ | — |
