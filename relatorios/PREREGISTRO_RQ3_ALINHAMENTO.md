# Pré-registro — RQ3, uso A: alinhamento de racionais

> Escrito em 09/10/2026, **antes** de a medida rodar no teste. Só o modo fumaça
> rodou, na **validação**, mostrando um exemplo e nenhuma métrica. Commitar este
> arquivo e o script antes de rodar `scripts/alinhamento_racionais.py`. Nada aqui
> é editado depois; correções vão num adendo datado, no fim.

**Pergunta (RQ3):** as explicações produzidas pelo modelo correspondem ao
raciocínio que o checador usou? Neste uso: as palavras que mais pesaram para o
modelo chamar um texto de FALSO aparecem na justificativa do checador mais do
que palavras quaisquer do mesmo texto?

---

## 1. Desenho fixado

| item | valor |
|---|---|
| modelos | os dois braços do comparativo: TF-IDF (1–2 gramas) + regressão logística, C escolhido na validação, treinados no `treino` do pareado (`braco_refinado.jsonl`) |
| braços | **bruto** (`texto_entrada`) e **refinado** (`ideia_central`) |
| itens | falsas do `teste` com justificativa (2.579 de 2.581) **que o braço acertou** (principal); todas as falsas (sensibilidade) |
| racional do modelo | as k primeiras palavras distintas por contribuição para FALSO (peso × TF-IDF > 0); bigramas contribuem com as duas palavras. Exato para regressão logística, sem LIME ou SHAP |
| racional do checador | palavras da **`justificativa`** (principal: 2 a 4 frases sobre as evidências, resumidas pelo LLM a partir do recorte da apuração); **`justificativa_bruta`** (sensibilidade: trecho literal da apuração, que começa recontando o boato) |
| palavras | minúsculas, só letras, 3 ou mais caracteres, fora de uma lista fixa de palavras vazias (no script) |
| comparação | palavra exata (principal); radical de 5 letras (sensibilidade) |
| k | **5** (principal) e 10 |

**Medida principal:** para cada item, **precisão do modelo** = fração das k
palavras do modelo presentes na justificativa; **controle** = fração das
palavras distintas do próprio texto presentes na justificativa (a esperança
exata de sortear k palavras do texto ao acaso). **Diferença modelo − controle**,
pareada por item, com IC95% por bootstrap (2.000 reamostragens, semente 0).
Itens com menos de k palavras distintas ficam de fora da medida daquele k
(relatado).

**Por que o controle importa:** a justificativa fala do mesmo assunto que o
texto, então qualquer palavra do texto tem alguma chance de aparecer nela. Sem
o controle, 30% de sobreposição não diz se é muito ou pouco.

**Descritivo:** as 30 palavras mais frequentes no top-5 do modelo, por braço.

**Limites declarados agora:**
- A `justificativa` foi escrita pelo LLM a partir da apuração. Ela pode ser
  mais "temática" ou mais "limpa" que o texto do checador. Por isso a
  sensibilidade com o texto literal.
- Sobreposição de palavras é uma medida pobre de "raciocínio": mede se o modelo
  e o checador falam das mesmas coisas, não se raciocinam igual.
- Só falsas: o lado verdadeiro não tem justificativa.

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| A1 | bruto, k = 5: modelo − controle | **≤ 0 pp** (o modelo escolhe palavras de estilo e de chamada, que o checador não usa) |
| A2 | refinado, k = 5: modelo − controle | **0 a +10 pp** |
| A3 | refinado − bruto, na diferença modelo − controle | **≥ +5 pp** |
| A4 | sensibilidade com `justificativa_bruta` | mesma direção de A1 a A3 |
| A5 | top-5 do bruto | dominado por palavras de chamada e de estilo ("compartilhe", "urgente", "atenção", "você") e por nomes políticos |

## 3. Leituras fixadas antes do resultado

| modelo − controle (k = 5, principal) | leitura |
|---|---|
| **≥ +10 pp** | o modelo destaca palavras que o checador também usa, acima do acaso: há alinhamento, ao menos de assunto |
| **−5 a +10 pp** | nenhum alinhamento além do que o assunto do texto já explica |
| **≤ −5 pp** | o modelo decide por palavras que o checador evita: desalinhamento (estilo, chamada, nomes) |

Lida por braço. A comparação entre braços diz se tirar a forma (refinado)
aproxima o modelo do raciocínio do checador. Se a direção mudar entre a
referência principal e a sensibilidade, a conclusão é declarada como dependente
da referência.

## 4. Previsões do pesquisador

> Preencher antes de rodar, em uma linha cada.

Registradas em 09/10/2026, antes da rodada.

- **A1** (bruto: o modelo usa palavras do checador mais, igual ou menos que o acaso?): ___mais que o acaso___
- **A2** (refinado: idem): ___ mais que o acaso___
- **A3** (o refinado se alinha mais que o bruto?): ___sim___
- **A5** (que tipo de palavra domina o top-5 do bruto?): ___Atenção, Aprova, Especialista, porque e nega___

---

## Adendos

### Adendo 1 — 09/10/2026, resultado da rodada

**Execução.** Pré-registro commitado às 01:08 (b328f55). Uma primeira rodada foi
interrompida antes de gravar qualquer resultado: o script reconstruía o
vocabulário do TF-IDF a cada chamada e levaria mais de uma hora. A correção, só
de desempenho (cache, sem mudança de lógica), foi conferida na validação, sem
métricas: 0 divergências em 300 comparações por braço contra a versão original.
Commitada às 01:35 (9f2a692) e rodada em seguida (58 s). Resultado:
`relatorios/resultado_rq3_alinhamento.json`.

**Resultado principal** (falsas do teste acertadas pelo braço; `justificativa`;
palavra exata):

| braço | k | n | modelo | controle | **modelo − controle** |
|---|---:|---:|---:|---:|---:|
| bruto | 5 | 2.361 | 15,3% | 12,5% | **+2,8 pp** [2,2; 3,4] |
| bruto | 10 | 2.081 | 12,9% | 11,0% | +1,9 pp [1,5; 2,3] |
| refinado | 5 | 2.112 | 23,1% | 22,2% | **+0,9 pp** [0,3; 1,6] |
| refinado | 10 | 1.420 | 22,0% | 21,2% | +0,8 pp [0,3; 1,4] |

**Sensibilidade com a `justificativa_bruta`** (texto literal, que reconta o
boato): bruto +8,4 pp [7,6; 9,3] (k = 5), +6,4 pp (k = 10); refinado +3,6 pp
[2,9; 4,3] (k = 5), +2,6 pp (k = 10). Mesma direção da referência principal: os
dois braços ficam acima do controle, e o bruto acima do refinado. A conclusão
não depende da referência. As demais sensibilidades (radical, todas as falsas)
estão no arquivo de resultado.

**Leitura fixada aplicável (§3), nos dois braços: entre −5 e +10 pp → "nenhum
alinhamento além do que o assunto do texto já explica".** As diferenças são
estatisticamente maiores que zero, mas pequenas: das 5 palavras que mais
pesaram para o modelo bruto, 15% aparecem na justificativa do checador, contra
12,5% de palavras quaisquer do mesmo texto. Tirar a forma (braço refinado)
**não** aproximou o modelo do raciocínio do checador; afastou um pouco, e a
diferença em relação ao controle encolhe de 2,8 para 0,9 pp.

**Descritivo — palavras mais frequentes no top-5:**
- bruto: agora, lula, vídeo, bolsonaro, pra, urgente, dinheiro, brasil, deus,
  povo, atenção, gente, notícia, acaba, verdade, apenas, qualquer, homem, nada,
  olha. São chamadas e marcas de corrente ("agora", "urgente", "atenção",
  "gente", "olha") e nomes políticos.
- refinado: bolsonaro, lula, brasil, enquanto, alexandre, texto, moraes,
  presidente, morreu, após, estados, oferece, governo, homem, polícia,
  brasileiro, vídeo, família, morte, gerando. São nomes e assuntos, e de novo
  "texto" (o metatexto do LLM, já visto na C2).

**Confronto com as previsões:**

| | resultado | §2 | pesquisador |
|---|---|---|---|
| A1 bruto, modelo − controle | +2,8 pp | ≤ 0 ✗ | mais que o acaso ✓ (por pouco) |
| A2 refinado | +0,9 pp | 0 a +10 ✓ | mais que o acaso ✓ (por pouco) |
| A3 refinado − bruto | **−1,9 pp** | ≥ +5 ✗ | sim ✗ |
| A4 sensibilidade na mesma direção | sim (direção observada, não a prevista em A1) | parcial | — |
| A5 top-5 do bruto | chamadas, estilo e nomes políticos | ✓ | "atenção, aprova, especialista, porque, nega": só "atenção" aparece; "porque" está na lista de palavras vazias e não poderia aparecer ✗ |

**Leitura para a RQ3:** o classificador acerta por motivos diferentes dos do
checador. O que ele destaca são marcas de corrente e nomes de políticos; o que
o checador escreve são evidências (registros oficiais, datas, origem de
imagens). A sobreposição mal passa do que o assunto do texto explica. Isso liga
a RQ3 à RQ4 (atalhos de forma) e à RQ1 (o refinado aprende assunto): nenhuma das
duas representações chega perto do raciocínio de verificação, o que reforça a
proposta de recuperação de alegações checadas (uso B) como caminho para
explicações ancoradas em evidência.

**Desvios declarados:** (1) a rodada original foi interrompida e o script,
otimizado depois do pré-registro, sem mudança de lógica (equivalência
conferida); (2) as previsões foram escritas pelo pesquisador antes do commit,
mas o campo de data ficou em branco e foi preenchido pelo assistente
(09/10/2026) no próprio commit do pré-registro; (3) a validação manual da
`justificativa` não foi feita, e a concordância com a `justificativa_bruta`
reduz, mas não elimina, essa dependência.
