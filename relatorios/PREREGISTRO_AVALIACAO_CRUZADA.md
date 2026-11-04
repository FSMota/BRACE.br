# Pré-registro — avaliação cruzada no Fake.br (braço bruto)

> Escrito em 29/09/2026, **antes** de qualquer texto do Fake.br passar pelo modelo.
> Commitar este arquivo antes de rodar: o histórico do git é a prova da data.
> Nada aqui pode ser editado depois da rodada. Correções vão num adendo datado, no fim.

Por que existe: o mesmo autor escreveu o pipeline e vai interpretar o resultado.
Fixar desenho, previsões e leituras antes impede que o resultado escolha a
interpretação (e que a interpretação escolha o desenho).

---

## 1. Desenho fixado

**Modelo.** O mesmo do `scripts/modelo_bruto.py`, rodada A: TF-IDF de palavras
(1–2 gramas, `min_df=3`, `sublinear_tf`) + regressão logística, `C=10`,
treinado no `treino` do `subconjunto_pareado` do `corpus_fundido.jsonl`.
**Nenhum ajuste usando o Fake.br**: nem C, nem limiar de decisão, nem vocabulário.

**Dados de teste.** Fake.br, pasta `full_texts`, os 3.600 pares (7.200 textos;
numeração 1–3602 sem 697 e 1468). Nenhum texto do Fake.br é excluído, exceto
arquivo ilegível — e, se houver, a exclusão é relatada.

**Entrada A — texto.** Nas DUAS classes: os primeiros caracteres do texto, com
quebras de linha e tabulações trocadas por espaço, cortados no primeiro fim de
frase (`.`, `!`, `?`) depois de 250 caracteres; se não houver, corte em 300. A
mesma `normalizar_fonte()` do `fundir_corpus.py` ("hoje (12)" → "hoje";
menções à Agência Brasil). Maiúsculas e pontuação intactas.

Justificativa: as falsas do Fake.br trazem título na 1ª linha e as verdadeiras
às vezes não; usar "a primeira linha" seria assimétrico entre as classes. O
tamanho aproxima a nossa entrada (medianas 290 e 241 caracteres).

**Entrada B — título.** Só os pares em que os DOIS lados têm título
identificável: 1ª linha terminada por quebra de linha, ou texto antes do
primeiro `.` seguido de dois espaços, com 20 a 200 caracteres. O número de
pares que satisfazem a regra é relatado antes do resultado.

**Métrica principal.** Acurácia balanceada. **Secundárias:** revocação de cada
classe; acurácia por categoria do Fake.br.

---

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| P1 | acurácia balanceada, entrada A (texto) | **55% – 70%** |
| P2 | queda em relação ao teste interno (92,05%) | **≥ 20 pontos** |
| P3 | revocação de FALSO **menor** que a de VERDADEIRO, entrada A | diferença **≥ 20 pontos** |
| P4 | entrada B (título) acima da entrada A | **≥ 5 pontos** acima |
| P5 | pior categoria: `politica` ou `sociedade_cotidiano`; melhor: `tv_celebridades` | — |

**Por que cada uma.**
- P1/P2: a leitura dos pesos do modelo interno é que os 92% vêm principalmente
  de gênero (corrente × agência). As falsas do Fake.br têm gênero de notícia.
- P3: é a consequência direta. Uma matéria de blog hiperpartidário tem o
  registro jornalístico ("disse", "afirmou", "segundo") que o modelo associou
  a VERDADEIRO, então deve ser chamada de verdadeira com frequência.
- P4: o título do blog ("Dilma se irrita e confessa crime") é sensacionalista,
  a parte do texto mais parecida com uma mensagem viral.
- P5: celebridade é o balde em que boato e site falso mais se parecem
  (morte, escândalo); política do Fake.br é notícia real com viés.

**Hipótese para depois (RQ1), registrada agora:** no braço refinado
(`ideia_central` com prompt cego nos dois corpora), a queda da avaliação
cruzada será **menor** que a do braço bruto.

---

## 3. Leituras fixadas antes do resultado

| resultado (entrada A) | leitura |
|---|---|
| **≥ 85%** | refuta a leitura principal: há sinal que transfere além do gênero. Os 92% internos NÃO se explicam só por corrente × agência. |
| **70% – 85%** | transferência parcial: parte do sinal é de conteúdo ou de sensacionalismo, parte de gênero. |
| **55% – 70%** | confirma: o modelo aprendeu majoritariamente gênero e fonte. |
| **≤ 55%** | nada transfere. Não separa "aprendeu só gênero" de "o Fake.br mede outra definição de falsa" — o braço refinado e a entrada B passam a ser os testes decisivos. |
| **< 45%** | o modelo inverte as classes no Fake.br: o registro jornalístico das falsas de lá o engana sistematicamente. Evidência forte de dependência de gênero. |

**Limite de interpretação, declarado agora:** o Fake.br usa outra definição de
notícia falsa (matéria de site de desinformação, não alegação checada). Queda
na avaliação cruzada é compatível com as duas explicações — atalho de gênero E
mudança de definição. Nenhum resultado da entrada A sozinho separa as duas.

---

## 4. Previsões do pesquisador

> Preencher antes de rodar, em uma linha cada. Divergência com as previsões
> acima é informação: se o resultado ficar do lado do pesquisador, vale saber.

Registradas em 29/09/2026, antes da rodada.

- **P1** (acurácia balanceada, texto): **~55%** — pela diferença de formato entre as notícias.
- **P2** (queda em relação aos 92,05%): **~35 pontos** (maior que a prevista na seção 2).
- **P3** (revocação FALSO vs. VERDADEIRO): **parecidas** — diverge da seção 2, que prevê FALSO ≥ 20 pontos abaixo.
- **P4** (título vs. texto): **título PIOR que texto** — diverge da seção 2, que prevê título ≥ 5 pontos acima.

Onde as duas previsões divergem (P3, P4), o resultado decide qual leitura do
modelo estava mais certa.

---

## Adendos

### Adendo 1 — 29/09/2026, resultado da rodada

Script: `scripts/avaliacao_cruzada.py`. Saída completa:
`relatorios/resultado_avaliacao_cruzada.json`. Teste interno de conferência:
92,05% (igual à referência). Fake.br: 3.600 pares, 0 ilegíveis.

**Ambiguidades do desenho, resolvidas no código (sem ver resultado):**
1. A normalização foi aplicada também à entrada B (o pré-registro só a fixava
   para a A), para as duas entradas terem o mesmo pré-processamento.
2. "Primeiro fim de frase depois de 250 caracteres" foi lido literalmente. As
   medianas resultantes (315 falsas, 330 verdadeiras) mostram que a leitura não
   distorceu o tamanho da entrada.

**Resultado e confronto:**

| | resultado | seção 2 | pesquisador |
|---|---|---|---|
| P1 acurácia balanceada, texto | **73,03%** | 55–70% ✗ | ~55% ✗ |
| P2 queda vs. interno | **19,0 pp** | ≥ 20 pp ✗ | ~35 pp ✗ |
| P3 rev. VERDADEIRO − rev. FALSO | **+4,9 pp** (75,50% vs. 70,56%) | ≥ +20 pp ✗ | parecidas ✓ |
| P4 título − texto | **−5,6 pp** (67,44% vs. 73,03%) | ≥ +5 pp ✗ | título pior ✓ |
| P5 categorias | melhor `tv_celebridades` 79,5%; pior `ciencia_tecnologia` 61,6% (n=112); entre as grandes, pior `sociedade_cotidiano` 67,1% | melhor ✓, pior ✗ | — |

**Leitura pré-registrada aplicável (entrada A = 73,03%):** transferência
parcial — parte do sinal é de conteúdo ou sensacionalismo, parte de gênero.

**O que o resultado contraria.** A leitura da seção 2 — os 92% internos vêm
*principalmente* de gênero — era forte demais. O mecanismo previsto em P3 (o
registro jornalístico das falsas do Fake.br levaria o modelo a chamá-las de
verdadeiras) não se confirmou: as duas classes são reconhecidas em taxas
parecidas.

**Ressalvas sobre a entrada B, observadas depois do resultado:**
- Só 946 de 3.600 pares (26%) passaram na regra de título (falsas 3.388,
  verdadeiras 1.126). P4 compara 3.600 pares (A) com 946 pares selecionados
  (B) — não é o mesmo conjunto.
- Nas verdadeiras, a regra pode ter capturado o primeiro parágrafo em vez do
  título. A revocação de VERDADEIRO na entrada B (51,90%) é compatível com isso.
  Por essas duas razões, P4 não deve ser citado como "título pior que texto"
  sem a análise exploratória 2.

**Análises EXPLORATÓRIAS decididas depois de ver o resultado** (não
pré-registradas; relatar como tal):
1. *Alvo político.* Hipótese levantada após o resultado: a parte que
   transferiu pode ser o alvo político comum às duas bases de falsas ("lula",
   "dilma", "pt" estão entre os termos de maior peso para FALSO no modelo
   interno; as falsas do Fake.br são de blogs hiperpartidários anti-PT). Se
   for isso, o modelo aprendeu "texto que ataca o PT", não "falso" — um
   confundidor político.
2. *Título e texto no mesmo conjunto.* Entrada A restrita aos 946 pares da
   entrada B, e inspeção do que a regra de título capturou nas verdadeiras.

Script: `scripts/exploratoria_cruzada.py`.

### Adendo 2 — 29/09/2026, análises exploratórias

Script `scripts/exploratoria_cruzada.py`, commitado (1d5f75f) com hipóteses,
alfa (0,05) e correção (Holm sobre T2–T4) fixados ANTES de rodar. Continua
exploratório: a escolha dos testes veio de olhar o resultado do adendo 1.
Saída: `relatorios/resultado_exploratoria_cruzada.json`.

| teste | resultado | p (Holm) | leitura |
|---|---|---|---|
| T1 acaso | 73,03%, IC95% [72,06%; 73,94%] | — | transferência real, bem acima de 50% |
| T2 nomes políticos apagados (M0 × M1) | Fake.br 73,03% → 71,69%; **−1,33 pp, IC95% [0,74; 1,92]**; interno 92,05% → 91,90% | 5,2·10⁻⁵ | significativo, mas **pequeno**: os nomes explicam ~1,3 dos ~23 pp acima do acaso |
| T3 com × sem menção política | 77,76% × 70,43%; **+7,33 pp**; revocação VERDADEIRO 77,85% × 74,90% | 0,0010 | textos políticos são mais FÁCEIS; o diagnóstico do confundidor (revocação de VERDADEIRO caindo com menção) **não** aparece |
| T4 título × texto, mesmos 946 pares | 67,44% × 70,14%; **−2,70 pp** | 0,016 | título pior também na comparação justa (previsão do pesquisador) |

Prevalência de menção política (lula, dilma, pt…): treino interno 16,7% das
falsas × 6,0% das verdadeiras; Fake.br 39,3% × 20,2%. O desequilíbrio existe
nas duas bases, mas o modelo pouco se apoia nele.

**Conclusões desta rodada:**
- A hipótese "a transferência é o alvo político" é pouco sustentada quando
  medida por nomes. Limite: viés político expresso por outros termos
  ("esquerda", "corrupção", "Globo", "STF") não foi medido.
- A ressalva do adendo 1 sobre a regra de título capturar parágrafo em vez de
  título não se confirmou nos exemplos inspecionados (títulos reais).
- O sinal que transfere (~23 pp acima do acaso) segue sem explicação. Hipótese
  de trabalho: estilo opinativo e sensacionalista comum a correntes e blogs
  hiperpartidários. Teste previsto: o braço refinado (RQ1).
- Lição de método: relatar tamanho de efeito com intervalo, não só p-valor —
  T2 tem p = 5·10⁻⁵ e efeito de 1,3 pp.

### Adendo 3 — 29/09/2026, teste 2 (modelo só de estilo) — ANTES de rodar

Script `scripts/modelo_estilo.py` (commit 2bdbe86), exploratório. Previsões
do cabeçalho: E1 interno 78–88%; E2 Fake.br 60–68%; E3 atributo de maior peso
para FALSO = caixa alta ou "!". Leituras: Fake.br ≥ 70% → o estilo explica a
transferência; 58–70% → explica boa parte; ≤ 58% → não é o que transfere.

**Previsões do pesquisador:** iguais às do cabeçalho (E1, E2 e E3).

**Resultado (rodado no Python do WSL — no Windows a política de Controle de
Aplicativo bloqueia a DLL de árvores do scikit-learn; mesma versão 1.9.1).**
Saída: `relatorios/resultado_modelo_estilo.json`.

| | regressão logística | gradient boosting | previsão | |
|---|---|---|---|---|
| interno | 86,11% | **88,42%** | 78–88% | ✗ (por 0,42 pp) |
| Fake.br | 65,03% [64,07; 65,96] | **65,79%** [64,78; 66,81] | 60–68% | ✓ |
| rev. FALSO / VERD. no Fake.br | 40,17% / 89,89% | 77,06% / 54,53% | — | |
| maior peso para FALSO (RL) | `palavras` +6,99 (≈ `exclamacao_100c` +6,96) | — | caixa alta ou "!" | ✗ |

Leitura fixada aplicável (58–70%): **o estilo explica boa parte da
transferência; o resto é léxico/conteúdo.**

Observações:
- Só com 17 atributos de forma, o acerto interno é 88,42% — 3,6 pp abaixo do
  TF-IDF de palavras (92,05%). Quase toda a separação interna é de forma.
- No Fake.br, o estilo leva a 65,79% contra 73,03% das palavras: **~16 dos ~23
  pp acima do acaso (≈ 2/3) são estilo; ~7 pp vêm do léxico.**
- `palavras` (+) e `log_caracteres` (−) são colineares — juntos codificam
  densidade de palavras por caractere; os pesos individuais não devem ser
  lidos isoladamente. E3 conta como ✗ pela letra da previsão.
- As diferenças de estilo apontam na mesma direção nas duas bases, mas são
  muito menores no Fake.br (maiúsculas 13,7% × 4,5% aqui; 5,8% × 5,0% lá).
  Por isso os dois modelos calibram o limiar de formas opostas (revocações
  invertidas) com acurácia balanceada parecida.
- "você" inverte de direção no Fake.br (falsas 0,075 × verdadeiras 0,104) —
  marca de corrente de WhatsApp que não existe em blog.

### Adendo 4 — 04/10/2026, mesmo desenho no corpus com a extração v2

O lado falso foi reprocessado (extração v2: introdução do checador, "Versão N"
e lixo eliminados na origem; 3.022 registros novos, literalidade 97,2%). Corpus:
15.450 falsas + 13.146 verdadeiras; pareado 12.319 + 12.319. Os cinco scripts
foram rodados de novo, **sem nenhuma alteração de desenho, modelo ou
parâmetro**. Os `resultado_*.json` foram sobrescritos; os valores dos adendos
1–3 continuam no histórico do git (e295bd6, a7b2660, 024a76e).

| | adendos 1–3 (base v1) | **base v2** |
|---|---:|---:|
| interno (A / B sem veredito) | 92,05% / 91,81% | **91,73% / 91,57%** |
| rótulos embaralhados | 50,00 / 49,64 / 49,76% | **50,86 / 51,20 / 49,51%** ✓ |
| sobreposição no pareado | 0 (4 inícios iguais) | **0 (3 inícios iguais)** |
| **Fake.br, entrada A** | 73,03% [72,06; 73,94] | **73,35% [72,38; 74,31]** |
| queda interno → cruzado | 19,0 pp | **18,4 pp** |
| revocação FALSO / VERDADEIRO | 70,56 / 75,50% | **70,72 / 75,97%** |
| título × texto (946 pares) | 67,44 × 70,14%, p = 0,016 | **67,76 × 70,19%, p = 0,033** |
| T2 nomes políticos (M0 − M1) | 1,33 pp [0,74; 1,92] | **1,52 pp [0,93; 2,08]** |
| T3 com × sem menção | +7,33 pp | **+7,56 pp** (p Holm 0,001) |
| só forma, interno (GB) | 88,42% | **87,76%** |
| só forma, Fake.br (GB) | 65,79% [64,78; 66,81] | **65,46% [64,43; 66,49]** |

**Leituras pré-registradas aplicáveis — inalteradas:** entrada A na faixa 70–85%
(transferência parcial); estilo na faixa 58–70% (explica boa parte: 15,5 dos
23,4 pp acima do acaso, ≈ 2/3).

**Confronto com as previsões da seção 2 e do pesquisador — inalterado:** P1, P2,
P3 e P4 da seção 2 continuam ✗ (P2 por 1,6 pp); P3 e P4 do pesquisador continuam
✓. No teste 2, o atributo de maior peso passou a ser `exclamacao_100c` (+7,39,
antes `palavras`); como `palavras` e `log_caracteres` são colineares, a troca de
posição não deve ser interpretada — E3 continua registrado como ✗ na rodada
original.

**Conclusão:** a limpeza da extração não alterou nenhuma conclusão. Os números
do braço bruto não se apoiavam nos defeitos de extração; o termo "versão" saiu
da lista de maior peso para FALSO. Estes passam a ser os números de referência
do braço bruto.

### Adendo 5 — 06/10/2026, mesmo desenho no corpus com a extração v3

A auditoria de vocabulário dirigida por dados (`scripts/auditar_vazamento.py`)
mostrou que 873 falsas do pareado (7,2%; 844 no treino) eram o TÍTULO do artigo
de checagem com a etiqueta de veredito "#boato", não a mensagem viral. A
extração v3 recusa a voz do checador (título, assinatura, narração) nas três
vias; 1.419 + 895 artigos reprocessados (uma rodada saiu com o código anterior e
foi refeita). Corpus: 14.856 falsas (v1 10.902, v2 3.243, v3 711) + 13.146
verdadeiras; pareado 11.975 + 11.975. Os ~990 registros a menos são artigos sem
mensagem transcrita, em que o título era tudo o que havia. Os cinco scripts
rodados de novo, **sem alteração de desenho, modelo ou parâmetro**; os valores
do adendo 4 ficam no histórico do git (5b8e739).

| | adendo 4 (base v2) | **base v3** |
|---|---:|---:|
| "#boato" / assinatura nas entradas falsas | 7,2% / 1,6% | **0% / 0%** |
| falsas marcadas com veredito (corpus) | 1.545 | **121** |
| interno A / B (sem veredito) | 91,73% / 91,57% | **91,19% / 91,47%** |
| peso de "boato" (posição entre os termos de FALSO) | 21,0 (1º) | **7,2 (7º)** |
| rótulos embaralhados | 50,86 / 51,20 / 49,51% | **51,38 / 51,07 / 48,14%** ✓ |
| sobreposição no pareado | 0 (3 inícios iguais) | **0 (2 inícios iguais)** |
| **Fake.br, entrada A** | 73,35% [72,38; 74,31] | **73,51% [72,54; 74,44]** |
| queda interno → cruzado | 18,4 pp | **17,7 pp** |
| revocação FALSO / VERDADEIRO | 70,72 / 75,97% | **72,75 / 74,28%** |
| título × texto (946 pares) | 67,76 × 70,19%, p = 0,033 | **67,18 × 70,45%, p = 0,003** |
| T2 nomes políticos (M0 − M1) | 1,52 pp [0,93; 2,08] | **2,04 pp [1,51; 2,60]** |
| T3 com × sem menção | +7,56 pp | **+5,97 pp** (p Holm 0,001) |
| só forma, interno / Fake.br (GB) | 87,76% / 65,46% | **87,21% / 65,18% [64,18; 66,17]** |

**Leituras pré-registradas aplicáveis — inalteradas:** entrada A na faixa 70–85%
(transferência parcial); estilo na faixa 58–70% (15,2 dos 23,5 pp acima do
acaso, ≈ 2/3). **Confronto com as previsões — inalterado** (P1–P4 da seção 2 ✗;
P3 e P4 do pesquisador ✓). No teste 2, o atributo de maior peso é agora
`exclamacao_100c`; E3 continua registrado como ✗ na rodada original (colinearidade
entre `palavras` e `log_caracteres`).

**Conclusão:** o título com "#boato" era vazamento de rótulo na entrada — mas
quase todo no treino, e por isso não inflava a nota do teste: com ele fora, o
interno caiu 0,5 pp e a rodada sem veredito passou a ficar *acima* da completa.
O que mudou foi o que o modelo aprende ("boato" de 1º para 7º termo). Nenhuma
conclusão sobre o braço bruto mudou. Estes passam a ser os números de
referência do braço bruto.
