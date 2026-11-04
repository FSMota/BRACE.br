# Pré-registro — pareamento datado × não datado (controle temporal)

> Escrito em 09/10/2026, **antes** de qualquer verdadeira não datada ser sorteada
> ou coletada. Commitar este arquivo antes do sorteio: o histórico do git é a
> prova da data. Nada aqui é editado depois; correções vão num adendo datado, no
> fim.

**Origem:** sugestão da orientadora (09/10/2026): comparar o balanceamento da
base com e sem datação e levantar uma pergunta de pesquisa a partir disso.

**Pergunta (RQ5, ou parte da RQ4, a decidir na orientação):** quanto da acurácia
de um corpus montado **sem controle de data** vem da época, e não da falsidade?

---

## 0. Por que não basta tirar o ano do pareamento atual

A datação está na **coleta**, não só no pareamento: as verdadeiras foram
sorteadas por célula tema × ano para acompanhar as falsas
(`scripts/coletar_verdadeiras.py`, etapa 2V). Por isso, na base atual, um
classificador que só vê o ano chega a **51,5%**. Refazer o pareamento sem o ano,
sobre as mesmas verdadeiras, quase não mudaria nada.

A comparação informativa é entre **coletas**: verdadeiras sorteadas por ano
(como foi feito) × verdadeiras sorteadas do arquivo da Agência Brasil **sem
olhar a data**, que é o que acontece quando uma base não controla a época.
Estimativa feita antes deste pré-registro, só com contagens: no sitemap de
2014–2020, os anos de 2014 a 2018 somam cerca de 75% das matérias, enquanto as
falsas se concentram em 2019 e 2020. A melhor regra que olha só o ano
acertaria cerca de **67,7%** nesse cenário.

## 1. Desenho fixado

| item | valor |
|---|---|
| janela | **2014 a 2020**. É o período em que temos a distribuição natural do arquivo (sitemap, 150.641 URLs). As URLs de 2021 em diante vieram do Internet Archive, já buscadas por tema e ano, e não servem de amostra natural |
| falsas | **2.500** sorteadas (semente 0) das 5.085 falsas do `subconjunto_pareado` publicadas de 2014 a 2020. **As mesmas nas duas variantes** |
| variante **D** (datada) | para cada falsa sorteada, uma verdadeira do pareado atual da mesma célula **tema × ano**, por sorteio (semente 0). Nenhuma coleta nova |
| variante **N** (não datada) | para cada tema, tantas verdadeiras quanto falsas daquele tema, sorteadas (semente 0) entre **todas** as URLs do sitemap de 2014–2020 daquele tema, **sem olhar o ano**. URL já coletada é reaproveitada; as demais são coletadas e preparadas pelo mesmo pipeline (título + lide, `preparar_verdadeiras.py`, `normalizar_fonte`). Falha de coleta é substituída pela próxima URL do mesmo ranking, e o número de substituições é relatado |
| tema | o `tema_pareamento` das falsas; nas verdadeiras, a editoria da URL via `taxonomia.pareamento_da_editoria` (o mesmo critério da coleta original) |
| divisão | **aleatória**, 80/10/10, estratificada por classe, semente 0. Toda a janela cai no `treino` da divisão temporal atual, por isso a divisão temporal não serve aqui. Cada falsa fica na mesma parte nas duas variantes |
| modelo | TF-IDF (1–2 gramas, `min_df=3`, sublinear) + regressão logística, C na validação de cada variante, braço **bruto** (`texto_entrada`) |
| classificador só de ano | regressão logística sobre o ano (uma coluna por ano), mesma divisão |

**Medidas** (acurácia balanceada, IC95% por bootstrap, 2.000 reamostragens, semente 0):

| | o que mede |
|---|---|
| T1 | classificador só de ano, no teste de cada variante: o tamanho do atalho temporal |
| T2 | TF-IDF, N → N menos D → D: quanto a época infla a acurácia interna |
| T3 | **N → teste de D** menos D → D: o quanto o modelo sem datação erra quando a época deixa de ajudar. É a medida principal |
| T4 | D → teste de N menos D → D: o modelo datado não deveria depender da época |
| T5 | N → Fake.br menos D → Fake.br |
| descritivo | as 30 palavras de maior peso para VERDADEIRO e para FALSO em cada variante, e quantas delas são marcas de época (nomes de governantes, "covid", "pandemia", eventos datados), contadas por uma lista fixada antes de ver os pesos (§1.1) |

Nas medidas cruzadas (T3, T4), sai do teste da outra variante qualquer
verdadeira que esteja no treino do modelo avaliado.

### 1.1 Lista fixa de marcas de época

dilma, temer, bolsonaro, lula, haddad, aécio, cunha, impeachment, lava, jato,
copa, olimpíada, olimpíadas, zika, covid, coronavírus, pandemia, quarentena,
cloroquina, vacina, vacinas, eleição, eleições, 2014, 2015, 2016, 2017, 2018,
2019, 2020.

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| T1 | só de ano: D | **48% a 55%** |
| T1 | só de ano: N | **60% a 72%** |
| T2 | N → N − D → D | **+2 a +8 pp** |
| T3 | N → teste de D − D → D | **≤ −5 pp** |
| T4 | D → teste de N − D → D | **entre −3 e +3 pp** |
| T5 | N → Fake.br − D → Fake.br | **≤ −2 pp** (as falsas do Fake.br são de 2016–2018, a época que N associa a VERDADEIRO) |

## 3. Leituras fixadas antes do resultado

| resultado | leitura |
|---|---|
| T1 (N) ≥ 60% **e** T3 ≤ −5 pp | sem datação, a época vira um atalho mensurável que infla a acurácia e não transfere: **o pareamento datado é necessário** |
| T1 (N) ≥ 60% e T3 > −5 pp | a época separa as classes, mas o modelo de texto não se apoia nela: **atalho disponível, pouco usado** |
| T1 (N) < 55% | nesta janela, a coleta sem data não cria desequilíbrio temporal relevante: **a datação não fez diferença aqui** |
| entre 55% e 60% | inconclusivo, e assim declarado |

**Limites declarados agora:**
- só 2014–2020, só o braço bruto, só um tipo de modelo (a matriz com outros
  modelos, se aprovada, entra em outro pré-registro);
- divisão aleatória, diferente do resto do TCC;
- 2.500 pares é menos que a base inteira. Para efeitos do tamanho previsto
  (≥ 5 pp), o IC esperado no teste (cerca de 500 itens) é de uns ±4 pp;
- a coleta nova leva horas (cerca de 2.200 matérias, pela taxa da coleta
  original) e pode falhar mais nas matérias antigas. As substituições são
  relatadas por ano.

## 4. Previsões do pesquisador

> Preencher antes do sorteio, em uma linha cada.

Registradas em _13_/_10_/2026, antes do sorteio.

- **T1** (classificador só de ano, variante não datada: acurácia balanceada): ___56% - 67%___
- **T2** (a variante não datada vai mais alto no próprio teste? quanto?): ___70% - 80%___
- **T3** (o modelo não datado cai quando testado na variante datada? quanto?): ___4 pp__
- **T5** (no Fake.br, qual variante vai melhor?): ___datada'___

---

## Adendos

### Adendo 1 — 13/10/2026, resultado da rodada

**Execução.** Pré-registro (1148435), amostra (a7759d6) e script de análise
(a88c773) commitados antes; coleta de N concluída (2.182 matérias novas, 9
descartes definitivos); rodada única da análise (25 s). Saída:
`relatorios/pareamento_datado/resultado.json`.

**Montagem de N.** 2.500 de 2.500: 2.178 da coleta nova e 322 reaproveitadas do
corpus. Trocas pela próxima URL do ranking: 49 não coletadas (descartes e erros
de rede), 4 que caíram nos filtros de entrada, 1 do corpus inelegível. Anos das
verdadeiras de N: de 315 a 400 por ano, quase uniforme; nas falsas (= D), 53%
são de 2019–2020. Testes cruzados: 12 verdadeiras do teste de D estavam no
treino de N, e 10 do teste de N no treino de D; saíram das medidas cruzadas.

**Resultado** (acurácia balanceada, IC95%):

| medida | D (datada) | N (não datada) |
|---|---:|---:|
| só o ano, no próprio teste (T1) | 47,0% [42,6; 51,5] | **64,0%** [60,2; 67,8] |
| TF-IDF, no próprio teste | 95,20% | 95,20% |
| TF-IDF, no teste de D (488 itens) | 95,07% | 94,65% |
| TF-IDF, no teste de N (490 itens) | 95,30% | — |
| TF-IDF, no Fake.br | 70,07% | 69,93% |

| diferença | valor | IC95% |
|---|---:|---|
| T2: N → N − D → D | 0,0 pp | [−2,8; 2,6] |
| **T3: N → teste de D − D → D** (mesmos 488 itens; McNemar 9 × 11, p = 0,82) | **−0,42 pp** | [−2,3; 1,4] |
| T4: D → teste de N − D → D | +0,1 pp | [−2,5; 2,7] |
| T5: N → Fake.br − D → Fake.br (McNemar 329 × 339, p = 0,73) | −0,14 pp | [−0,8; 0,6] |

Marcas de época entre as 30 palavras de maior peso: 2 para FALSO e 1 para
VERDADEIRO, **nas duas variantes**. As de maior peso para VERDADEIRO são de
registro de agência ("hoje", "disse", "pela", "nesta"), e as de FALSO são de
corrente ("lula", "globo", "pra", "vídeo", "você", "agora", "boato").

O empate exato em 95,20% (mesmas revocações nas duas classes) é coincidência
de contagem: os modelos são diferentes (C igual, pesos diferentes, e resultados
diferentes nos mesmos itens de T3).

**Leitura fixada aplicável (§3): T1 (N) = 64,0% ≥ 60% e T3 = −0,42 pp > −5 pp
→ "atalho disponível, pouco usado".** Sem datação, a época passa a separar as
classes (64% só com o ano, contra 47% com datação), mas o modelo de texto não
se apoia nela: o contraste de gênero (corrente × agência) já separa 95% dos
textos, e a época não acrescenta nada. Nem dentro da base nem no Fake.br a
variante não datada se sai diferente da datada.

**Confronto com as previsões:**

| | resultado | §2 | pesquisador |
|---|---|---|---|
| T1 só de ano, D | 47,0% | 48–55 ✗ (1 pp abaixo; o IC inclui 50%) | — |
| T1 só de ano, N | 64,0% | 60–72 ✓ | 56–67 ✓ |
| T2 N → N − D → D | 0,0 pp | +2 a +8 ✗ | "70–80%" lido como a acurácia de N no próprio teste: 95,2% ✗ |
| T3 N → teste de D − D → D | −0,42 pp | ≤ −5 ✗ | "cai 4 pp" ✗ |
| T4 D → teste de N − D → D | +0,1 pp | −3 a +3 ✓ | — |
| T5 N → Fake.br − D → Fake.br | −0,14 pp | ≤ −2 ✗ | "datada melhor": a direção coincide, mas a diferença não é significativa; lido como empate ✗ |

**Leitura para o TCC.** Nesta janela, o pareamento datado é uma boa prática,
porque elimina um atalho que existe (64% só com o ano), mas não muda o que o
modelo de texto aprende nem o quanto ele transfere. Isso é coerente com a RQ4:
o atalho dominante da base é o gênero, e ele encobre o temporal. Num corpus em
que o gênero fosse controlado (verdadeiras e falsas no mesmo gênero), o atalho
temporal poderia passar a pesar, e por isso a datação continua recomendada.

**Desvios e limites declarados:** (1) as verdadeiras de N passaram pelos mesmos
filtros de entrada do pareado, o que o pré-registro não dizia explicitamente;
decidido e commitado no script antes da rodada (4 trocas); (2) a previsão T2 do
pesquisador foi escrita como acurácia, não como diferença, e foi lida como a
acurácia de N no próprio teste; (3) só 2014–2020, só o braço bruto, só TF-IDF +
regressão logística, divisão aleatória (o que explica os 95%, acima dos 91% da
divisão temporal); (4) com 500 itens de teste, o IC é de cerca de ±2,5 pp nas
diferenças pareadas: efeitos menores que 2 pp não seriam detectados.
