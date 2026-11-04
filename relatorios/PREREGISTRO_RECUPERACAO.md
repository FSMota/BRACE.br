# Pré-registro — protótipo de recuperação de checagens (RQ3, uso B)

> Escrito em 13/10/2026, **antes** de qualquer consulta ser feita. Commitar este
> arquivo antes do sorteio das consultas. Nada aqui é editado depois;
> correções vão num adendo datado, no fim.

**Pergunta:** quando chega um boato novo, a base encontra a checagem já feita
da **mesma alegação**, com a justificativa do checador? E o resumo da alegação
feito pelo LLM (`ideia_central`) ajuda a encontrar?

**Por que importa:** é o uso em que a base serve diretamente ao trabalho de um
checador, sem depender da acurácia de um classificador. Também é onde a
justificativa entra como recurso: a resposta do sistema é a checagem antiga e
o motivo pelo qual a alegação é falsa.

**Por que precisa de julgamento humano:** nenhum boato do `teste` foi agrupado
como quase-duplicata de uma falsa do `treino` (conferido em 13/10/2026). Boatos
que voltam a circular aparecem reescritos, e só a leitura diz se a alegação é a
mesma.

---

## 1. Desenho fixado

| item | valor |
|---|---|
| consultas | **50** falsas do `teste` do pareado (de 06/08/2024 em diante), sorteadas com semente 0 |
| acervo de busca | as falsas do `treino` e da `validacao` do pareado (9.393, todas anteriores a 06/08/2024), com `texto_entrada`, `ideia_central` e `justificativa` |
| sistemas | **S1**: TF-IDF de palavras (1–2 gramas, `min_df=2`, sublinear), cosseno, sobre `texto_entrada`. **S2**: o mesmo sobre `ideia_central`. **S3**: vetores `bge-m3` (Ollama) de `texto_entrada`, cosseno. **S4**: `bge-m3` de `ideia_central`. **S0** (linha de base): 3 falsas do acervo sorteadas do mesmo tema da consulta |
| profundidade | as **3** primeiras de cada sistema |
| conjunto julgado | a união das 3 primeiras de S0 a S4, por consulta, sem repetição, em ordem embaralhada. **Às cegas:** a planilha mostra a consulta e o candidato, sem o sistema e sem a posição |
| julgamento 1 | **mesma alegação** (o candidato afirma o mesmo fato falso, ainda que com outras palavras, outra data ou outro tom) · **relacionada** (mesmo assunto, alegação diferente) · **não relacionada** |
| julgamento 2 | só para "mesma alegação": a justificativa do candidato **desmente a consulta?** (sim / em parte / não) |
| anotadores | o pesquisador, às cegas (resultado principal); um LLM como segundo anotador declarado, sem ver o primeiro; kappa de Cohen; adjudicação das divergências pelo pesquisador, como na C3 |

**Medidas:**
- **sucesso@3** de cada sistema: fração das 50 consultas com pelo menos uma
  "mesma alegação" entre as 3 primeiras. IC95% de Wilson.
- **precisão@1**: fração das consultas em que a 1ª é "mesma alegação".
- **cobertura estimada**: fração das consultas em que **algum** sistema achou
  uma "mesma alegação" (limite inferior de quantos boatos novos já tinham sido
  checados).
- **utilidade da justificativa**: entre as "mesma alegação", fração com "desmente
  a consulta: sim".
- **comparações fixadas:** S2 × S1 e S4 × S3 (o `ideia_central` ajuda?), S3 × S1
  (vetores × TF-IDF), pelo McNemar exato pareado por consulta, com Holm sobre as
  três.

**Limites declarados agora:**
- 50 consultas dão IC largos (±12 pp perto de 30%); serve de protótipo, não de
  medida definitiva.
- a cobertura é um limite inferior: uma checagem antiga que nenhum sistema pôs
  entre as 3 primeiras não é vista.
- o acervo só tem falsas: o sistema não diz se uma mensagem é verdadeira.
- S3 e S4 dependem dos vetores `bge-m3`, extraídos no PC com GPU na matriz de
  modelos; a planilha só é montada depois que eles existirem.

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| B1 | cobertura estimada (algum sistema acha a mesma alegação) | **30% a 55%** |
| B2 | sucesso@3 do melhor sistema | **25% a 45%** |
| B3 | S0 (sorteio do mesmo tema) | **≤ 4%** |
| B4 | S4 − S3 e S2 − S1 (o `ideia_central` ajuda) | **0 a +10 pp** |
| B5 | S3 − S1 (vetores × TF-IDF) | **+0 a +15 pp** |
| B6 | utilidade da justificativa ("sim") | **≥ 70%** |

**Por que:** o Boatos.org desmente com frequência boatos que voltam (eleições,
saúde, golpes); o TF-IDF falha quando a alegação volta reescrita, o que os
vetores e o resumo da alegação deveriam compensar.

## 3. Leituras fixadas antes do resultado

| sucesso@3 do melhor sistema | leitura |
|---|---|
| **≥ 30%** | a base funciona como **memória de checagens**: ao menos um em cada três boatos novos já tinha sido checado e é encontrado |
| **10% a 30%** | útil em parte: encontra uma fração dos boatos que voltam |
| **< 10%** | ou os boatos pouco se repetem, ou a busca falha; a cobertura estimada separa as duas coisas |

Para B4: se S2 > S1 **e** S4 > S3 com McNemar significativo, o `ideia_central`
agrega valor à busca. Se só um dos dois, o efeito depende da representação. Se
nenhum, o refinamento não agrega valor nem aqui.

## 4. Previsões do pesquisador

> Preencher antes do sorteio, em uma linha cada.

Registradas em 13/10/2026, antes do sorteio (data preenchida pelo assistente no commit).

- **B1** (em quantos % dos boatos novos já existe uma checagem antiga da mesma alegação?): ___75%___
- **B2** (sucesso@3 do melhor sistema): ___30% - 45%___
- **B4** (o resumo do LLM ajuda a buscar?): ___sim <= 5pp___
- **Melhor sistema** (S1, S2, S3 ou S4?): ___S3___

---

## Adendos

### Adendo 1 — 29/10/2026, pré-preenchimento por regra e papel do segundo anotador (antes de qualquer resultado)

**Desvio declarado.** Para reduzir a carga de anotação (665 pares), os pares com
**0 ou 1 palavra significativa em comum** entre a consulta e o candidato
(palavras minúsculas, sem uma lista curta de palavras vazias, sem redução à
raiz) são pré-preenchidos como `nao_relacionada`, com a marca "auto: ≤1 palavra
em comum" na observação. Regra proposta pelo pesquisador; script
`scripts/justificativa_lixo.py`. Só linhas ainda vazias são preenchidas (16
já estavam anotadas à mão quando a regra foi aplicada). Cópia da planilha
antes da regra: `relatorios/recuperacao/planilha_recuperacao_antes_preenchimento.xlsx`.

**Risco e controle.** A regra é lexical e pode marcar como não relacionada a
mesma alegação escrita com outras palavras (sinônimos, paráfrase, flexões). Esse
erro pesaria mais sobre os sistemas semânticos (S3 e S4) do que sobre os
lexicais (S1 e S2), justamente a comparação S3 × S1. Por isso:

- o segundo anotador (o assistente, já previsto no §1) anota **as 665 linhas**,
  sem ver as respostas do pesquisador nem a chave dos sistemas; ele conhece a
  regra, mas trabalha sobre a planilha **anterior** ao pré-preenchimento;
- toda linha pré-preenchida que o segundo anotador marcar como
  `mesma_alegacao` ou `relacionada` volta ao pesquisador para decisão; a
  decisão dele substitui a da regra;
- a apuração relata quantas linhas cada sistema teve pré-preenchidas e
  quantas a regra errou (as revertidas), por sistema.

### Adendo 2 — 01/11/2026, segundo anotador e adjudicação (antes da apuração)

**Anotador 1 (pesquisador).** 665 de 665 pares: 304 pela regra do adendo 1 (o
pesquisador mudou 1 delas, n = 180, para `relacionada`) e 361 à mão. Grafia
"em parte" normalizada para `em_parte` (2 células). Um salvamento interrompido
danificou o índice do arquivo; ele foi reconstruído a partir das partes
íntegras (CRC conferido), sem mudar o conteúdo (commit 4ded009).

**Anotador 2 (assistente).** Anotou os 665 pares sobre a cópia anterior ao
pré-preenchimento, lendo só a consulta, o candidato e a justificativa, sem ver
o anotador 1 nem a chave dos sistemas (`relatorios/recuperacao/anotacao_assistente.json`,
commit 2c5aa43). Distribuição: 3 `mesma_alegacao`, 72 `relacionada`, 590
`nao_relacionada` (anotador 1: 3, 27 e 635).

**Concordância.** 75 divergências em 665 (88,7% de concordância; kappa de
0,24 nas três categorias). As duas categorias de "mesma alegação" não
coincidiram em nenhum par (anotador 1: 469, 601 e 610; anotador 2: 254, 258 e
529). O anotador 2 usou `relacionada` com mais frequência, e quase todas as
divergências são entre `relacionada` e `nao_relacionada`.

**Controle da regra (adendo 1).** Das 304 linhas pré-preenchidas, o anotador 2
não marcou nenhuma como `mesma_alegacao`; marcou 10 como `relacionada` (57, 59,
130, 362, 383, 398, 428, 481, 488 e 598). A regra **não retirou nenhum acerto**
de nenhum sistema; os 10 erros dela são todos entre `relacionada` e
`nao_relacionada`.

**Adjudicação (pesquisador).** Coluna `consenso_j1`/`consenso_j2`, usada pela
apuração:

- as 6 divergências que envolvem `mesma_alegacao`: 469 → `relacionada`;
  601 → `mesma_alegacao`/`sim`; 610 → `mesma_alegacao`/`em_parte`;
  254 → `mesma_alegacao`/`em_parte`; 258 → `mesma_alegacao`/`em_parte`;
  529 → `relacionada`;
- as 10 linhas da regra marcadas `relacionada` pelo anotador 2: aceitas como
  `relacionada`.

**Desvio declarado.** As outras 59 divergências (todas entre `relacionada` e
`nao_relacionada`) **não foram adjudicadas**: vale a resposta do anotador 1.
Essa distinção não entra no sucesso@3, na precisão@1, na cobertura nem na
utilidade da justificativa, que só contam `mesma_alegacao`.

Consenso final: 4 pares `mesma_alegacao` (254, 258, 601 e 610).

### Adendo 3 — 01/11/2026, resultado

**Execução.** `scripts/recuperacao.py --apurar` sobre a coluna de consenso
(adendo 2). Saída: `relatorios/recuperacao/resultado_recuperacao.json`.

| sistema | sucesso@3 | precisão@1 |
|---|---:|---:|
| S0 (sorteio do tema) | 0% [0; 7,1] | 0% |
| S1 (TF-IDF, texto) | 2% [0,4; 10,5] | 2% |
| S2 (TF-IDF, ideia central) | 2% [0,4; 10,5] | 2% |
| S3 (bge-m3, texto) | **4% [1,1; 13,5]** | 4% |
| S4 (bge-m3, ideia central) | **4% [1,1; 13,5]** | 4% |

**Cobertura estimada:** 2 das 50 consultas (4%, IC95% de 1,1 a 13,5): a conta
de luz falsa (encontrada como as versões da Copel e da Sabesp) e as
"nanopartículas" da vacina no Japão. **Utilidade da justificativa:** dos 4
pares de mesma alegação, 1 `sim` e 3 `em_parte`.

**Comparações:** S2 − S1 = 0, S4 − S3 = 0, S3 − S1 = +2 pp (1 consulta
discordante); nenhuma significativa (p de Holm = 1). Com 2 acertos em 50, o
teste não tem poder para separar os sistemas.

**Confronto:**

| | resultado | §2 | pesquisador |
|---|---|---|---|
| B1 cobertura | 4% | 30% a 55% ✗ | 75% ✗ |
| B2 melhor sistema | 4% | 25% a 45% ✗ | 30% a 45% ✗ |
| B3 S0 | 0% | ≤ 4% ✓ | — |
| B4 ideia central ajuda | 0 pp | 0 a +10 pp ✓ (no limite) | sim, ≤ 5 pp ✓ |
| B5 vetores × TF-IDF | +2 pp | 0 a +15 pp ✓ (não significativo) | — |
| B6 utilidade | 25% sim | ≥ 70% ✗ | — |
| melhor sistema | S3 = S4 | — | S3 (empatado) |

**Leitura fixada (§3): sucesso@3 do melhor sistema < 10% → "ou os boatos pouco
se repetem, ou a busca falha; a cobertura estimada separa as duas coisas".**
Aqui ela **não separa**: a cobertura (4%) é igual ao melhor sistema, e é um
limite inferior que só enxerga as 3 primeiras posições de cada sistema. O
resultado admite três explicações, que este protótipo não distingue: (1) os
boatos do período de teste (de agosto de 2024 em diante) são em grande parte
novos, e não versões de boatos antigos; (2) a busca falha em pôr a checagem
antiga entre as 3 primeiras; (3) o critério de "mesma alegação" é difícil de
aplicar: os dois anotadores não coincidiram em nenhum par de mesma alegação
antes da adjudicação (kappa ≈ 0 nessa decisão), o que torna a medida frágil.
Entre os pares julgados `relacionada` (27 do anotador 1, 72 do anotador 2), a
busca acha o **mesmo assunto** com frequência, mas raramente a mesma alegação.

**Consequência para o trabalho.** O uso B (recuperação de checagens) fica como
protótipo com resultado negativo no desenho atual: a BRACE.br, nesta avaliação,
não funciona como memória de checagens para boatos novos. A justificativa do
checador, quando o par é a mesma alegação, serve só em parte (3 de 4), porque
trata de outra versão (outra empresa, outro vídeo).
