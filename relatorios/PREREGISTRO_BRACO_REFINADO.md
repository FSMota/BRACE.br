# Pré-registro — braço refinado (RQ1)

> Escrito em 06/10/2026, **antes** de a fila de refinamento rodar. Nenhuma saída
> do refinamento cego existe ainda além do piloto (50 textos por grupo, só do
> treino do corpus e do Fake.br — ver §1.2).
> Commitar este arquivo antes de rodar `scripts/refinar_cego.py`: o histórico do
> git é a prova da data. Nada aqui pode ser editado depois da rodada. Correções
> vão num adendo datado, no fim.

Por que existe: o mesmo autor escreveu o prompt, o pipeline e vai interpretar o
resultado. O braço refinado é o coração da RQ1 e tem mais graus de liberdade que
o bruto (o texto de entrada passa a ser gerado por um modelo). Fixar desenho,
previsões, checagens de validade e leituras antes impede que o resultado escolha
a interpretação.

Pré-registro anterior, do qual este herda o modelo e a avaliação cruzada:
`relatorios/PREREGISTRO_AVALIACAO_CRUZADA.md` (adendo 5 = números de referência
do braço bruto).

---

## 1. Desenho fixado

### 1.1 O refinamento

| item | valor fixado |
|---|---|
| script | `scripts/refinar_cego.py` (execução completa, sem `--limite`) |
| prompt | **`cego-v2`, sha `61d3e5623fccdf41`** (sha256 dos 16 primeiros dígitos do texto do prompt) |
| modelo | qwen3.5:9b, Ollama, `think=False`, mesmo cliente da extração (`web_scraping_limpo.chamar_llm`) |
| saída estruturada | JSON Schema com um campo, `ideia_central` |
| tentativas | até 3 por texto; frase com < 3 palavras depois da limpeza conta como falha |
| limpeza da saída | `limpar_saida()` (corta quebra de linha, markdown, autoanálise; tira aspas). Não muda o prompt nem o hash. Cada corte fica em `saida_ajustes` |
| saída | `datasets/refinamento_cego.jsonl`; todas as linhas com `prompt_sha = 61d3e5623fccdf41` (o script recusa misturar) |

**Fila:** `datasets/refinamento_fila.jsonl` — 31.150 textos:

| grupo | textos | entrada |
|---|---:|---|
| corpus_falso | 11.975 | `texto_entrada` do subconjunto pareado (base v3) |
| corpus_verdadeiro | 11.975 | idem |
| fakebr_falso | 3.600 | entrada A da avaliação cruzada (`entrada_texto`) |
| fakebr_verdadeiro | 3.600 | idem |

sha256 do `.jsonl`: `3533440ab268976dd9d9ef7f71c14fd7bb1590f5c3f7397f230678f43e35220a`
sha256 do `.jsonl.gz` (5,7 MB, o que vai para a máquina com GPU):
`6b3341513735b35be351f78da79019d607194aa383beddf64510342bdd536994`
Corpus de origem: `corpus_fundido.jsonl`, sha `901d8373a1f5452c`.

Ordem da fila embaralhada com semente 0: uma interrupção deixa as quatro classes
misturadas na saída parcial.

**Condição de validade (já registrada na RQ1):** o mesmo prompt, cego ao
veredito, sobre o mesmo tipo de entrada, nas duas classes e no Fake.br. O prompt
não menciona veracidade, checagem nem origem do texto.

### 1.2 O que já foi visto antes deste pré-registro (declarado)

- Piloto do `cego-v1` (25 por grupo, 05/10/2026) — motivou o `cego-v2`.
- Piloto do `cego-v2` (50 por grupo, só `treino` e Fake.br):
  `relatorios/piloto_refinamento_cego-v2.json`. Diferenças falso − verdadeiro:
  corpus palavras −4,0, atribuição +4,0 pp, distanciamento 0, veredito +2,0 pp,
  metatexto +2,0 pp; Fake.br palavras −2,3, atribuição −2,0 pp, distanciamento
  −4,0 pp. Tempo médio ~1,5 s/texto (~13 h a fila inteira).
- A leitura das frases do piloto mostrou inversões de sentido ("Congresso vota" →
  "aprovou"; notícia verdadeira virando falsa). É a origem da checagem C3.
- Nenhuma frase do `teste` do corpus foi refinada ou lida.

### 1.3 Os três braços

Mesmo modelo do braço bruto: TF-IDF de palavras (1–2 gramas, `min_df=3`,
`sublinear_tf=True`, demais parâmetros padrão do scikit-learn, como no
`modelo_bruto.py`) + regressão logística (`max_iter=2000`). **C escolhido na
validação, separadamente para cada braço**, na grade `[0,1; 1; 10]` do
`modelo_bruto.py`. Treino no `treino` do pareado; C pela `validacao`; nota no
`teste`.

| braço | entrada | representação |
|---|---|---|
| **bruto** | `texto_entrada` | um TF-IDF |
| **refinado** | `ideia_central` | um TF-IDF, mesmos parâmetros |
| **combinado** | `texto_entrada` + `ideia_central` | **dois TF-IDF separados (um por campo, cada um ajustado só no treino), concatenados lado a lado** (`hstack`), uma regressão logística |

Concatenação de matrizes, não de strings: a mesma palavra no texto e na frase
refinada vira dois atributos diferentes, e dá para ler qual parte o modelo usou.

**Mesmos ids nos três braços.** O conjunto de avaliação é a interseção:
1. ids do pareado (e do Fake.br) com saída válida do refinamento — os que
   falharem nas 3 tentativas saem **dos três braços** e o número é relatado por
   grupo;
2. menos os ids anotados como "não é boato" na anotação manual
   (`relatorios/anotacao/`, `anotacao_manual.py --apurar`) — saem **dos três
   braços** pelo id. Se a anotação não estiver apurada quando o comparativo
   rodar, roda sem essa exclusão e a exclusão vira adendo.

O braço bruto é **retreinado** nessa interseção (não se reaproveitam os 91,19%
do adendo 5). Se a interseção perder menos de 0,5% dos ids, a diferença em
relação ao adendo 5 deve ser < 0,5 pp; relatar as duas.

**Rodada B (secundária):** a mesma, sem os 121 registros com
`entrada_com_veredito`, como no braço bruto.

### 1.4 Avaliação cruzada

O Fake.br entra **só como teste**: os 3.600 pares refinados (`fakebr_falso`,
`fakebr_verdadeiro`), cada braço com o vetorizador e o C escolhidos no corpus.
**Nenhum ajuste usando o Fake.br** — nem C, nem limiar, nem vocabulário. No
braço combinado, o par (entrada A, frase refinada) de cada texto do Fake.br.

### 1.5 Métricas e testes

- **Principal:** acurácia balanceada. **Secundárias:** revocação por classe;
  acurácia por categoria do Fake.br.
- **Queda** = acurácia interna − acurácia no Fake.br, por braço.
- **Intervalos:** bootstrap percentil, 2.000 reamostragens, semente 0;
  reamostragem por item dentro de cada conjunto de teste (interno e Fake.br,
  independentes). Para a diferença entre quedas, as duas reamostragens são
  feitas juntas e a diferença é calculada em cada réplica.
- **Comparação entre braços no mesmo conjunto:** McNemar exato (mesmos itens).
- **Família confirmatória, com correção de Holm (alfa 0,05):**
  H1 (queda refinado < queda bruto), bruto × refinado no Fake.br, bruto ×
  combinado no Fake.br. Todo o resto é descritivo ou exploratório.
- **Tamanho de efeito com IC sempre**, não só p-valor.

### 1.6 Ordem de execução (fixada)

1. Refinamento na GPU. Na volta: conferir sha256 da saída, contagem por grupo,
   `prompt_sha` único, falhas e `saida_ajustes` por grupo — **só isso** pode
   rodar antes do passo 2.
2. Escrever e **commitar** o script de junção (saída → corpus pelo id) e o do
   comparativo, com as checagens da §3 implementadas, **antes de qualquer
   análise ler as frases refinadas**.
3. Rodar: checagens C1, C2, C4 (automáticas) e os três braços.
4. Auditoria de fidelidade C3 (manual).
5. Adendo com o resultado e o confronto.

Fora deste pré-registro: escada de modelos (BERTimbau, LLM sem treino) nos dois
braços — terá desenho próprio.

---

## 2. Previsões

Referência do braço bruto (adendo 5): interno **91,19%**, Fake.br **73,51%**
[72,54; 74,44], queda **17,7 pp**. Só forma (17 atributos): 87,21% interno,
65,18% Fake.br. Léxico além da forma no Fake.br: ~8 pp.

| # | previsão | faixa |
|---|---|---|
| R1 | acurácia interna, refinado | **75% – 85%** |
| R2 | acurácia no Fake.br, refinado | **55% – 63%** |
| R3 | queda interno → Fake.br, refinado | **15 – 25 pp** |
| R4 | combinado, interno | **91% – 93%** (≥ bruto − 0,5 pp) |
| R5 | combinado, Fake.br | **72% – 76%** (bruto ± 2 pp) |
| R6 | revocação FALSO × VERDADEIRO no Fake.br, refinado | diferença **≤ 10 pp** |
| R7 | termos de maior peso para FALSO no refinado | assunto de corrente (vacina, golpe, WhatsApp, urna, morte) |

**Por que cada uma.**
- R1: o refinamento apaga a forma (caixa alta, "!", "você"), mas os assuntos
  dentro do mesmo tema seguem muito diferentes — corrente fala de golpe de
  WhatsApp e vacina que mata; a agência, de programa de governo e índice
  econômico. O pareamento é por tema × ano, não por assunto. Abaixo de 87,21%
  (só forma) porque a forma era a maior parte do sinal.
- R2: se o léxico além da forma rendia ~8 pp no Fake.br, o conteúdo sozinho,
  reescrito em frase neutra, deve ficar por aí ou abaixo (o refinamento também
  apaga parte do léxico).
- R3: **esta previsão diverge da hipótese H1 de 29/09/2026** (queda do
  refinado menor que a do bruto). Com R1 e R2 nos centros das faixas, a queda é
  ~21 pp — maior. O motivo: o conteúdo que separa as classes no corpus (assuntos
  de corrente × pauta de agência) é específico deste corpus e pode transferir
  menos que a forma. Fica registrado o conflito; o resultado decide.
- R4/R5: a frase refinada é quase toda redundante com o texto bruto; o modelo
  combinado deve se apoiar no bruto.
- R6: no braço bruto as revocações ficaram parecidas (72,75 × 74,28%); sem
  forma, não há razão para o limiar se deslocar para um lado.
- R7: é o que sobra quando a forma sai.

### H1 — hipótese da RQ1 (registrada em 29/09/2026, repetida aqui)

> No braço refinado, a queda da avaliação cruzada será **menor** que a do braço
> bruto (17,7 pp).

Teste: diferença das quedas (bruto − refinado) com IC95% por bootstrap (§1.5).
**Confirmada** se o IC excluir zero do lado positivo; **refutada** se excluir
zero do lado negativo; **inconclusiva** se contiver zero.

**Ressalva fixada agora:** uma queda menor pode vir de duas formas muito
diferentes, e a leitura depende de qual:
- (a) **o refinado vai melhor no Fake.br que o bruto** → o refinamento melhora a
  generalização;
- (b) **o refinado vai pior no Fake.br, mas cai ainda mais no interno** → o
  refinamento remove atalho interno (a forma), não melhora a generalização.
  Nesse caso H1 é confirmada pela letra, mas **não** pode ser descrita como
  "o refinamento generaliza melhor".

---

## 3. Checagens de validade obrigatórias

Rodam **antes** de interpretar os braços. Cada uma tem limiar fixado e
consequência fixada. O braço refinado só é lido como "conteúdo sem forma" se
passar em todas; falha não esconde o resultado — vira atalho declarado
(resultado de RQ1/RQ4).

### C1 — Baseline de TAMANHO no braço refinado

Regressão logística só com o número de palavras (e o de caracteres) da
`ideia_central`, mesmo treino/validação/teste. No piloto, as falsas saíram ~4
palavras mais curtas no corpus.

- Previsão: **58% – 65%** interno.
- Limiar: **> 60%** interno → tamanho é atalho declarado do braço refinado.
  Consequência: relatar também a acurácia do refinado acima desse baseline e o
  modelo refinado + tamanho, para separar quanto do refinado é tamanho.
- Mesmo cálculo no Fake.br (relatado, sem limiar).

### C2 — Marcas por classe na saída

As quatro marcas do `refinar_cego.py` (`MARCAS`: atribuição, distanciamento,
veredito, metatexto), mais `saida_ajustes`, % de frases > 25 palavras e > 1
frase, por grupo, na fila inteira.

- Previsão: diferença falso − verdadeiro **< 5 pp** em cada marca, nas duas bases.
- Limiar: **≥ 10 pp** em qualquer marca (o mesmo do piloto) → atalho declarado.
  Consequência: rodar o braço refinado também sem as frases que têm a marca, e
  relatar as duas.
- Limiar à parte para **veredito** no corpus: se > 2% das frases falsas tiverem
  palavra de veredito, é vazamento de rótulo criado pelo refinamento (o modelo
  reconheceu o boato) — relatar o braço com e sem elas.

### C3 — Auditoria de FIDELIDADE (manual)

Amostra: **100 frases por classe do `teste` do corpus** (200 no total), sorteio
com semente fixada **no script, antes de ler qualquer frase** (sorteio por id
sobre a fila, que já existe). Ordem embaralhada, sem coluna de classe — com a
ressalva de que o texto bruto denuncia a classe pela forma, então a auditoria
não é cega de fato.

Para cada par (`texto_entrada`, `ideia_central`), uma categoria:

| categoria | definição |
|---|---|
| **fiel** | a frase diz o que o texto diz |
| **mudou o sentido** | alterou modalidade ou fato: "vota" → "aprovou", negação perdida ou acrescentada, tempo trocado, sujeito trocado |
| **inventou** | acrescentou fato, número, nome ou data que não está no texto |
| **perdeu o central** | fiel no que diz, mas descreve um detalhe e não a alegação principal |

- Previsão: fiel ≥ 80% em cada classe; mudou + inventou entre 5% e 15%.
- Limiar: diferença entre classes em (mudou + inventou) **≥ 10 pp** → o
  refinamento distorce uma classe mais que a outra; o braço refinado **não pode**
  ser lido como "o mesmo conteúdo sem a forma". Declarar e discutir a direção.
- Relatado com IC95% (Wilson) por classe. Um anotador (o pesquisador); se houver
  segundo anotador, kappa de Cohen.

### C4 — Controles de pipeline no refinado (e no combinado)

Os mesmos do `controles_pipeline.py`:
- **rótulos embaralhados** (3 sementes): previsão **45% – 55%**; fora disso, há
  vazamento no pipeline e nenhum número do braço é interpretado até achar a
  causa;
- **sobreposição entre partes**: frases refinadas idênticas (após normalização)
  entre treino e teste. Previsão: **poucas (< 0,5%)** — boatos reciclados podem
  virar a mesma frase. Relatar quantas e a acurácia sem elas.

---

## 4. Leituras fixadas antes do resultado

### Refinado, interno

| resultado | leitura |
|---|---|
| **≥ 88%** | o conteúdo sozinho separa quase como o texto bruto: forma e conteúdo são sinais **redundantes** no corpus, não "a forma explica tudo" |
| **75% – 88%** | o conteúdo separa bem, mas abaixo da forma: o corpus tem diferença de assunto entre as classes dentro do mesmo tema |
| **60% – 75%** | a separação interna era majoritariamente forma; o conteúdo contribui pouco |
| **≤ 60%** | sem a forma, quase nada separa — e é preciso descartar que o refinamento apagou o conteúdo (C3) antes de concluir |

### Refinado, Fake.br

| resultado | leitura |
|---|---|
| **≥ 70%** | o conteúdo transfere tanto quanto o texto bruto: a decomposição do adendo 3 (≈ 2/3 forma) subestimou o conteúdo, ou forma e conteúdo transferem pelo mesmo canal |
| **60% – 70%** | o conteúdo transfere de forma comparável à forma (65,18%) |
| **55% – 60%** | o conteúdo transfere pouco, compatível com os ~8 pp de léxico além da forma |
| **≤ 55%** | o conteúdo aprendido é específico do corpus; o que o braço bruto transferia era quase só forma |

### Combinado

| resultado | leitura |
|---|---|
| Fake.br **≥ bruto + 2 pp** (IC excluindo zero) | a frase refinada acrescenta sinal que transfere — o refinamento ajuda como **complemento** |
| dentro de ± 2 pp | a frase refinada é redundante com o texto bruto |
| **≤ bruto − 2 pp** | a frase refinada atrapalha fora da base (o modelo se apoia em assunto do corpus) |

**Limite de interpretação, declarado agora** (o mesmo do pré-registro anterior,
reforçado): o Fake.br usa outra definição de falsa (matéria de site de
desinformação, não alegação checada). No braço refinado isso pesa mais: uma
alegação de blog hiperpartidário e uma de corrente de WhatsApp podem virar
frases neutras parecidas, ou não — e nenhum resultado sozinho separa "o conteúdo
não transfere" de "os conteúdos falsos das duas bases são de outro tipo".

---

## 5. Previsões do pesquisador

> Preencher **antes de rodar**, em uma linha cada. Divergência com as previsões
> da §2 é informação: no pré-registro anterior, P3 e P4 do pesquisador
> acertaram onde as da seção 2 erraram.

Registradas em 06/10/2026, antes da rodada.

- **R1** (refinado, interno): _79%___90%__
- **R2** (refinado, Fake.br): _60%___70%__
- **R3** (queda do refinado; maior ou menor que 17,7 pp?): _10 -15pp_
- **R4** (combinado, interno): __91% - 93%____
- **R5** (combinado, Fake.br): __70% - 80%____
- **R6** (revocação FALSO × VERDADEIRO no Fake.br, refinado): __<=10pp____
- **C1** (baseline de tamanho no refinado): __50% - 60%____
- **C2** (alguma marca vai diferir ≥ 10 pp? qual?): __não vai diferir____
- **C3** (fidelidade: qual classe o refinamento distorce mais?): __falsa____
- **H1** (confirmada, refutada ou inconclusiva? por (a) ou (b)?): __confirmada____

---

## Adendos

### Adendo 1 — 09/10/2026, resultado do braço refinado e da checagem de fidelidade

**Refinamento.** Rodado na GPU em 07/10/2026 (~13 h). Saída
`datasets/refinamento_cego.jsonl`, sha256
`e3d1db4dd7cab842f6456df958ec03cc67f0a5d7ccc44ef8adb76402f516eaec`: 31.149 de
31.150 textos (1 falsa do corpus sem saída nas 3 tentativas), `prompt_sha` único
`61d3e5623fccdf41`, nenhum JSON inválido, 11 saídas cortadas por quebra de linha
(`saida_ajustes`). Conferência: `relatorios/conferencia_refinamento.json`.

**Comparativo.** `scripts/comparativo_bracos.py`, saída
`relatorios/resultado_comparativo_bracos.json`. Conjunto: 23.949 de 23.950 do
pareado (perda 0,00%) e os 7.200 do Fake.br; a anotação manual (RQ2) ainda não
estava apurada, então os "não é boato" **não** foram excluídos (§1.3 previa
esse adendo). O bruto reproduziu o adendo 5 (91,19% / 73,53%).

| braço | C | interno | Fake.br | queda | rev. F / V no Fake.br |
|---|---:|---:|---:|---:|---:|
| bruto | 10 | 91,19% [90,43; 91,91] | 73,53% [72,47; 74,50] | 17,66 pp | 73,14 / 73,92% |
| **refinado** | 10 | **81,73%** [80,69; 82,78] | **66,31%** [65,24; 67,36] | **15,43 pp** | 68,39 / 64,22% |
| combinado | 10 | 91,17% [90,40; 91,88] | 72,96% [71,91; 73,94] | 18,21 pp | 70,75 / 75,17% |

Rodada B (sem `entrada_com_veredito`): bruto 91,29 / 73,62%; refinado
82,03 / 66,32%; combinado 91,30 / 72,78%.

**Família confirmatória (Holm, alfa 0,05):**

| teste | resultado | p (Holm) |
|---|---|---|
| H1: queda bruto − queda refinado | **+2,23 pp** [0,66; 3,85] | 0,014 → **confirmada** |
| refinado − bruto no Fake.br (McNemar) | **−7,22 pp** [−8,40; −6,06] | 7,6·10⁻³² |
| combinado − bruto no Fake.br (McNemar) | −0,57 pp [−1,28; +0,12] | 0,123 |

**Via da H1, pela ressalva fixada na §2: (b).** O refinado é pior que o bruto no
Fake.br; a queda é menor só porque a nota interna caiu mais. Pela leitura fixada,
o resultado **não** pode ser descrito como "o refinamento generaliza melhor".

**Checagens de validade:**
- **C1, tamanho:** 57,25% interno (55,62% no Fake.br), abaixo do limiar de 60% → passou.
- **C2, marcas:** nenhuma diferença ≥ 10 pp; veredito em 1,47% das falsas
  (limiar 2%) → passou. Diferenças maiores, abaixo do limiar: metatexto
  +3,96 pp (4,43% × 0,47%) e frases com > 25 palavras −9,44 pp. Observado
  depois: "texto" é o 2º termo de maior peso para FALSO no refinado, o que
  indica que o metatexto, embora abaixo do limiar, é usado pelo modelo.
  Registrado como ressalva, sem análise adicional.
- **C3, fidelidade:** abaixo → passou.
- **C4, controles:** rótulos embaralhados 52,07 / 49,30 / 49,73% (refinado) e
  54,01 / 50,72 / 48,53% (combinado), dentro de 45–55%; 2 frases idênticas em
  mais de uma parte (4 registros, 0,017%) → passou.

**Leituras fixadas aplicáveis (§4):** refinado interno 81,73% → "o conteúdo
separa bem, mas abaixo da forma: o corpus tem diferença de assunto entre as
classes dentro do mesmo tema". Refinado no Fake.br 66,31% → "o conteúdo
transfere de forma comparável à forma (65,18%)". Combinado → "a frase refinada
é redundante com o texto bruto".

**Confronto com as previsões:**

| | resultado | §2 | pesquisador |
|---|---:|---|---|
| R1 refinado interno | 81,73% | 75–85% ✓ | 79–90% ✓ |
| R2 refinado Fake.br | 66,31% | 55–63% ✗ | 60–70% ✓ |
| R3 queda do refinado | 15,43 pp | 15–25 pp ✓ | 10–15 pp ✗ (por 0,43) |
| R4 combinado interno | 91,17% | 91–93% ✓ | 91–93% ✓ |
| R5 combinado Fake.br | 72,96% | 72–76% ✓ | 70–80% ✓ |
| R6 revocação F × V (Fake.br) | 4,17 pp | ≤ 10 ✓ | ≤ 10 ✓ |
| C1 tamanho | 57,25% | 58–65% ✗ | 50–60% ✓ |
| C2 marca ≥ 10 pp | nenhuma | — | nenhuma ✓ |
| C3 classe mais distorcida | falsa (às cegas); nenhuma (adjudicado) | — | falsa ✓ às cegas |
| H1 | confirmada, via (b) | — | confirmada ✓ |

**C3 — auditoria de fidelidade.** 200 pares do teste, 100 por classe (sorteio
de 06/10, semente 20261006). Resultado em `relatorios/fidelidade/resultado_fidelidade.json`.

| | falsas | verdadeiras | diferença |
|---|---:|---:|---:|
| **anotador 1, o pesquisador, às cegas (resultado principal)**: distorção | 4% [1,6; 9,8] | 2% [0,6; 7,0] | **+2 pp** |
| anotador 2 (Claude, LLM) | 13% | 14% | −1 pp |
| **adjudicado** (34 divergências decididas pelo pesquisador) | **13%** [7,8; 21,0] | **13%** [7,8; 21,0] | **0 pp** |

Concordância entre os anotadores antes da adjudicação: kappa 0,09 (quatro
categorias) e 0,14 (distorção sim × não). Das 34 divergências, 28 eram itens que
o anotador 1 marcou como fiéis e o anotador 2 como problemáticos; na
adjudicação, 21 deles passaram de "fiel" para distorção e 1 saiu da distorção
(de 6 itens com distorção na leitura às cegas para 26 no consenso). O critério passa nas
duas leituras: o refinamento não distorce uma classe mais que a outra. O nível
de distorção, cerca de 1 frase em 8 depois da adjudicação (sentido invertido,
fato trocado, expectativa transformada em fato), é um resultado em si e reforça
a conclusão de que o LLM não é confiável como "limpador" de texto.

**Desvios do desenho, declarados:**
1. **Commit dos scripts depois da rodada.** A §1.6 exigia commitar a junção e o
   comparativo antes de qualquer análise ler as frases. Eles foram escritos em
   06/10/2026 (22:31), antes de a saída existir (07/10, 21:35), e não foram
   alterados, mas o commit (3e5ce50) veio depois da rodada.
2. **Anotação descartada.** Uma primeira versão da planilha da C3 foi preenchida
   por uma IA, por falta de tempo. Ela foi descartada antes de qualquer uso, e
   a planilha foi regenerada em branco para a leitura humana. O resultado dela
   fica guardado como `resultado_fidelidade_IA_nao_humano.json` e não entra em
   nenhuma conclusão.
3. **Segundo anotador é um LLM.** A §3 previa kappa "se houver segundo
   anotador"; o segundo anotador foi o Claude (Anthropic), declarado como tal, que
   anotou antes de ver as respostas do pesquisador (`anotacao_claude.json`).
   Ressalva: nos itens 11, 19, 24, 26, 36, 46, 50 e 56, ele tinha visto antes as
   observações da anotação descartada.
4. **Correção no script da C3.** `auditoria_fidelidade.py` passou a ignorar
   linhas vazias salvas pelo editor de planilha e a relatar o consenso à parte.
   Nenhuma regra de decisão mudou.
5. **Exclusão dos "não é boato" pendente:** fica para um adendo, quando a
   anotação manual estiver apurada.
