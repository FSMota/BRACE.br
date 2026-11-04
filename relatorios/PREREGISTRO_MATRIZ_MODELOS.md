# Pré-registro — matriz de modelos: as conclusões valem para outros modelos?

> Escrito em 13/10/2026, **antes** de qualquer representação nova ser extraída
> ou de qualquer classificador novo ser treinado. Commitar este arquivo antes
> de rodar. Nada aqui é editado depois; correções vão num adendo datado, no fim.

**Origem:** sugestão da orientação (09/10/2026): reforçar a confiança nos
resultados com mais extratores de características (BERT, Ollama), mais
classificadores e mais um modelo clássico de PLN além do TF-IDF de palavras. O
pesquisador trocou o Random Forest, da proposta inicial, por Naive Bayes
(13/10/2026).

**Pergunta:** as conclusões do TCC, tiradas até aqui com um único modelo
(TF-IDF de palavras + regressão logística), continuam valendo quando a
representação e o classificador mudam?

**O que este pré-registro NÃO faz:** não procura o melhor modelo. As afirmações
abaixo são conferidas em todas as combinações, e o resultado é "vale em X de Y".
Nenhuma combinação é escolhida depois por ter ido bem.

---

## 1. Desenho fixado

### 1.1 Dados

Os do comparativo dos braços (`datasets/braco_refinado.jsonl`): `treino`
(16.707), `validacao` (2.080), `teste` (5.162) e Fake.br (7.200). Braços
**bruto** (`texto_entrada`) e **refinado** (`ideia_central`). Sem a exclusão
dos "não é boato" (a anotação manual ainda não foi apurada), como no
comparativo; declarado.

### 1.2 Representações

| sigla | representação | parâmetros fixos |
|---|---|---|
| **P** | TF-IDF de palavras (o modelo atual) | 1–2 gramas, `min_df=3`, sublinear |
| **Ch** | TF-IDF de caracteres | `char_wb`, 3–5 gramas, `min_df=3`, sublinear |
| **Bc** | BERTimbau congelado | `neuralmind/bert-base-portuguese-cased`, até 128 tokens, média dos vetores da última camada (sem os de preenchimento) |
| **Ol** | vetores de um LLM via Ollama | `bge-m3` (multilíngue), vetor do modelo sem normalização adicional |

Representações ajustadas só no `treino` (o vocabulário do TF-IDF) ou sem
ajuste (Bc, Ol). Os vetores de Bc e Ol são extraídos uma vez, no PC com GPU, e
salvos por id.

### 1.3 Classificadores

| sigla | classificador | grade escolhida na `validacao` |
|---|---|---|
| **LR** | regressão logística | C ∈ {0,1; 1; 10} |
| **SVM** | SVM linear (`LinearSVC`) | C ∈ {0,01; 0,1; 1} |
| **NB** | Naive Bayes: **multinomial** nas representações esparsas (P, Ch); **gaussiano** nas densas (Bc, Ol), porque elas têm valores negativos | multinomial: α ∈ {0,1; 0,5; 1}; gaussiano: sem grade |

**Mais um modelo fora da grade:** **Ba**, o BERTimbau **ajustado**
(fine-tuning), com a cabeça de classificação padrão. Até 128 tokens, taxa de
aprendizado 2e-5, lote 32, 3 épocas, semente 0. A época é escolhida pela
acurácia balanceada na `validacao`.

**Total:** 4 representações × 3 classificadores + Ba = **13 modelos por braço**,
26 no total, em **duas direções** (incluídas a pedido do pesquisador,
13/10/2026, antes de qualquer rodada):

| direção | treino | escolha de C / α / época | teste |
|---|---|---|---|
| **nossa base →** | `treino` da nossa base | `validacao` da nossa base | nosso `teste` e Fake.br |
| **Fake.br →** | Fake.br | média de 5 dobras **por par** do Fake.br (as mesmas da avaliação inversa, semente 0) | Fake.br fora da dobra e nosso `teste` |

Em nenhuma direção algo é ajustado na base de teste. **Exceção declarada:** o
Ba na direção Fake.br → é treinado nas dobras 1 a 4 do Fake.br, com a dobra 0
como validação para escolher a época, e testado só no nosso `teste`. A
validação cruzada completa custaria cinco ajustes finos por braço, então o Ba
fica sem a célula Fake.br → Fake.br.

### 1.4 Controle

Rótulos do `treino` embaralhados (semente 0), no braço bruto, nos 12 modelos
da grade. O Ba não entra: um ajuste fino a mais por controle custa horas, e
isso fica declarado.

### 1.5 Métricas

Acurácia balanceada, com IC95% por bootstrap (2.000 reamostragens, semente 0).
Bruto × refinado no mesmo modelo e nos mesmos itens: McNemar exato, com
correção de Holm sobre os 13 testes de cada afirmação.

## 2. Afirmações conferidas e previsões

Cada afirmação é uma conclusão já tirada com P + LR:

| # | afirmação (de onde vem) | critério por modelo | previsão |
|---|---|---|---|
| A1 | o refinado é **pior** que o bruto no Fake.br (RQ1) | bruto − refinado > 0, McNemar com Holm < 0,05 | **≥ 11 de 13** |
| A2 | o refinado é **pior** que o bruto no teste interno (RQ1) | o mesmo critério, no teste | **13 de 13** |
| A3 | a queda fora da base é grande (avaliação cruzada) | bruto: interno − Fake.br ≥ 10 pp | **≥ 12 de 13** |
| A4 | os controles dão o acaso (C4) | rótulos embaralhados entre 45% e 55% | **12 de 12** |
| A5 | o modelo atual não é um caso extremo | P + LR, bruto, no Fake.br, a até 3 pp da mediana dos 13 | **sim** |
| A6 | Fake.br → nosso é transferência **parcial** (avaliação inversa) | bruto, Fake.br → nosso `teste` entre 65% e 80% | **≥ 10 de 13** |
| A7 | o refinado também é pior na direção inversa (avaliação inversa) | bruto − refinado em Fake.br → nosso `teste` > 0, McNemar com Holm < 0,05 | **≥ 11 de 13** |
| A8 | a assimetria favorece o Fake.br (avaliação inversa) | bruto: (Fake.br → nosso) − (nosso → Fake.br) > 0 | **≥ 9 de 13** |

**Descritivo, sem previsão de faixa:** a matriz completa (13 × 2 braços ×
2 testes) e o melhor modelo no Fake.br, relatado como **exploratório**.

**Previsões de números (para conferir, sem leitura associada):**
- Ba, bruto, Fake.br: **70% a 82%**. O ajuste fino aprende melhor o estilo da
  corrente, e o estilo transfere só em parte.
- Ch, bruto, interno: **≥ P** (os n-gramas de caracteres pegam caixa alta,
  pontuação e ortografia, que são o atalho de forma da RQ4).
- NB gaussiano em Bc e Ol: os piores da matriz no interno.

## 3. Leituras fixadas antes do resultado

Para A1, A2, A3, A5, A6, A7 e A8:

| vale em | leitura |
|---|---|
| **≥ 10 de 13** | **robusta**: a conclusão não depende do modelo |
| **5 a 9 de 13** | **dependente do modelo**: a conclusão passa a ser escrita com a família de modelos em que vale |
| **≤ 4 de 13** | **não se sustenta**: era do modelo, não da base. A conclusão é revista no texto |

Para A4: **12 de 12** é o esperado. Qualquer modelo fora de 45% a 55% indica
vazamento ou erro no pipeline daquele modelo, e ele sai da contagem das outras
afirmações até ser explicado.

**Se o Ba for o único a quebrar uma afirmação**, isso é relatado à parte, por
ser o modelo mais forte e o mais usado na literatura (Guedes et al., PROPOR
2026).

## 4. Execução e custo

| etapa | onde | estimativa |
|---|---|---|
| P, Ch × LR, SVM, NB | esta máquina (WSL) | minutos |
| extração de Bc (31.149 textos × 2 campos) | PC com GPU | menos de 1 h |
| extração de Ol (`bge-m3` no Ollama) | PC com GPU | ~1 h |
| Ba, 2 braços × 3 épocas, nossa base → | PC com GPU | algumas horas |
| Ba, 2 braços × 3 épocas, Fake.br → (5.760 textos de treino) | PC com GPU | ~1 h |
| grade na direção Fake.br → (5 dobras × grade) | esta máquina | minutos a 1 h |
| LR, SVM, NB sobre Bc e Ol | esta máquina | minutos |

O script de extração confere o sha dos dados de entrada e grava os vetores
com o id de cada registro. O script de análise é commitado antes de rodar.

## 5. Previsões do pesquisador

> Preencher antes de rodar, em uma linha cada.

Registradas em _13_/_10_/2026, antes da rodada.

- **A1** (em quantos dos 13 modelos o refinado fica pior que o bruto no Fake.br?): __10____
- **A3** (em quantos dos 13 a queda fora da base passa de 10 pp?): __10____
- **Ba** (BERTimbau ajustado, bruto, no Fake.br: acurácia balanceada): __76% - 81%____
- **A6** (em quantos dos 13 o modelo treinado no Fake.br fica entre 65% e 80% na nossa base?): __13____
- **A8** (na maioria dos modelos, qual direção transfere melhor?): __fakebr para treino e nossa base para teste____
- **Melhor no Fake.br** (qual combinação você acha que vai melhor?): __SVM + Base bruta____

---

## Adendos

### Adendo 1 — 13/10/2026, resultado parcial: parte TF-IDF (6 de 13 modelos)

**Execução.** Pré-registro e script commitados antes (94182f3); rodada da parte
TF-IDF (P e Ch × LR, SVM, NB) em 15 min; predições e resumo em
`relatorios/matriz/` (6fa0cde). Conferência: P + LR reproduz os números da
avaliação inversa (91,19%, 73,53%, 89,57%, 78,07%).

**Resultados (braço bruto; acurácia balanceada):**

| modelo | nosso → teste | nosso → Fake.br | Fake.br → Fake.br | Fake.br → nosso |
|---|---:|---:|---:|---:|
| P + LR | 91,19 | 73,53 | 89,57 | 78,07 |
| P + SVM | 91,32 | 73,79 | 89,42 | 76,44 |
| P + NB | 90,97 | 70,15 | 86,25 | 82,66 |
| Ch + LR | 90,76 | 74,74 | 92,04 | 71,25 |
| Ch + SVM | 90,93 | 74,74 | 92,18 | 70,13 |
| Ch + NB | 89,95 | 69,58 | 88,40 | 80,72 |

O braço refinado está em `relatorios/matriz/resumo_tfidf.json`.

**Controle (A4) — desvio declarado.** O controle do §1.4 (rótulos embaralhados,
semente 0, grade escolhida na validação) deu 54,6% a 59,0%: fora de 45–55% em 5
dos 6 modelos. Pela regra do §3, esses modelos sairiam da contagem até haver
explicação. Diagnóstico (exploratório, feito depois da rodada; scripts em
`scripts/diagnosticos/controle_embaralhado_*.py`, rodados da raiz do repositório):
- P + LR com 20 sementes e a grade do controle original (C = 0,1): média 48,4%,
  desvio 4,8 pp, 6 de 20 fora de 45–55%. Com C = 10 (o do modelo real): média
  49,6%, desvio 1,5 pp, 0 de 20 fora. Não há vazamento: com regularização forte,
  o modelo de rótulos embaralhados se alinha ao acaso, a favor ou contra, com a
  direção que mais separa os textos (o gênero), e oscila muito.
- Controle refeito como no comparativo dos braços (3 sementes, hiperparâmetro
  do modelo real; f493bce, commitado antes de rodar): **5 de 6 dentro**. Ch + NB
  teve uma semente em 56,49%. Diagnóstico com 20 sementes no NB: média 49,49%
  (P) e 49,79% (Ch), com Ch + NB oscilando mais (desvio 3,4 pp; 4 de 20 fora,
  para os dois lados). Sem sinal de vazamento; os seis modelos ficam na contagem.

**Apuração parcial** (`scripts/apurar_matriz.py`, commitado antes de rodar;
saída `relatorios/matriz/apuracao.json`; Holm sobre os 6 modelos):

| afirmação | vale em | observação |
|---|---|---|
| A1 refinado pior no Fake.br | **6 de 6** | +4,9 a +8,9 pp, todos com p (Holm) < 10⁻¹⁶ |
| A2 refinado pior no teste interno | **6 de 6** | +9,5 a +11,2 pp |
| A3 queda fora da base ≥ 10 pp | **6 de 6** | 16,0 a 20,8 pp |
| A4 controle | 5 de 6 | ver acima |
| A5 P + LR perto da mediana | sim | mediana 73,66%, P + LR 73,53% |
| A6 Fake.br → nosso entre 65% e 80% | 4 de 6 | os dois NB passam de 80% (82,7% e 80,7%) |
| A7 refinado pior em Fake.br → nosso | 4 de 6 | Ch + LR (+1,7, p = 0,053) e Ch + SVM (+0,3) não |
| A8 assimetria a favor do Fake.br | 4 de 6 | Ch + LR e Ch + SVM invertem (−3,5 e −4,6 pp) |

**Leitura provisória** (a do §3 só com os 13): as conclusões sobre a nossa
base (A1, A2, A3) valem em todos os modelos TF-IDF; as que dependem da direção
inversa (A6, A7, A8) já dependem da representação: com n-gramas de caracteres,
o modelo do Fake.br transfere menos para a nossa base, e o refinado deixa de
ser pior nessa direção.

### Adendo 2 — 14/10/2026, resultado parcial: parte densa (12 de 13 modelos)

**Execução.** Vetores extraídos no PC com GPU (`scripts/extrair_vetores.py`) e
conferidos aqui: sha da entrada igual ao de `braco_refinado.jsonl`
(cdf4c63ee2190ede), 31.149 ids na mesma ordem, sem NaN nem linhas nulas.
Parte densa e controle rodados com o código commitado em f493bce (8 min).
O BERTimbau ajustado (Ba) ainda não rodou.

**Resultados (braço bruto; acurácia balanceada):**

| modelo | nosso → teste | nosso → Fake.br | Fake.br → Fake.br | Fake.br → nosso | controle (3 sementes) |
|---|---:|---:|---:|---:|---|
| Bc + LR | 97,37 | 76,33 | 94,56 | 82,78 | 51,8 / 53,8 / 48,0 |
| Bc + SVM | 97,38 | 76,06 | 94,83 | 82,22 | 51,7 / 54,2 / 47,7 |
| Bc + NB gaussiano | 81,77 | 62,74 | 83,90 | 86,46 | 47,7 / **77,7** / 55,4 |
| Ol + LR | 97,29 | 73,61 | 88,07 | 85,97 | 54,6 / 53,6 / 52,7 |
| Ol + SVM | 97,44 | 74,00 | 88,47 | 85,65 | 54,9 / 53,2 / 50,7 |
| Ol + NB gaussiano | 92,35 | 66,74 | 79,67 | 79,78 | **65,0 / 64,4 / 58,8** |

**Controle (A4).** Os dois NB gaussianos saem de 45–55%. Diagnóstico
(exploratório, `scripts/diagnosticos/controle_embaralhado_nb_gaussiano.py`,
10 sementes): o NB gaussiano com rótulos embaralhados vai de 31,0% a 77,7%
(Bc) e de 35,3% a 68,6% (Ol), para os dois lados (média ~55%). Com os mesmos
vetores e a mesma divisão, LR e SVM ficam entre 47,7% e 54,9%. Não é vazamento
de dados nem do pipeline, que é o mesmo; é instabilidade do classificador sobre
vetores de 768–1024 dimensões. A hipótese de que ele decide pela distância ao
centro dos dados **não se confirmou** (a distância sozinha separa 57–58%, e a
correlação com as predições varia de sinal). **Pela regra do §3, os dois NB
gaussianos saem da contagem das outras afirmações**: falham no controle, e a
explicação encontrada mostra que os números deles não são confiáveis, inclusive
com os rótulos reais.

**Apuração parcial** (`relatorios/matriz/apuracao.json`; Holm sobre os 12):

| afirmação | com os 12 | sem os 2 NB gaussianos (10) | observação |
|---|---|---|---|
| A1 refinado pior no Fake.br | 11 de 12 | **10 de 10** | só o Bc + NB não (+0,8 pp, p = 0,33) |
| A2 refinado pior no teste interno | 12 de 12 | **10 de 10** | |
| A3 queda fora da base ≥ 10 pp | 12 de 12 | **10 de 10** | 16,0 a 23,7 pp |
| A4 controle | 9 de 12 | 9 de 10 | Ch + NB multinomial: uma semente em 56,5% (adendo 1) |
| A5 P + LR perto da mediana | sim | sim | mediana 73,70%, P + LR 73,53% |
| A6 Fake.br → nosso entre 65% e 80% | 5 de 12 | 4 de 10 | **6 dos 10 passam de 80%** (até 86,0%) |
| A7 refinado pior em Fake.br → nosso | 10 de 12 | 8 de 10 | Ch + LR e Ch + SVM não |
| A8 assimetria a favor do Fake.br | 10 de 12 | 8 de 10 | Ch + LR e Ch + SVM invertem |

**Leitura provisória** (a do §3 só com os 13): A1, A2 e A3 valem em todos os
modelos que passam no controle. As representações mais fortes (BERTimbau e
bge-m3, com LR ou SVM) sobem o teste interno de ~91% para ~97%, mas o Fake.br
só de ~74% para 74–76%: a queda fora da base **aumenta** (21–24 pp). Na
direção inversa, o modelo do Fake.br chega a 82–86% na nossa base, acima da
faixa de "transferência parcial" prevista em A6.

### Adendo 3 — 15/10/2026, resultado final (13 de 13 modelos)

**Execução do Ba.** `scripts/ajustar_bertimbau.py` rodado no PC com GPU, sem
alteração no código commitado em f493bce (o commit 8feb529 do pesquisador
traz as predições, o log e mudanças no notebook, não no script). Log em
`relatorios/matriz/log_ajustar_bertimbau.txt` (10 min; o "NativeCommandError"
do PowerShell é o aviso do Hugging Face redirecionado, não um erro). Épocas
escolhidas na validação: bruto 1 (nossa direção) e 3 (Fake.br); refinado 3 e 2.

| Ba | nosso → teste | nosso → Fake.br | Fake.br → nosso |
|---|---:|---:|---:|
| bruto | **99,05** | **70,26** | 76,87 |
| refinado | 88,11 | 67,97 | 76,60 |

O modelo mais forte é o que mais cai fora da base: 28,8 pp, contra 16–24 pp dos
outros. No Fake.br, fica abaixo de todos os modelos TF-IDF com LR e SVM.

**Apuração final** (`relatorios/matriz/apuracao.json`). O script conta os 13; a
regra do §3 tira da contagem os modelos que falham no controle sem explicação
que os valide (os dois NB gaussianos, adendo 2). Para 11 modelos, os limiares
do §3 foram aplicados na mesma proporção (robusta ≥ 10/13 ≈ 77%, isto é, ≥ 9 de
11; dependente de 5/13 a 9/13, isto é, de 5 a 8 de 11):

| afirmação | 13 modelos | leitura | 11 válidos | leitura |
|---|---|---|---|---|
| A1 refinado pior no Fake.br | 12 de 13 | robusta | **11 de 11** | **robusta** |
| A2 refinado pior no teste interno | 13 de 13 | robusta | **11 de 11** | **robusta** |
| A3 queda fora da base ≥ 10 pp | 13 de 13 | robusta | **11 de 11** | **robusta** |
| A4 controle | 9 de 12 (Ba sem controle) | — | 9 de 10 | Ch + NB multinomial no limite (adendo 1) |
| A5 P + LR perto da mediana | sim (mediana 73,61%) | sim | sim | sim |
| A6 Fake.br → nosso entre 65% e 80% | 6 de 13 | dependente do modelo | 5 de 11 | **dependente do modelo** (5 passam de 80%) |
| A7 refinado pior em Fake.br → nosso | 10 de 13 | robusta | 8 de 11 | **dependente do modelo** (Ba, Ch + LR, Ch + SVM não) |
| A8 assimetria a favor do Fake.br | 11 de 13 | robusta | 9 de 11 | **robusta** (Ch + LR e Ch + SVM invertem) |

A leitura que vale é a dos 11 válidos. A7 muda de leitura conforme a contagem,
e fica declarada como **dependente do modelo**, a leitura mais conservadora.

**Confronto com as previsões:**

| | resultado (11 válidos) | §2 | pesquisador |
|---|---|---|---|
| A1 | 11 de 11 (12 de 13) | ≥ 11 de 13 ✓ | 10 ✗ (a afirmação valeu em mais modelos que os 10 previstos) |
| A2 | 11 de 11 | 13 de 13 ✓ | — |
| A3 | 11 de 11 (13 de 13) | ≥ 12 de 13 ✓ | 10 ✗ (a afirmação valeu em mais modelos que os 10 previstos) |
| A4 | 9 de 10 | 12 de 12 ✗ | — |
| A5 | sim | sim ✓ | — |
| A6 | 5 de 11 | ≥ 10 de 13 ✗ | 13 ✗ |
| A7 | 8 de 11 | ≥ 11 de 13 ✗ | — |
| A8 | 9 de 11 | ≥ 9 de 13 ✓ | "Fake.br treina, nossa base testa" ✓ |
| Ba, bruto, Fake.br | 70,26% | 70–82% ✓ (no limite) | 76–81% ✗ |
| Ch ≥ P no interno | 90,76 < 91,19 (LR) | ✗ | — |
| melhor no Fake.br | Bc + LR 76,33% (Bc + SVM 76,06%) | — | "SVM + bruto": Bc + SVM em 2º ✗ |

**Leitura para o TCC.** As três conclusões sobre a nossa base valem em todos os
modelos válidos: o refinamento por LLM piora o desempenho dentro e fora da base
(RQ1), e a queda fora da base é grande em qualquer modelo (RQ4). Ela **cresce**
com a força do modelo: as representações mais fortes aprendem mais o que é
próprio da base (99% dentro, 70% fora para o BERTimbau ajustado). As conclusões
sobre a direção inversa (Fake.br → nossa base) dependem do modelo e entram no
texto com essa ressalva.

**Desvios e limites:** os do adendo 1 (controle refeito) e do adendo 2 (NB
gaussianos fora da contagem); o Ba sem controle e sem a célula Fake.br →
Fake.br (exceção declarada no §1.3); a aplicação proporcional dos limiares do
§3 para 11 modelos, que o pré-registro não previa.
