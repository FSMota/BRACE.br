# Pré-registro — alucinação na extração por LLM (exploratório)

> Escrito em 29/10/2026, **antes** de calcular qualquer cobertura. Só foi
> contada a população (11.113 falsas da via `llm`, todas com o texto bruto do
> artigo guardado). Commitar este arquivo e o script antes de rodar. Correções
> vão num adendo datado, no fim.

**Origem.** Na anotação manual das 200 falsas (29/10/2026), o pesquisador achou
duas entradas da via LLM que **não existem no artigo** de origem: "O dólar vai
subir 50% amanhã…" (artigo sobre Luiz Gonzaga) e "VACINA CONTRA A GRIPE PODE
CAUSAR AUTISMO?…" (artigo sobre a Torre Eiffel). Nos dois artigos, **todas** as
instâncias eram inventadas (3 e 2), com justificativas também inventadas, e o
boato real não estava na base. A anotação só pega esse erro quando o texto não
combina com o assunto do artigo; uma invenção no mesmo tema passa como `boato`.
Por isso a taxa da via LLM na anotação (3,3%) é um piso.

**O que já existe.** O pipeline marca, em `verificacao_viral`, se o texto
extraído aparece literalmente no artigo (`conferido`), não aparece com a fonte
completa (`falhou`) ou não aparece com a fonte cortada no teto de 5.000
caracteres (`inconclusivo_fonte_truncada`). A taxa de literalidade da via `llm`
já relatada é 87,1%. Mas `falhou` mistura duas coisas: **paráfrase** (o modelo
reescreveu um trecho que existe) e **invenção** (o trecho não existe). Esta
medida separa as duas.

**Pergunta:** quantas falsas da via LLM são invenção, e não paráfrase?

---

## 1. Desenho fixado

| item | valor |
|---|---|
| população | falsas de `datasets/corpus_fundido.jsonl` com `origem_viral = llm`; relatado também só no subconjunto pareado |
| texto avaliado | `texto_entrada` (o que entra no modelo) |
| fonte | `texto_bruto_jornalista` do artigo em `datasets/dataset_bruto.json` (pela URL) |
| cobertura lexical | fração das palavras de conteúdo do texto (minúsculas, sem acento, ≥ 4 letras, fora de uma lista de palavras vazias) que aparecem em qualquer lugar do artigo. Textos com menos de 3 palavras de conteúdo ficam de fora (relatado) |
| faixas | **< 0,5: invenção provável**; 0,5 a 0,8: paráfrase forte; ≥ 0,8: literal ou quase |
| fonte cortada | itens com artigo no teto (≥ 5.000 caracteres) são relatados à parte: neles, cobertura baixa pode ser só o trecho que ficou fora da cópia |
| concentração | por artigo: em quantos artigos com alguma invenção provável **todas** as instâncias da via LLM caem na faixa < 0,5 |
| validação humana | 30 itens sorteados (semente 0) da faixa < 0,5 com fonte completa e 20 da faixa 0,5 a 0,8. O pesquisador lê o texto e abre o link, **sem ver a cobertura nem a faixa**, e marca: **existe no artigo** (literal ou reescrito), **inventado** ou **não dá para saber**. Planilha em branco commitada antes da leitura |

**Controle de sanidade (casos conhecidos):** as 5 instâncias dos dois artigos
achados na anotação devem cair na faixa < 0,5. Se alguma não cair, o limiar
está errado, e isso é declarado antes de qualquer leitura do resto.

**Limites declarados agora:**
- palavras de conteúdo em comum não provam que o trecho existe (uma invenção no
  mesmo tema compartilha vocabulário com o artigo); a medida tende a
  **subestimar** invenções no mesmo tema, e a validação mede isso só na faixa
  0,5 a 0,8;
- o texto bruto guardado pode diferir do site (cortes, blocos não capturados).

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| A1 | invenção provável (< 0,5, fonte completa) entre as falsas da via LLM | **≤ 3%** |
| A2 | concentração: artigos em que todas as instâncias LLM são < 0,5, entre os artigos com alguma | **≥ 50%** |
| A3 | validação: "inventado" entre os 30 da faixa < 0,5 | **≥ 70%** |
| A4 | validação: "inventado" entre os 20 da faixa 0,5 a 0,8 | **≤ 20%** |

## 3. Leituras fixadas antes do resultado

| resultado | leitura |
|---|---|
| A3 ≥ 70% **e** taxa estimada de invenção ≥ 1% do pareado | invenção é um defeito material: as instâncias da faixa < 0,5 são marcadas (`entrada_inventada`), retiradas da versão publicada, e o modelo de referência é refeito sem elas como teste de sensibilidade (como na checagem de discurso) |
| A3 ≥ 70% **e** taxa < 1% | defeito real, mas pequeno: marcar, retirar da versão publicada e declarar como limitação, sem refazer os modelos |
| A3 < 70% | a cobertura lexical não separa bem invenção de paráfrase: relatar a taxa só como limite superior e declarar a medida como fraca |

A taxa estimada de invenção é a fração da faixa < 0,5 multiplicada pela
proporção de "inventado" na validação dela, mais a fração da faixa 0,5 a 0,8
multiplicada pela proporção de "inventado" na validação dela.

## 4. Previsões do pesquisador

> Preencher antes de rodar, em uma linha cada, com a data.

Registradas em 29/10/2026, antes da rodada (data preenchida pelo assistente no commit).

- **A1** (que % das falsas da via LLM é invenção?): ___<10%___
- **A2** (as invenções se concentram em artigos inteiros?): ___<30%___
- **A3** (dos 30 com cobertura baixa, quantos são inventados?): ___<15___

---

## Adendos

### Adendo 1 — 29/10/2026, resultado da medida (validação humana pendente)

**Execução.** Pré-registro e script commitados antes (3e32e18). 11.101 falsas
da via LLM avaliadas; 12 ficaram de fora por terem menos de 3 palavras de
conteúdo. Saída: `relatorios/alucinacao/resultado_cobertura.json`.

**Controle de sanidade:** as 5 instâncias conhecidas caíram na faixa < 0,5
(coberturas de 0 a 0,23). O limiar fica como pré-registrado.

**Resultado.** A distribuição é **bimodal**: quase tudo é literal ou quase, e
o resto não tem quase nada em comum com o artigo; o meio é praticamente vazio.

| conjunto | n | < 0,5 | 0,5 a 0,8 | ≥ 0,8 |
|---|---:|---:|---:|---:|
| fonte completa | 7.777 | **94 (1,21%)** | 4 (0,05%) | 7.679 |
| fonte completa, pareado | 6.283 | **71 (1,13%)** | 4 | 6.208 |
| fonte cortada | 3.324 | 17 (0,51%) | 2 | 3.305 |

Por `verificacao_viral`: nenhum `conferido` cai abaixo de 0,8 (o esperado,
já que o texto literal tem todas as palavras no artigo); entre os 746
`falhou`, 94 (12,6%) caem abaixo de 0,5 e 648 (86,9%) ficam em 0,8 ou mais. Ou
seja, a maioria das falhas de literalidade é **paráfrase**, e uma minoria é
candidata a invenção.

**Concentração (A2):** dos 56 artigos com alguma instância abaixo de 0,5, em
**52 (92,9%)** todas as instâncias da via LLM estão abaixo de 0,5. A invenção é
um fenômeno de **artigo inteiro**: quando o modelo inventa, inventa tudo o que
extrai daquele artigo.

**Confronto parcial:**

| | resultado | §2 | pesquisador |
|---|---|---|---|
| A1 invenção provável | 1,21% | ≤ 3% ✓ | < 10% ✓ |
| A2 concentração | 92,9% | ≥ 50% ✓ | < 30% ✗ |
| A3, A4 | pendentes | | |

**Desvio declarado — amostra da faixa média.** O desenho previa sortear 20
itens da faixa 0,5 a 0,8 com fonte completa, mas ela só tem 4. Entram os 4. A
A4 passa a ser só descritiva. Planilha com 34 itens (30 + 4, embaralhados, sem
cobertura nem faixa) gerada e commitada em branco, antes da leitura.

### Adendo 2 — 29/10/2026, mudança de quem valida (antes de qualquer validação)

**Desvio declarado.** Por falta de tempo do pesquisador, que ainda anota a
planilha de recuperação (665 pares), a validação dos 34 itens passa a ser feita
**pelo assistente**, e não pelo pesquisador. Decidido antes de qualquer item
ser validado.

- **Fonte independente.** O assistente confere cada texto na página **ao vivo**
  do artigo no Boatos.org, e não na cópia guardada que a medida usou. Assim, um
  trecho que existe no site, mas ficou fora da cópia, aparece como
  .
- **Critério.**  se o trecho, literal ou reescrito, está na
  página;  se nem a alegação nem o trecho aparecem; 
  se a página não carregar ou mudou.
- **Cegamento parcial.** O assistente gerou a amostra e sabe que 30 dos 34 são
  da faixa baixa; não consulta a faixa de cada item durante a validação.
- **Concordância.** O pesquisador valida às cegas 5 dos 34, sorteados com
  semente 0, numa planilha sem as respostas do assistente. A concordância é
  relatada; com 5 itens, ela é só um indicativo.
- **Sensibilidade.** O modelo de referência é refeito sem as instâncias da faixa
  < 0,5, qualquer que seja a leitura da §3, porque o custo é pequeno.

### Adendo 3 — 29/10/2026, validação e sensibilidade

**Validação (assistente, site ao vivo).** Os 34 artigos foram baixados de novo
do Boatos.org (todos responderam). Saída:
`relatorios/alucinacao/resultado_validacao.json`.

| faixa | n | inventado | existe no artigo |
|---|---:|---:|---:|
| < 0,5 | 30 | **30 (100%)** | 0 |
| 0,5 a 0,8 | 4 | 2 | 2 |

Na faixa baixa, nenhum texto aparece na página, e em todos o artigo é sobre
outro boato. São frases genéricas de desinformação, sobretudo sobre vacinas, a
Terra plana, o dólar e "o presidente", repetidas com variações entre artigos
diferentes. Na faixa média, os 2 que existem são a alegação do título
reescrita e um trecho literal; os 2 inventados são uma frase sem relação e uma
distorção que troca o sujeito ("Faustão" vira "o presidente").

**Taxa estimada de invenção:** **1,24%** das falsas da via LLM com fonte
completa e **1,16% no subconjunto pareado**.

**Confronto:**

| | resultado | §2 | pesquisador |
|---|---|---|---|
| A1 | 1,21% | ≤ 3% ✓ | < 10% ✓ |
| A2 | 92,9% | ≥ 50% ✓ | < 30% ✗ |
| A3 | 30 de 30 | ≥ 70% ✓ | < 15 ✗ |
| A4 | 2 de 4 (descritiva, adendo 1) | ≤ 20% ✗ | — |

**Leitura fixada (§3): A3 ≥ 70% e taxa ≥ 1% do pareado → defeito material.**
As 111 instâncias da faixa < 0,5 (94 com fonte completa e 17 com fonte
cortada, estas não validadas, mas do mesmo padrão) ficam marcadas em
`relatorios/marcacoes/entrada_inventada.json` e saem da versão publicada.

**Sensibilidade (modelo de referência retreinado sem elas).** Das 111, 68
estavam no treino, 5 na validação e 12 no teste. Saída:
`relatorios/alucinacao/resultado_sensibilidade.json`.

| braço | teste, com → sem | Fake.br, com → sem |
|---|---|---|
| bruto | 91,19 → 91,34 (+0,15) | 73,53 → 73,58 (+0,05) |
| refinado | 81,73 → 81,73 (0,00) | 66,31 → 66,26 (−0,05) |

O hiperparâmetro escolhido na validação não mudou (C = 10). **Retirar as
entradas inventadas não muda nenhuma conclusão**: a variação máxima é de 0,15
ponto, dentro do ruído. O defeito importa para a qualidade da base publicada e
para a RQ2 (é um modo de falha que só a via LLM tem), não para os resultados
dos modelos.

**Concordância.** Planilha do pesquisador com os 5 itens sorteados (n = 3, 17,
25, 27 e 33), sem as respostas do assistente:
`relatorios/alucinacao/planilha_concordancia_pesquisador.xlsx`. Resultado
(29/10/2026): **5 de 5 concordâncias**, todos `inventado` nos dois. O kappa
não é calculável (uma só categoria), e com 5 itens a concordância é só um
indicativo.
