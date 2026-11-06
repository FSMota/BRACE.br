# Perguntas de pesquisa — TCC

As perguntas, a motivação de cada uma, o resultado atual e as contribuições
declaráveis. Cada número vem de um pré-registro em `relatorios/` (desenho,
previsões e leituras commitados antes da rodada; resultados em adendos datados).
Atualizado em 19/10/2026.

> **Numeração (19/10/2026):** as perguntas foram renumeradas para priorizar as que
> tiveram resultados positivos: a atual RQ1 (validade) era a RQ4, e a atual RQ4
> (representação) era a RQ1. Os pré-registros em `relatorios/` usam a numeração
> antiga e não foram editados.

**Origem do trabalho:** o levantamento de Baracho, Lisboa e Lopes (WICS 2025,
doi:10.5753/wics.2025.9464) apontou os problemas das bases de fake news em
português (volume reduzido, seleção manual, concentração temática, poucas
fontes, quase todas apenas textuais — só a Desinfopedia inclui imagens e
vídeos). Ele foi o motivador da construção e o direcionador
das soluções.

---

## Pergunta geral (a principal)

> **É viável construir automaticamente, a partir de portais de checagem, um corpus
> de larga escala em português para detecção de desinformação que sustente tanto a
> comparação entre representações textuais quanto a avaliação de explicabilidade —
> e quais são os limites de validade dessa construção?**

**Resposta atual: viável, com estas ressalvas medidas.** A resposta é a própria
base, a contribuição central: 28.002 instâncias (14.856 falsas + 13.146
verdadeiras), subconjunto pareado de 11.975 + 11.975 por tema × ano × parte da
divisão temporal, veredito por alegação no lado falso, justificativa do checador
em 99,7% das falsas, licença aberta no lado verdadeiro. É mais que o dobro da
maior base do levantamento de Baracho et al. (FakeRecogna, 11.902). As quatro
RQs qualificam a base: como foi construída, o que permite estudar e quais são
os limites.

---

## RQ1 — Validade: atalhos e limites

> **Quais atalhos estatísticos um corpus construído automaticamente a partir de
> portais de checagem introduz, e quanto da acurácia reportada eles explicam?**

Validade interna: sem ela, RQ4 e RQ3 ficam sem lastro.

**Resultado.** O atalho dominante é o **gênero** (corrente de WhatsApp ×
matéria de agência), e ele é estável entre modelos.

| atalho / medida | resultado | fonte |
|---|---|---|
| forma (17 atributos, nenhuma palavra) | 87,21% interno; 65,18% no Fake.br (≈ 2/3 da transferência) | `modelo_estilo.py`, adendo 5 da avaliação cruzada |
| comprimento | corpo inteiro separa 83,5% → neutralizado com título + lide (47,7%); a frase refinada sozinha, 57,25% (C1, abaixo do limiar de 60%) | `DECISOES_E_APRENDIZADOS.md`; `PREREGISTRO_BRACO_REFINADO.md` |
| queda fora da base (TF-IDF + LR) | 91,19% → 73,51% no Fake.br (17,7 pp); **nos mesmos itens, 16,0 pp** [14,9; 17,1] | `PREREGISTRO_AVALIACAO_CRUZADA.md`, `PREREGISTRO_AVALIACAO_INVERSA.md` |
| direção inversa | Fake.br → nossa base: 78,07% ("transferência parcial"); com 7.200 textos de treino de cada lado, o Fake.br transfere 5,45 pp melhor | `PREREGISTRO_AVALIACAO_INVERSA.md` |
| época | sem datação, só o ano acerta 64,0% (com datação, 47,0%); o modelo de texto não usa esse atalho (−0,42 pp, p = 0,82): "atalho disponível, pouco usado". Exploratório: na validação cruzada aleatória, ano-mês e data completa ainda separam 57,6% e 60,8% (50% na divisão temporal; `scripts/diagnosticos/atalho_data.py`) → a data é para controlar e dividir, não atributo | `PREREGISTRO_PAREAMENTO_DATADO.md` |
| data da checagem como proxy da circulação | mediana de 3 dias entre o post viral incorporado e a checagem; 93,1% em até um ano (248 artigos; validação humana de 30 pendente) | `PREREGISTRO_ATRASO_CHECAGEM.md` |
| robustez entre modelos | 13 modelos (4 representações × LR, SVM, NB + BERTimbau ajustado); nos 11 válidos, "refinado pior" e "queda ≥ 10 pp" valem em 11 de 11. A queda **cresce** com a força do modelo: BERTimbau ajustado 99,05% interno, 70,26% no Fake.br. A assimetria inverte com n-gramas de caracteres | `PREREGISTRO_MATRIZ_MODELOS.md` (adendos 1–3) |
| checagens de discurso (G1 e Lupa) | 57 instâncias de discurso de político, não de boato; marcadas, não apagadas; sem elas, nenhuma métrica muda mais de 0,44 pp | `PREREGISTRO_SENSIBILIDADE_DISCURSO.md` |
| vazamento de veredito, assinatura da fonte | ~0,2 pp cada | `DECISOES_E_APRENDIZADOS.md` §4 |
| concentração de fonte | 99,6% das falsas do Boatos.org × 100% das verdadeiras da Agência Brasil — estrutural, declarado; barreiras de acesso (paywall, termos de uso, bloqueio de robôs) documentadas | — |
| vazamento no pipeline | nenhum: rótulos embaralhados ~50% (3 sementes, hiperparâmetro do modelo real); sobreposição zero entre as partes e entre as bases | controles; diagnósticos em `scripts/diagnosticos/` |

---

## RQ2 — Método de construção: extração determinística vs. LLM

> **Em que medida a extração ancorada na estrutura do documento supera a extração
> por LLM na construção automática de corpora de checagem?**

Emergiu da construção. As primeiras medidas (estrutura 99,4% × LLM 87,1% de
literalidade) comparavam conjuntos disjuntos de artigos.

**Resultado (experimento pareado, 400 artigos, o LLM recebe o artigo sem o
recorte estrutural):** leitura **"parcial"**.
- rendimento do LLM: 97,5%; literalidade **97,2% × 100%** da estrutura nos
  mesmos artigos (McNemar 11 × 0, p = 0,001): a estrutura é mais fiel, por 2,8 pp;
- recorte igual ao estrutural em 62,5%; a via `html` é a mais difícil;
- contaminação pela voz do checador quase nula com o prompt v3 (0,5%);
- a diferença antiga era, em boa parte, **seleção**;
- exploratório: em 20 recortes divergentes, a estrutura errou em 11 (tweets de
  evidência, links do site, perguntas e respostas do checador, versões juntas).

**Lição:** "não peça ao modelo o que a fonte informa" continua valendo (pedir o
tema ao LLM inflou "Saúde" de 7,7% para 47%), mas a fonte também precisa de
auditoria. Fonte: `PREREGISTRO_RQ2_PAREADO.md`.

---

## RQ3 — Explicabilidade

> **As explicações produzidas por modelos de detecção de desinformação
> correspondem ao raciocínio que checadores humanos empregam?**

Habilitada pelo campo `justificativa` (precedentes em inglês: LIAR-PLUS,
Alhindi et al., 2018; PUBHEALTH, Kotonya & Toni, 2020). Em português, o
FACTCK.BR (Moreno e Bressan, 2019) traz o texto da checagem (`reviewBody`) para
1.309 alegações, a partir do ClaimReview; esta base o traz para quase 15 mil
instâncias, com a mensagem viral como entrada (conferido em 19/10/2026).

**Uso A — alinhamento de racionais (feito):** para cada falsa acertada, as 5
palavras de maior contribuição para FALSO (peso × TF-IDF) contra as da
justificativa; controle: palavras quaisquer do mesmo texto. Bruto: 15,3% ×
12,5% (**+2,8 pp** [2,2; 3,4]); refinado: 23,1% × 22,2% (**+0,9 pp**). Leitura:
**nenhum alinhamento além do que o assunto do texto já explica.** O modelo
destaca marcas de corrente ("agora", "urgente", "atenção"); o checador escreve
evidências. Fonte: `PREREGISTRO_RQ3_ALINHAMENTO.md`.

**Uso B — recuperação de checagens (em andamento):** 50 boatos do teste buscados
entre as 9.393 falsas anteriores por TF-IDF e bge-m3, sobre o texto e sobre o
`ideia_central`, mais um sorteio do tema; 665 pares em julgamento humano às
cegas. Fonte: `PREREGISTRO_RECUPERACAO.md`.

**Restrição:** a justificativa **não pode ser entrada** do classificador de
veracidade — só existe do lado falso. A RQ3 reprovou o **classificador** como
explicador; a justificativa foi a régua da medida.

---

## RQ4 — Representação: bruto vs. refinado por LLM

> **O refinamento por LLM da representação textual altera o desempenho de modelos
> de detecção de desinformação em relação ao texto bruto? Em que direção, e o
> efeito se mantém sob avaliação cross-dataset?**

Mesmo modelo, mesmas divisões, variando só a entrada:

| configuração | entrada FALSO | entrada VERDADEIRO |
|---|---|---|
| bruta | viral limpo (`texto_entrada`) | título + lide (`texto_entrada`) |
| refinada | `ideia_central`, prompt cego `cego-v2` | `ideia_central`, prompt cego `cego-v2` |
| combinada | bruta + refinada | bruta + refinada |

**Condição de validade:** um único prompt, cego ao veredito, sobre o mesmo tipo de
entrada nos dois lados e no Fake.br.

**Resultado: o refinamento piora, dentro e fora da base.** Refinado 81,73%
interno e 66,31% no Fake.br, contra 91,19% e 73,51% do bruto (−7,22 pp no
Fake.br [−8,40; −6,06]). O combinado é redundante (−0,57 pp, n.s.; 64% do peso
no bruto). A hipótese registrada (queda menor no refinado) se confirmou pela
letra (2,23 pp [0,66; 3,85], p de Holm 0,014), mas pela via (b): o refinado cai
menos porque começa mais baixo. Vale nos 11 modelos válidos da matriz. O
refinado aprende assunto ("globo", "ivermectina" × "senado", "câmara"), e o
metatexto do LLM ("o texto…") aparece em 4,4% das falsas × 0,5% das
verdadeiras. Checagens do braço: C1 a C4 passaram; na C3 (anotação humana às
cegas + LLM como segundo anotador + adjudicação), 13% das frases distorcem o
sentido nas duas classes. Fonte: `PREREGISTRO_BRACO_REFINADO.md`.

---

## Contribuições declaráveis

1. **Recurso.** Corpus em português de 28.002 instâncias (pareado 11.975 +
   11.975), com veredito por alegação no lado falso, justificativa do checador,
   procedência rastreável por registro, marcações em vez de exclusões, divisão
   temporal e licença aberta no lado verdadeiro. Mais que o dobro da maior base
   do levantamento de Baracho et al. (2025). *"Maior corpus público em
   português": confirmar no levantamento bibliográfico.*
2. **Auditoria de validade.** Os atalhos da base medidos e publicados junto
   (forma, comprimento, época, fonte), com avaliação nas duas direções com o
   Fake.br, queda medida nos mesmos itens e conclusões verificadas em 13
   modelos.
3. **Método de extração auditado.** No mesmo artigo, estrutura e LLM ficam a 2,8
   pp de literalidade, e as duas vias erram; a vantagem antiga era seleção.
4. **Resultado negativo medido.** Resumir o texto com LLM antes do treino, uma
   prática comum, piora a generalização em português.
5. **Explicabilidade avaliável.** A justificativa do checador permite medir se
   o modelo decide pelos motivos certos (ele não decide) e serve de base para a
   recuperação de checagens.
6. **Protocolo.** Pré-registro de cada experimento, com previsões e leituras
   commitadas antes da rodada, e desvios declarados em adendos.

## Pendências

- anotação manual das 200 falsas (`relatorios/anotacao/`), obrigatória;
- julgamento humano do protótipo de recuperação (uso B) e segunda anotação;
- validação humana de 30 artigos do atraso da checagem;
- integrar a marcação `checagem_de_discurso` ao `fundir_corpus.py` na versão
  publicada da base.
