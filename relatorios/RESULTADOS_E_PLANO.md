# Resultados até aqui e plano para a escrita

> Estado em 06/10/2026. Consolida o que foi medido, organizado pelas quatro
> perguntas de pesquisa, e o que falta para cada uma. Números com fonte
> (script e arquivo de resultado) para poderem ser citados.
>
> Os números do braço bruto são os do **corpus com a extração v3** (adendo 5 do
> pré-registro) — os de referência. Os das bases anteriores (adendos 1–4) ficam
> no pré-registro e mostram que as limpezas não mudaram nenhuma conclusão.

---

## 1. O corpus — contribuição central

| | valor | fonte |
|---|---:|---|
| instâncias FALSO | 14.856 (extração v1 10.902 + v2 3.243 + v3 711) | `dataset_tcc_fake_news_final.json` |
| instâncias VERDADEIRO | 13.146 (Agência Brasil, CC BY 3.0 BR) | `dataset_verdadeiras.jsonl` |
| **total** | **28.002** | `corpus_fundido.jsonl` |
| subconjunto pareado (tema × ano × split) | **11.975 + 11.975 = 23.950** | `fundir_corpus.py` |
| tamanho mediano da entrada no pareado | falsas 247 × verdadeiras 240 caracteres | idem |
| artigos sem saída válida (JSON inválido, 3 tentativas) | 3 de 13.140 (0,02%) | `dataset_bruto.json` |
| período | 2013–2026 | |
| divisão | temporal, 70/10/20, quase-duplicatas do mesmo lado | `corrigir_corpus.py`, `preparar_verdadeiras.py` |

Comparação: Fake.br 7.200; FakeRecogna 11.900. **"Maior corpus público em
português" precisa ser confirmado no levantamento bibliográfico** antes de
entrar no texto — citar os corpora comparados e as datas.

Diferenciais além do volume, todos já implementados:
- **definição por item**: cada falsa tem veredito de checador profissional
  para aquela alegação (o Fake.br rotula pelo site de origem);
- **justificativa do checador** por instância (99,7%) — nenhum corpus em
  português tem (precedentes: LIAR-PLUS, PUBHEALTH);
- **proveniência** de cada extração (`origem_viral`, `texto_via`,
  `lide_origem`, `entrada_prefixos_removidos`);
- **nada apagado, tudo marcado** (duplicata, entrada degenerada, veredito,
  repetida, pareado) — qualquer filtro pode ser refeito por terceiros;
- **auditoria de atalhos publicada junto** (seção RQ4).

### Método de construção (capítulo de método)

**Lado falso:** raspagem do Boatos.org, Lupa e G1 → extração do texto viral
(`<blockquote>`, marcador textual ou LLM) → extração por LLM (qwen3.5:9b, saída
estruturada por JSON Schema) → correções A/B → **extração v2** em 2.625 artigos
com defeito (`limpeza_viral.py` nas três vias; literalidade 97,2%) → **extração
v3** recusando a voz do checador (873 títulos com "#boato" no pareado; 2.314
artigos reprocessados) → tema da URL, data, suporte → MinHash/LSH (Jaccard ≥
0,70: 223 grupos, 408 excedentes) → divisão temporal.

**Lado verdadeiro:** descoberta de URLs (sitemap 2013–2020: 150.641; CDX do
Internet Archive 2021–2026: 70.937) → cota por (tema, ano) igual à demanda
das falsas, sorteio reproduzível → coleta a `Crawl-delay: 10` (~37 h),
retomável → **fallback para o Internet Archive** em páginas que o site não
serve mais (1.050, 8%; 21% das de 2021–2026) → deduplicação (64) → lides com
legenda de foto corrigidos (372) → mesmos cortes temporais do lado falso.

**Fusão:** `texto_entrada` = viral limpo | título + lide; normalização da
assinatura da fonte nas duas classes ("hoje (12)" 3.647; "à Agência Brasil"
132); pareamento por tema × ano × split. Depois da extração v2, nenhuma
introdução do checador, versão colada ou entrada degenerada sobra para a fusão.

**Cobertura do pareamento:** 83% da demanda (13.146 de 15.845). Buraco
estrutural: Ciência e Tecnologia (16,9%) e Meio Ambiente (19,9%) — a agência
não tem essas editorias em boa parte da série. Decisão: não misturar licenças
restritivas; complemento previsto via tags editoriais da própria agência.

---

## 2. RQ1 — O refinamento por LLM altera o desempenho? (e cross-dataset)

### Medido — braço BRUTO

| experimento | resultado | fonte |
|---|---|---|
| TF-IDF + regressão logística, teste interno | **91,19%** acurácia balanceada | `modelo_bruto.py` |
| mesmo, sem registros com veredito | 91,47% (+0,28 pp) | idem |
| **avaliação cruzada no Fake.br (texto)** | **73,51%**, IC95% [72,54; 74,44] | `avaliacao_cruzada.py`, pré-registrado |
| queda interno → cruzado | 17,7 pp | idem |
| revocação FALSO / VERDADEIRO no Fake.br | 72,75% / 74,28% | idem |
| título × texto no Fake.br (mesmos 946 pares) | 67,18% × 70,45% (−3,3 pp, p = 0,003) | `exploratoria_cruzada.py` |

Pré-registro com previsões e leituras fixadas antes (commit 954a2d9); resultado
e confronto no adendo 1; testes exploratórios no adendo 2; teste 2 no adendo 3;
reexecuções no corpus v2 (adendo 4) e v3 (adendo 5). A leitura principal pré-registrada ("os 92%
são principalmente gênero") foi **refutada**: há transferência parcial.

### Falta — o coração da RQ1

1. **Braço REFINADO**: `ideia_central` com **prompt cego ao veredito**, igual
   nas duas classes e no Fake.br, gerado a partir do `texto_entrada` (a mesma
   entrada do braço bruto). O `ideia_central` atual das falsas **não serve**:
   foi gerado vendo o artigo inteiro do checador, com prompt que pedia "resuma
   a afirmação falsa".
2. **Braço combinado** (bruto + refinado).
3. **Mesma bateria nos dois braços**: interno, cruzado, controles.
4. **Pré-registro do braço refinado** antes de rodar (hipótese já registrada:
   a queda cruzada do refinado será menor que a do bruto).
5. Escada de modelos (seção 5) nos dois braços.

---

## 3. RQ2 — Extração ancorada na estrutura × extração por LLM

### Medido

| evidência | valor |
|---|---|
| literalidade do viral por método | `html` 99,4% · `texto` 100%* · `llm` 87,1% |
| tema: categoria editorial × LLM | concordância **48,7%**; LLM inflou "Saúde" de 7,7% para 47% |
| introduções do checador no viral | 1.071 de 1.075 "Leia:" vieram do caminho LLM |
| alegação dentro da evidência (defeito B) | caminho LLM 87,0% → 3,0% após correção |
| artigos-lista | 104 grupos com o mesmo viral para alegações diferentes (LLM) |
| **contraponto**: extração determinística também falha | `og:description` com legenda + crédito de foto em 372 lides (2014–2019); texto 1.354 com defeito A antes da correção |

\* tautológica: o trecho foi obtido procurando texto dentro do artigo.

Lição declarável: **não peça ao modelo o que a fonte informa de forma
determinística** — e audite a extração determinística também, porque ela
falha de outro jeito (silenciosamente, por mudança de marcação).

### Falta

1. **Experimento pareado** (item 9 da fila original): rodar o caminho LLM em
   300–500 artigos que JÁ têm extração determinística e comparar item a item
   no mesmo conjunto. Os números acima são sugestivos, não conclusivos: os dois
   caminhos foram aplicados a conjuntos disjuntos. Custo: ~1 h de GPU.
2. **Verificação manual por amostra**, com dois anotadores e concordância
   (kappa de Cohen). Sugestão: 200 registros estratificados por `origem_viral`.

---

## 4. RQ3 — As explicações dos modelos correspondem ao raciocínio dos checadores?

### Medido

Só a matéria-prima: `justificativa` (99,7%) e `justificativa_bruta` (99,2%),
com contaminação pela alegação reduzida de 87% para 3% no caminho LLM.

### Falta — nada rodado ainda

Três usos previstos (ver `README.md` §3.5):
1. geração da explicação a partir da alegação, avaliada com ROUGE/BERTScore;
2. verificação baseada em evidência (exige evidência recuperada para as DUAS
   classes — hoje só o lado falso tem);
3. comparar explicações do classificador (LIME/SHAP/atenção) com a
   justificativa do checador.

**Decisão para o orientador:** a RQ3 é a mais cara e a menos adiantada. O
item 3 é o mais barato e conversa com o que já foi feito (os pesos do modelo
já são uma explicação). Opções: reduzir a RQ3 ao item 3, ou deixá-la como
trabalho futuro com o corpus pronto para ela.

---

## 5. RQ4 — Que atalhos o corpus introduz e quanto da acurácia eles explicam?

### Medido

| atalho | quanto explica | como |
|---|---|---|
| comprimento, entrada título + lide | nada (47,7%, acaso) | baseline trivial |
| comprimento, corpo inteiro | **83,5%** sozinho | baseline trivial — por isso o recorte título + lide |
| exclamação sozinha | 68,3% | baseline trivial |
| maiúsculas sozinhas | 63,8% | baseline trivial |
| vazamento de veredito na entrada | nenhum na base v3 (91,19 → 91,47 sem os 121 marcados); título com "#boato" (7,2% das falsas, quase todo no treino) removido | com × sem `entrada_com_veredito` |
| assinatura da fonte ("hoje (12)", "Agência Brasil") | ~0,2 pp (92,24 → 92,05, medido na base v1) | antes × depois da normalização |
| nomes políticos (Lula, Dilma, PT) | ~2,0 pp no cruzado, IC95% [1,51; 2,60] | M0 × M1, McNemar |
| gênero textual + fonte (proxy) | ~17,7 pp | interno × cruzado |
| **forma sozinha (17 atributos, nenhuma palavra)** | **87,21% interno** (palavras: 91,19%) | `modelo_estilo.py`, adendos 3–5 |
| **forma sozinha, cruzado** | **65,18% no Fake.br** (palavras: 73,51%) → 15,2 dos 23,5 pp acima do acaso (≈ 2/3); léxico ~8 pp | idem |
| vazamento no pipeline | nenhum | rótulos embaralhados: 51,38 / 51,07 / 48,14% |
| sobreposição entre partes | zero no pareado; 2 boatos reciclados com o mesmo início | `controles_pipeline.py` |

Os baselines triviais foram medidos ANTES da limpeza e da normalização —
precisam ser refeitos no corpus final.

### Falta — outros testes (seção 7)

---

## 6. Custos de GPU

Base do cálculo: 8,6 s por artigo **medido** na extração das falsas. Esse
tempo é dominado pela **geração** (~300 tokens de saída: viral copiado,
alegação, palavras-chave, justificativa), o que dá ~40 tokens/s de saída na
RTX 4060 com qwen3.5:9b. O refinamento cego gera ~90 tokens (uma frase + 5–10
palavras-chave + JSON) a partir de uma entrada curta (~300 caracteres).
Estimativa: **~2,5 s por texto (faixa 2–4 s)**. Confirmar cronometrando 50
textos no piloto antes de soltar a fila.

| etapa | itens | tempo (faixa) |
|---|---:|---:|
| reprocessar defeitos do lado falso (extração v2) — em andamento | 2.633 artigos | ~6,3 h na RTX 4060 |
| piloto do prompt cego (3 gêneros, 100 de cada) | 300 | ~15 min |
| **refinamento cego — corpus** (elegíveis, as duas classes) | ~28.000 | **~19 h (16–31 h)** |
| refinamento cego — Fake.br (avaliação cruzada) | 7.200 | ~5 h (4–8 h) |
| experimento pareado da RQ2 | 300–500 | ~1 h |
| **total** | | **~30 h (25–46 h)** |

O refinamento das verdadeiras no pipeline original (`main_refinar_verdadeiras`)
lia até 3.800 caracteres do artigo inteiro e pedia "resuma o fato relatado"
— entrada e prompt diferentes dos do lado falso. **Não deve ser reaproveitado**:
os dois lados precisam do mesmo prompt sobre o mesmo tipo de entrada.

Risco a medir depois do refinamento: mesmo cego, o LLM pode inferir que uma
alegação é implausível e escrever com distanciamento ("supostamente",
"teria"). Contar essas marcas por classe no `ideia_central` — se diferirem, é
um atalho novo introduzido pelo refinamento (vira resultado de RQ1/RQ4).

---

## 7. Outros testes propostos para a base

Em ordem de custo × valor. Todos rodam nos dois braços quando o refinado existir.

| # | teste | pergunta | custo |
|---|---|---|---|
| 1 | **refazer os baselines triviais** no corpus final | os números da RQ4 ainda valem depois da limpeza? | segundos |
| 2 | ✅ **modelo só de estilo** — FEITO (adendo 3) | forma explica 87,21% interno e ~2/3 da transferência ao Fake.br | minutos |
| 3 | **o que decide no Fake.br**: termos que mais contribuem para os acertos cruzados | caracterizar o sinal que transfere | minutos |
| 4 | **TF-IDF de caracteres** (sem minúsculas) | quanto caixa alta, pontuação e emojis acrescentam às palavras | minutos |
| 5 | **deixar um tema de fora** (treina sem o tema X, testa em X) | o modelo depende do assunto? | minutos |
| 6 | **acurácia por ano no teste** | há deriva temporal (boatos novos mais difíceis)? | segundos |
| 7 | **subconjunto pareado por assunto** (verdadeiras escolhidas pela proximidade do título na URL, sem recoletar) | quanto dos 92% vem do assunto dentro do tema | ~1 h de código, sem GPU |
| 8 | **ruído de rótulo no lado verdadeiro**: verdadeiras que noticiam desmentidos ("é falso que…") | a classe VERDADEIRO contém a alegação falsa? | minutos |
| 9 | **BERTimbau** ajustado | o modelo de referência da literatura em português aprende mais atalho ou mais sinal? | ~1–2 h de GPU |
| 10 | **LLM sem treino** (zero-shot) | um modelo que não viu o treino transfere melhor? | ~5–10 h de GPU |
| 11 | anotação manual (RQ2) e de 100 casos de `suporte` | validade das extrações e do campo `suporte` | horas humanas |

---

## 8. Ordem de execução

1. Reprocessar os defeitos do lado falso (~5 h) → `corrigir_corpus` →
   `preparar_verdadeiras` → `fundir_corpus`.
2. Rodar de novo: `modelo_bruto`, `controles_pipeline`, `avaliacao_cruzada`,
   `exploratoria_cruzada`; testes 1–6 da seção 7. Registrar como adendo.
3. Piloto do prompt cego nos três gêneros (corrente, agência, Fake.br).
4. **Pré-registro do braço refinado.**
5. Refinamento cego: corpus + Fake.br (~24 h).
6. Comparativo definitivo bruto × refinado × combinado, interno e cruzado,
   com a escada de modelos.
7. RQ2 pareado + anotação manual. RQ3 conforme decisão com o orientador.

## 9. Para a escrita

- **Capítulos que já podem ser escritos:** construção do corpus (seção 1),
  RQ2 (parcial), auditoria da RQ4 (parcial), método de avaliação
  (pré-registro, controles, testes de hipótese).
- **Datasheet** (Gebru et al.): este arquivo + `README.md` +
  o pré-registro cobrem quase todas as seções.
- **Definição operacional** a declarar: "instância falsa = mensagem viral
  transcrita por checador profissional, com veredito falso para aquele item";
  contrastar com o Fake.br (falsa = matéria de site de desinformação).
- **Relatar sempre tamanho de efeito com IC**, não só p-valor.
- **Ambiente:** rodar os scripts pelo Python do WSL (`.venv/bin/python`). No
  Windows, o Controle Inteligente de Aplicativos bloqueia DLLs do
  scikit-learn (modelos de árvore) — não desligar: não volta sem reset.
- **Publicação:** o comparativo bruto × refinado com avaliação cruzada
  pré-registrada, somado ao corpus e à auditoria de atalhos, tem forma de
  artigo. Veículos naturais para PLN em português: PROPOR, STIL, BRACIS
  (confirmar prazos). O corpus pode sair como artigo de recurso (dataset paper).

---

## 10. Contribuição: os dois planos da detecção (forma × conteúdo)

> Ideia do pesquisador, discutida em 29/09/2026 depois do teste 2. Guardar para
> a discussão, as conclusões e os trabalhos futuros do TCC.

### A formulação

> Um detector de desinformação em português pode ser decomposto em dois planos:
> **forma** (estilo típico de corrente — forte, mas superficial e burlável) e
> **conteúdo** (proximidade com alegações já verificadas — mais fraco quando
> aprendido estatisticamente, mas verificável quando ancorado em checagens).
> O corpus permite medir cada plano separadamente e oferece a base de
> alegações checadas, com justificativa, para ancorar o segundo.

### O que já sustenta isso (medido)

- Forma sozinha: 87,21% interno contra 91,19% com palavras — dentro da base,
  a forma responde por quase toda a separação.
- Fora da base (Fake.br): forma 15,2 dos 23,5 pp acima do acaso (≈ 2/3); léxico ~8 pp.
- O "sotaque" das falsas aponta na mesma direção nas duas bases, mas é muito
  mais fraco nos blogs do Fake.br (maiúsculas 5,8% × 5,0% lá; 13,7% × 4,5% aqui).
- "você" inverte de direção no Fake.br: marca de WhatsApp, não de desinformação.

### Cuidados que o texto precisa declarar

**Plano da forma** — detecta *estilo sensacionalista*, não falsidade.
- Alarme falso: alerta legítimo em caixa alta ("ATENÇÃO!!!" da Defesa Civil)
  tem forma de corrente.
- Burlável: boato reescrito como matéria (ou por um LLM) passa. As falsas do
  Fake.br já mostram estilo contido.
- Uso adequado: indicador ("forma típica de corrente"), nunca veredito.

**Plano do conteúdo aprendido estatisticamente** — os ~7 pp vêm de assuntos
frequentes em boatos ("ivermectina", "fraude", "urna"), não de verificação.
- Envelhece: os boatos mudam de assunto (deriva temporal).
- Risco político: um filtro por assunto tende a atingir temas e grupos, não
  falsidades. O teste T2 mediu só os nomes (~1,3 pp); vocabulário político
  mais amplo não foi medido.

### Onde o plano do conteúdo vira verificação: recuperação de alegações checadas

O corpus tem ~15,8 mil alegações falsas, cada uma com a justificativa do
checador e o link da checagem. Para um texto novo, recuperar a alegação checada
mais parecida e mostrar a evidência:

> "Alegação semelhante já verificada como falsa pelo Boatos.org em 2023.
> Motivo: …"

- Tarefa estabelecida na literatura (*claim matching* / recuperação de
  alegações já checadas).
- Explicável: mostra a evidência do checador, não só uma probabilidade.
- Conversa direto com a RQ3 (justificativa como explicação de referência).

### Escopo

- **No TCC:** a decomposição (medida) e o braço refinado (que isola o
  conteúdo) como resultados; o filtro em dois planos e a recuperação de
  alegações como proposta e trabalho futuro. No máximo, um protótipo pequeno
  da recuperação, se houver tempo — aproveitaria a RQ3.
- **Depois:** o protótipo de recuperação é extensão natural para um segundo
  artigo, após a publicação do corpus.

### Literatura a confirmar e citar

- Potthast et al. (2018) — estudo estilométrico de notícias hiperpartidárias
  e falsas: o estilo separa o hiperpartidário, mas não o falso. Paralelo
  direto com o resultado do teste 2.
- Shaar et al. (2020), "That is a Known Lie" — recuperação de alegações já
  checadas.
- Conferir referências exatas no levantamento bibliográfico.
