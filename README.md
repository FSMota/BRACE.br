# BRACE.br — Boatos e Reportagens de Agências, Coletados em Escala

Base de desinformação em português, com boatos desmentidos por agências de
checagem (classe FALSO) pareados a reportagens de uma agência pública de
notícias, a Agência Brasil (classe VERDADEIRO), por tema e por ano. A base foi
construída automaticamente e auditada quanto aos seus limites de validade. Trabalho de Conclusão de Curso
(Ciência da Computação, Instituto de Computação, UFAL).

- **Autor:** Filipe Simões Mota
- **Orientadora:** Profa. Dra. Roberta Vilhena Vieira Lopes
- **Coorientador:** Prof. Me. Lucas Lisboa

---

## 1. Contexto

A detecção automática de desinformação depende de bases com exemplos
classificados, e em português elas são poucas e pequenas. O levantamento de
Baracho, Lisboa e Lopes (WICS 2025) mostrou que a maioria tem menos de 10 mil
instâncias, que elas se concentram em política e saúde, dependem de poucas
agências de checagem e são construídas em boa parte por seleção manual. Além
disso, modelos treinados nessas bases chegam a 99% de acurácia dentro delas e
caem para cerca de 71% em outra (Guedes et al., PROPOR 2026): aprendem
características da construção (atalhos), e não a diferença entre falso e
verdadeiro.

A BRACE.br responde aos dois problemas. Cada instância falsa é uma mensagem que
circulou e foi verificada e desmentida por um checador, com a justificativa
dele; cada verdadeira é uma matéria de agência pública de notícias com licença
aberta. As classes são pareadas por tema e por ano, a base é dividida no tempo,
e cada experimento foi pré-registrado, com as previsões escritas antes da
execução.

### Números

| | versão publicada (1.1.0) | versão dos experimentos (TCC) |
|---|---:|---:|
| total | **27.871** (14.721 falsas + 13.146 verdadeiras + 4 que não são checagem) | 28.002 (14.856 + 13.146) |
| subconjunto pareado (tema × ano × parte da divisão) | **23.646** (11.823 + 11.823) | 23.950 (11.975 + 11.975) |
| treino / validação / teste (pareado) | 16.544 / 2.068 / 5.034 | 16.707 / 2.080 / 5.162 |
| período | 2014 a 2026 | 2014 a 2026 |

A versão publicada retira as 111 entradas falsas inventadas pelo LLM na
extração e exclui do pareado as checagens de discurso, reequilibrando as
células afetadas. A partir da 1.1.0, cada falsa leva o veredito publicado pelo
próprio checador (`veredito_checador`: o selo do fim do artigo no Boatos.org ou
a hashtag #boato), lido por `scripts/veredito_checador.py`; nos artigos sem essa marca, o pesquisador localizou o veredito no texto do checador
(`scripts/aplicar_revisao_rotulos.py`), e as 20 mensagens verdadeiras ou ainda
em apuração saíram da base. O pareado exato dos experimentos vai no atributo
`subconjunto_pareado_tcc`; retirar as entradas inventadas altera a acurácia do
modelo de referência em no máximo 0,15 ponto.

Comparação com as maiores bases do levantamento:

| base | instâncias | observação |
|---|---:|---|
| Base COVID-19 | 13.087 | tema único; 21% falsas |
| FakeRecogna | 11.902 | balanceada |
| Fake.br | 7.200 | falsas de *sites* de desinformação |
| **BRACE.br** | **27.871** | veredito por alegação e justificativa do checador |

### Perguntas de pesquisa

- **Geral:** É viável construir automaticamente, a partir de portais de
  checagem, um corpus de larga escala em português para detecção de
  desinformação, que sustente tanto a comparação entre representações textuais
  quanto a avaliação de explicabilidade? Quais são os limites de validade dessa
  construção?
- **RQ1 · Validade:** Que atalhos a construção automática da base introduz, e
  quanto da acurácia dos modelos eles explicam?
- **RQ2 · Método de extração:** A extração ancorada na estrutura da página
  supera a extração por LLM na construção da base?
- **RQ3 · Explicabilidade:** As explicações dos modelos correspondem ao
  raciocínio dos checadores? A base permite recuperar a checagem já feita de um
  boato novo?
- **RQ4 · Representação:** O refinamento do texto por LLM altera o desempenho
  dos modelos em relação ao texto bruto, e o efeito se mantém em outra base?

Respostas e números em [`PERGUNTAS_DE_PESQUISA.md`](PERGUNTAS_DE_PESQUISA.md) e
em [`relatorios/RESULTADOS_E_PLANO.md`](relatorios/RESULTADOS_E_PLANO.md);
decisões de método em
[`relatorios/DECISOES_E_APRENDIZADOS.md`](relatorios/DECISOES_E_APRENDIZADOS.md).

---

## 2. Uma base, várias versões do texto

Cada instância tem uma versão **bruta** e uma versão **refinada** do mesmo
conteúdo. Não são duas bases: são colunas diferentes do
`corpus_fundido.jsonl`.

| versão | entrada FALSO | entrada VERDADEIRO |
|---|---|---|
| texto bruto | `texto_entrada` = mensagem viral limpa | `texto_entrada` = título + lide |
| texto refinado | `ideia_central` | `ideia_central` |
| combinada | bruto + refinado | bruto + refinado |

O `ideia_central` é uma frase neutra com a alegação central, gerada por **um
único prompt, cego à classe** (`cego-v2`), aplicado às duas classes e ao
Fake.br. A `justificativa` não entra em nenhuma versão: serve à
explicabilidade, e usá-la como entrada vazaria o rótulo.

---

## 3. Método

### 3.1 De onde vem cada classe

**FALSO — portais de checagem.** Boatos.org (99,6%), Lupa e G1 Fato ou Fake.
O trabalho de checadores profissionais é a fonte do rótulo, item a item.
Definição operacional: mensagem viral transcrita por um checador, com veredito
falso emitido para aquele item. Contraste com o Fake.br, em que a falsa é a
matéria publicada por um *site* de desinformação.

**VERDADEIRO — Agência Brasil (CC BY 3.0 BR).** A escolha é jurídica antes de
ser editorial: é preciso redistribuir o texto num corpus público. O rótulo se
apoia na linha editorial de uma agência pública de notícias.

### 3.2 Etapas

```
lado falso       web_scraping_limpo.py (main_scraper -> main_nlp)   painel_pipeline.ipynb
                 corrigir_corpus.py                                  campos, duplicatas, divisão temporal
lado verdadeiro  descobrir_cdx.py -> coletar_verdadeiras.py -> preparar_verdadeiras.py
fusão            fundir_corpus.py                                    corpus_fundido.jsonl
refinamento      refinar_cego.py -> juntar_refinamento.py            ideia_central (prompt cego-v2)
experimentos     um script por experimento, cada um com pré-registro (seção 6)
```

### 3.3 O que o LLM faz e o que ele não faz

Princípio: **o LLM só produz o que exige interpretação de linguagem.** O que a
fonte informa de forma determinística é lido diretamente.

| informação | quem produz | por quê |
|---|---|---|
| URL, fonte, título, veredito, data | Python | a raspagem já sabe |
| tema | categoria da URL; LLM só quando ela falta | pedir o tema ao LLM inflava a categoria Saúde |
| mensagem viral | marcação HTML (`html`) ou marcador textual (`texto`); LLM (`llm`) no restante | a marcação é literal por construção |
| idioma | LLM, com filtro de URL antes | exige ler o texto |
| frase refinada, palavras-chave | LLM | geração e seleção semântica |
| justificativa | LLM sobre o recorte da apuração | destilação do texto do checador |

Vias de extração nas falsas: `llm` 11.113, `html` 2.422, `texto` 1.321. No
mesmo artigo, a via estrutural é mais literal (100% contra 97,2% do LLM), mas
a leitura humana mostrou que ela é a que mais captura trechos que não são a
mensagem; a via LLM, por sua vez, inventou cerca de 1,2% das entradas, que
estão marcadas e saem da versão publicada.

---

## 4. Dicionário de campos

> **origem** diz quem produziu o valor; **uso** diz se o campo pode ser entrada
> de treino.

### 4.1 Identificação e procedência

- **`veredito`** — `FALSO` ou `VERDADEIRO`. O rótulo-alvo. Origem: determinística.
- **`id`, `url`, `fonte`** (e `url_origem_fake`, `fonte_fake`) — rastreabilidade,
  deduplicação e controle de concentração de fonte. Nunca entrada de treino.
- **`titulo_artigo_checagem`** — título do artigo do checador ("É falso que…").
  Descreve o **desmentido**, não a alegação. Não é entrada de treino.
- **`data_publicacao`, `split`** — data do artigo de checagem (não a da
  circulação da mensagem; a distância mediana medida foi de 2 dias) e divisão
  temporal: treino até 25/07/2023, validação até 05/08/2024, teste a partir de
  06/08/2024. O lado verdadeiro usa os mesmos cortes.

### 4.2 Os textos

- **`texto_bruto_jornalista`** ⚠️ — o artigo do portal de checagem, isto é, o
  jornalista desmentindo o boato. Não é a mensagem falsa. Uso: rastreabilidade
  e verificação de literalidade. **Nunca entrada de treino.**
- **`texto_viral_bruto`** — a mensagem falsa como circulou, transcrita pelo
  checador, com erros, caixa alta e emojis preservados, já sem a introdução do
  checador e só com a primeira versão (`viral_prefixos_removidos`,
  `viral_versoes`).
- **`conteudo_real`, `titulo`, `lide`** (lado verdadeiro) — a matéria da Agência
  Brasil. O corpo inteiro não é entrada, porque o comprimento sozinho separaria
  as classes; a entrada é título + lide.
- **`texto_entrada`** — a entrada do texto bruto. Falso: mensagem viral limpa
  (`limpeza_viral.py`). Verdadeiro: título + lide. Nas duas classes, a
  assinatura da fonte é normalizada; maiúsculas e pontuação são preservadas.

### 4.3 Campos derivados

- **`tema`** — 12 categorias (Política, Saúde, Economia, Golpes e Fraudes,
  Segurança Pública, Ciência e Tecnologia, Meio Ambiente, Celebridades,
  Religião, Esportes, Internacional, Outros). Origem: categoria da URL; LLM
  quando ela é genérica.
- **`tema_llm`, `tema_origem`** — o que o LLM classificou e de onde veio o
  valor final (`url` ou `llm`).
- **`tema_pareamento`, `tema_pareamento_origem`** — camada mais grossa
  (`taxonomia.py`), comum às duas classes, usada só para o pareamento.
- **`idioma`** — só pt-BR entra na base.
- **`ideia_central`** — o texto refinado (seção 2).
- **`palavras_chave`** — 5 a 10 termos escolhidos por relevância pelo LLM.

### 4.4 Metadados de qualidade

- **`viral_verificado`, `verificacao_viral`** — o texto extraído aparece no
  artigo? `conferido`, `falhou` ou `inconclusivo_fonte_truncada`. É uma
  verificação criada para este pipeline; descreva o procedimento ao citar.
- **`origem_viral`** — via de extração: `html`, `texto` ou `llm`.
- **`blocos_no_artigo`** — mais de 1 indica artigo com várias mensagens.
- **`modelo_refinamento`** — modelo que produziu os campos (qwen3.5:9b, local).
- **Marcações** — `duplicata_canonica`, `grupo_duplicata`, `entrada_degenerada`,
  `entrada_repetida`, `entrada_com_veredito`, `entrada_narrativa_checador`,
  `entrada_cabecalho_checador`, `subconjunto_pareado`; e, em
  `relatorios/marcacoes/`, `checagem_de_discurso`, `desmentido` e
  `entrada_inventada`. **Nada é apagado; os experimentos filtram.**
- **Lado verdadeiro** — `licenca`, `editoria`, `texto_via` (`site` ou
  `internet_archive`), `snapshot_archive`, `lide_origem`, `data_origem`.

### 4.5 Explicabilidade

- **`justificativa`** — em 2 a 4 frases, por que a alegação não se sustenta,
  segundo a apuração do checador: evidências, fontes, datas, origem de imagens.
  Precedentes: LIAR-PLUS (Alhindi et al., 2018) e PUBHEALTH (Kotonya e Toni,
  2020).
- **`justificativa_bruta`** — o recorte literal da apuração, quando delimitável.
- **`justificativa_origem`** — `recorte+llm` ou `llm`.

⚠️ **A justificativa não pode ser entrada do classificador de veracidade**: só
existe no lado FALSO, e a presença do campo prediz a classe. O pipeline remove
dela as palavras de veredito ("é falso", "boato", "não procede"…), como no
LIAR-PLUS. Usos legítimos:

1. **referência para avaliar explicações** — as palavras que pesam na decisão
   do modelo coincidem com as do checador? (RQ3, uso A);
2. **recuperação de checagens** — dado um boato novo, encontrar a checagem
   anterior da mesma alegação (RQ3, uso B);
3. **alvo de geração** — dada a alegação, gerar a explicação.

---

## 5. Estrutura do repositório

```
painel_pipeline.ipynb            painel do lado falso (raspagem e extração por LLM)
scripts/
  web_scraping_limpo.py          raspagem + extração (main_scraper, main_nlp)
  limpeza_viral.py               regras de limpeza da mensagem viral
  taxonomia.py                   tema -> tema_pareamento, nos dois lados
  corrigir_corpus.py             campos derivados, duplicatas, divisão temporal (falso)
  marcar_reprocessamento.py      devolve à fila os artigos com defeito de extração
  auditar_corpus.py              auditoria por tipo de defeito (somente leitura)
  descobrir_cdx.py               URLs 2021-2026 da Agência Brasil via Internet Archive
  coletar_verdadeiras.py         coleta do lado verdadeiro (retomável, intervalo de 10 s)
  preparar_verdadeiras.py        duplicatas, lides, divisão temporal (verdadeiro)
  fundir_corpus.py               corpus único + marcações + subconjunto pareado
  refinar_cego.py                refinamento cego por LLM (ideia_central)
  juntar_refinamento.py          junta o refinamento ao corpus
  modelo_bruto.py                modelo de referência: TF-IDF + regressão logística
  modelo_estilo.py               modelo só de forma
  controles_pipeline.py          rótulos embaralhados e sobreposição entre partes
  comparativo_bracos.py          texto bruto × refinado × combinado
  avaliacao_cruzada.py           BRACE.br -> Fake.br
  avaliacao_inversa.py           Fake.br -> BRACE.br
  exploratoria_cruzada.py        testes exploratórios da avaliação cruzada
  pareamento_datado.py           controle temporal (pareamento datado × não datado)
  analise_pareamento_datado.py   apuração do controle temporal
  experimento_rq2.py             extração estrutural × LLM no mesmo artigo
  alinhamento_racionais.py       RQ3, uso A
  recuperacao.py                 RQ3, uso B
  extrair_vetores.py             vetores BERTimbau e bge-m3 (GPU)
  ajustar_bertimbau.py           BERTimbau com ajuste fino (GPU)
  matriz_modelos.py              13 modelos × duas direções
  apurar_matriz.py               apuração da matriz
  auditoria_fidelidade.py        fidelidade do refinamento (C3)
  auditar_vazamento.py           busca de vazamento no pipeline
  anotacao_manual.py             anotação das 200 falsas
  marcar_checagem_discurso.py    marcação das checagens de discurso
  sensibilidade_discurso.py      resultados sem as checagens de discurso
  marcar_desmentido.py           marcação de desmentidos
  atraso_checagem.py             atraso entre a mensagem e a checagem
  alucinacao.py                  invenção na via LLM
  sensibilidade_alucinacao.py    resultados sem as entradas inventadas
  justificativa_lixo.py          pré-preenchimento da planilha de recuperação
  figuras_tcc.py                 figuras do TCC, a partir dos JSONs de resultado
  diagnosticos/                  diagnósticos (controles, sensibilidade ao C)
  sondas/                        sondagens da fase de escolha das fontes
relatorios/
  RESULTADOS_E_PLANO.md          documento central de resultados
  DECISOES_E_APRENDIZADOS.md     registro de método
  PREREGISTRO_*.md               pré-registros, com adendos datados
  resultado_*.json, */           saídas dos experimentos, planilhas de anotação
  marcacoes/                     instâncias marcadas (discurso, desmentido, invenção)
datasets/                        dados (fora do git)
```

## 6. Pré-registros

Cada experimento tem um pré-registro em `relatorios/PREREGISTRO_*.md`, com o
desenho, as previsões e as leituras possíveis, commitado **antes** da execução.
Os resultados e os desvios entram em adendos datados; o que já foi escrito não
é editado. A data dos commits prova a ordem entre previsão e resultado. Os
pré-registros mais antigos usam a numeração das perguntas da época: a atual
RQ1 (validade) era a RQ4, e a atual RQ4 (representação) era a RQ1.

## 7. Ordem de execução

```bash
# lado falso (exige Ollama com qwen3.5:9b)
python scripts/web_scraping_limpo.py          # main_scraper() + main_nlp()
python scripts/corrigir_corpus.py --aplicar

# lado verdadeiro
python scripts/descobrir_cdx.py
python scripts/coletar_verdadeiras.py         # ~37 h; retomável
python scripts/preparar_verdadeiras.py --aplicar

# corpus e refinamento
python scripts/fundir_corpus.py --aplicar
python scripts/refinar_cego.py
python scripts/juntar_refinamento.py

# experimentos (ver o pré-registro de cada um)
python scripts/modelo_bruto.py
python scripts/controles_pipeline.py
python scripts/avaliacao_cruzada.py
python scripts/matriz_modelos.py
python scripts/figuras_tcc.py
```

Os scripts que gravam dados simulam por padrão; `--aplicar` efetiva, com cópia
de segurança.

## 8. Ambiente

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

No Windows, rode pelo **WSL**: o Controle Inteligente de Aplicativos bloqueia
DLLs do scikit-learn. Não desligue a proteção, porque ela não volta sem
reinstalar o sistema. Os vetores BERTimbau e bge-m3 e o ajuste fino exigem GPU.

## 9. Dados e licenças

Os dados não vão para o git. A base está no Zenodo, versão 1.1.0:
**https://doi.org/10.5281/zenodo.23982799**. Este repositório fica com o
código, a documentação e os resultados; `scripts/exportar_publicacao.py` gera
a versão publicada a partir da base usada nos experimentos.

**Código:** licença MIT (ver `LICENSE`).

- **Lado verdadeiro:** Agência Brasil, CC BY 3.0 BR.
- **Lado falso:** transcrições feitas pelos portais de checagem, citadas para
  fins de pesquisa, com o link para cada artigo de checagem.
- **Fake.br:** usado só para avaliação; não é redistribuído.
