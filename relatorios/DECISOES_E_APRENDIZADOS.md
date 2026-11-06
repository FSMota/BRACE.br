# Decisões e aprendizados — registro de método

> Consolida o que antes estava espalhado em `MUDANCAS.md` (agosto), no documento
> de passagem de 22/09/2026 e nas decisões tomadas até 29/09/2026. Escrito para o
> capítulo de método do TCC: cada decisão com o motivo e, quando existe, o número
> que a justificou. Resultados de experimento ficam em `RESULTADOS_E_PLANO.md`;
> definições de campo, em `README.md`.

---

## 1. Regras que não podem ser quebradas

Cada uma nasceu de um erro medido. Violar qualquer uma invalida resultados.

| regra | por quê |
|---|---|
| **`texto_bruto_jornalista` nunca é entrada de treino** | é o artigo do checador ("é falso", "boato", "checamos"); o classificador aprenderia a reconhecer fact-check |
| **`justificativa` nunca é entrada para veracidade** | só existe no lado falso: a presença do campo prediz a classe |
| **`tema_pareamento` vem de rótulo editorial HUMANO nos dois lados** | se um lado fosse rotulado por LLM e o outro por editor, o método de rotulagem viraria atalho. Ressalva medida: 22,1% dos baldes do lado falso vêm de tema por LLM (declarado) |
| **o mesmo prompt, cego ao veredito, nos dois lados** | o LLM que sabe o rótulo escreve diferente ("teria", "supostamente") |
| **nunca rodar `main_scraper()` e `main_nlp()` ao mesmo tempo** | os dois reescrevem o `dataset_bruto.json` |
| **uma instância só do coletor contra a Agência Brasil** | duas somadas quebram o `Crawl-delay: 10` declarado (o coletor tem trava) |
| **dividir treino/teste por tempo, nunca aleatoriamente** | o mesmo boato reciclado cairia dos dois lados |
| **marcar, não apagar** | apagar é decisão irreversível tomada por script; marcar deixa o filtro no experimento e permite medir quanto ele muda o resultado |

---

## 2. Lado FALSO — fontes e extração

### 2.1 Sondagem das fontes antes de escrever o extrator
`scripts/sondas/sondar_estrutura_fontes.py`, 12 artigos por fonte
(`relatorios/coleta/sonda_estrutura.json`):

| fonte | achado | decisão |
|---|---|---|
| Boatos.org | 12/12 artigos com exatamente 1 `<blockquote>`, e em 12/12 era a mensagem viral | extrair o viral pela marcação |
| G1 Fato ou Fake | 42 blockquotes em 10 artigos, **todos** falas oficiais, laudos ou declarações checadas — nunca o conteúdo falso | **não** usar o blockquote: preencheria o viral com o desmentido |
| Lupa | mudou de domínio (`lupa.uol.com.br` → `agencialupa.org`) | seletor corrigido |
| Aos Fatos | feed renderizado por JavaScript | desativada (exigiria navegador headless) |

Resultado: 99,6% do corpus falso é Boatos.org (15.789 de 15.846) — concentração
de fonte declarada como limite.

### 2.2 Três vias de extração do viral
- **html** (`<blockquote>`): literal por construção. 3.115 registros, 99,4% literais.
- **texto** (marcador "Leia a mensagem:" … "Checagem"): 1.354; recupera parte do
  arquivo antigo do Boatos.org, que trocou de template e perdeu o blockquote
  (cobertura cai de 66% para 9% no arquivo profundo).
- **llm**: o modelo localiza o trecho. 11.377; 87,1% literais.

**O rendimento motivou a via estrutural:** antes dela, só 6 de 128 artigos (5%)
geravam registro pela via do LLM — o modelo não achava a mensagem no meio da
prosa, inclusive em artigos que a sondagem confirmou terem o blockquote.

### 2.3 Não peça ao modelo o que a fonte informa
Pedir o **tema** ao LLM inflou "Saúde" de 7,7% para 47% do corpus (ele marcava
qualquer boato que citasse morte ou corpo). Ler a categoria da URL resolveu sem
custo. Concordância medida entre categoria editorial e LLM: **48,7%**.

### 2.4 Filtro de idioma antes do LLM
O Boatos.org republica cada boato em espanhol e inglês: 29% do acervo inicial
era não-PT e gastava uma chamada de LLM para ser descartado. O padrão de URL
(`/espanol|english/`) passou a filtrar antes do modelo.

### 2.5 Correções do corpus (`corrigir_corpus.py`)
- **Defeito A** — o marcador textual sem âncora de fim levava a apuração junto
  (95,7% com veredito dentro). 2.675 registros devolvidos ao LLM → 0.
- **Defeito B** — a alegação dentro da evidência (`justificativa_bruta`): 87,0%
  no caminho LLM → 3,0%.
- Campos derivados: `data_publicacao` (100%), `suporte` (20,1% declarado),
  `tema_pareamento`.
- MinHash/LSH, Jaccard ≥ 0,70: 353 grupos, 747 excedentes (4,7%); divisão
  temporal 70/10/20 com o grupo inteiro do mesmo lado.

### 2.6 Extração v2 (29/09/2026)
A fusão revelou defeitos nas **três** vias — não só na do LLM:

| defeito | html | texto | llm |
|---|---:|---:|---:|
| degenerado (lixo) | 262 | 0 | 46 |
| "Versão N" no meio | 158 | 365 | 572 |
| introdução do checador ("Leia:", "Versão 1:"...) | 275 | 469 | 1.671 |
| mesmo viral para alegações diferentes | 192 | 24 | 70 |

Correção: `scripts/limpeza_viral.py` (regras compartilhadas pela extração e pela
fusão), prompt novo na via LLM, um bloco = um registro, fallback para o LLM
quando todos os blocos são lixo. **Instância = alegação checada:** de várias
versões do mesmo boato, fica a primeira. 2.625 artigos reprocessados
(`scripts/marcar_reprocessamento.py`); registros novos com
`extracao_versao = "v2-2026-09"`.

**Lição para a RQ2:** a extração determinística também falha — silenciosamente,
por mudança de marcação (embed do Instagram, links do site dentro do blockquote).
Auditar as duas vias, não só a do LLM.

---

## 3. Lado VERDADEIRO — rota final e becos sem saída

### 3.1 Princípio: descoberta ≠ admissão
Um índice de notícias (GDELT) não avalia credibilidade. O rótulo VERDADEIRO
precisa se apoiar na linha editorial de um veículo — a mesma estratégia do
Fake.br (G1, Folha, Estadão).

### 3.2 Por que não gerar notícia verdadeira sintética
"Verdadeiro" significa correspondência com fatos; texto gerado não tem valor de
verdade, só não está marcado como falso. E texto de LLM é distinguível de texto
humano: o classificador aprenderia "escrito pelo modelo → verdadeiro".

### 3.3 Becos sem saída (não repetir)
- **GDELT DOC 2.0** cobre só uma janela deslizante de **3 meses** (documentação
  oficial: *"It must be within the last 3 months"*). Pareamento por ano via DOC é
  impossível. Alternativa não explorada: GKG via BigQuery (URLs, não texto).
- **Checagens com veredito VERDADEIRO não existem em escala**: 9 em 12.708
  títulos do Boatos.org, quase todos "parcialmente verdadeiro"; sonda em G1,
  Lupa, AFP, Comprova, Estadão e Aos Fatos sem volume
  (`scripts/sondas/sondar_verdadeiras.py`).
- **Outras agências públicas**: Senado (sitemap 403), Câmara (sem sitemap),
  Agência Gov (quase só imagens).
- **Folha, Estadão, O Globo**: paywall.

### 3.4 Licença decide a fonte
Redistribuir texto integral de notícia protegida num corpus público não se
sustenta. **Agência Brasil (CC BY 3.0 BR).** Veículos CC BY-NC-**ND** (Jornal da
USP, Agência FAPESP) foram descartados: proíbem derivadas e criariam dois
regimes de distribuição no mesmo corpus.

### 3.5 Descoberta de URLs
O sitemap da Agência Brasil é legado e **congela em 2020** (150.641 URLs).
2021–2026 via **CDX do Internet Archive** (84 consultas, 70.937 URLs) em vez de
paginar o site (4.176 requisições, instável em profundidade).

### 3.6 Coleta (`coletar_verdadeiras.py`)
- Cota por (tema, ano) = demanda do lado falso; sorteio reproduzível por célula;
  retomada que desconta o já coletado; reposição de falhas; trava de instância.
- **Fallback para o Internet Archive:** 21% das matérias de 2021–2026 (0% das
  anteriores) devolvem "Página temporariamente indisponível" com HTTP 200, de
  forma persistente. Repor por outra URL trocaria a amostra aleatória por "o que
  o site ainda serve". 1.050 textos (8%) vieram do Archive, marcados em `texto_via`.
- Resultado: **13.146 = 83% da demanda.** Buraco estrutural em Ciência e
  Tecnologia (16,9%) e Meio Ambiente (19,9%). Complemento previsto: tags
  editoriais da própria agência (mesma fonte, mesma licença).

### 3.7 Preparação (`preparar_verdadeiras.py`)
- Duplicatas pelo **corpo inteiro** (Jaccard ≥ 0,70, o mesmo critério do lado
  falso): 64 excedentes — 58 idênticas (mesma matéria, URL nova após correção
  de título) e 12 boletins de texto-modelo.
- **365 lides** de 2014–2019 traziam legenda + crédito da foto grudados
  (`og:description`); corrigidos, original em `lide_original`.
- Split com os **mesmos cortes temporais** do lado falso.

---

## 4. Fusão e entrada do experimento (`fundir_corpus.py`)

- `texto_entrada` = viral limpo | **título + lide**, unidos por um espaço (um
  separador especial só existiria de um lado). O corpo inteiro separa as classes
  só pelo comprimento (83,5%); título + lide neutraliza (47,7%).
- **Normalização da assinatura da fonte, nas duas classes:** "hoje (12)" → "hoje";
  menções "à Agência Brasil" removidas. Efeito medido: ~0,2 pp.
- **Maiúsculas e pontuação NÃO são tocadas** (decisão do pesquisador): caixa alta
  e "!!!" são o estilo do boato — apagá-los perderia informação legítima.
- **Vazamento de veredito marcado, não excluído:** regra de texto não separa
  "É falso que…" (checador) de "FAKE NEWS É DESMENTIDA…" (o próprio boato).
  Efeito medido: ~0,2 pp.
- Pareamento por **tema × ano × split**: 12.317 + 12.317.

---

## 5. Armadilhas de medição (padrão a repetir)

Cada uma quase produziu um número falso. O que as pegou foi sempre **medir sobre
a base inteira antes de concluir**.

- **Amostra não representativa.** Cobertura do recorte de justificativa: 84% numa
  amostra de 51, 37,8% sobre os 12.708 — a fila estava ordenada e o começo usava
  um template mais fácil.
- **Bug que o log não reporta.** Metade do corpus passava por um caminho sem o
  campo `justificativa` no schema; o log dizia "51 gravadas" e estava certo.
- **Duplicatas da CDX.** `collapse=urlkey` só funde linhas adjacentes e a chave
  inclui query string: sem normalizar, a oferta contava o dobro.
- **Imagens no índice.** Sitemap e CDX indexam a foto da matéria no mesmo caminho.
- **Tautologia na validação.** Literalidade de 100% no método `texto` é
  consequência de o trecho ter sido obtido procurando texto no artigo.
- **Taxonomia instável no tempo.** A Agência Brasil criou `meio-ambiente` e
  `cultura` por volta de 2024; antes, essas matérias viviam em `geral`.
- **Significância não é tamanho.** Com 7.200 textos, apagar nomes políticos deu
  p = 5·10⁻⁵ e efeito de 1,3 pp. Relatar sempre efeito com intervalo.
- **Previsão escrita antes.** A leitura "os 92% são gênero" parecia óbvia e foi
  refutada (adendo 1). Só apareceu porque a previsão estava registrada.

---

## 6. Métodos de avaliação adotados

- **Pré-registro** com desenho, previsões e leituras fixados e commitados antes
  de rodar (`PREREGISTRO_AVALIACAO_CRUZADA.md`); resultados em adendos datados.
- **Controles feitos para falhar:** rótulos embaralhados (~50%) e sobreposição
  entre partes (zero no pareado) — `scripts/controles_pipeline.py`.
- **Testes de hipótese** com correção de Holm e intervalo por bootstrap
  (`scripts/exploratoria_cruzada.py`), relatados como exploratórios.
- **Avaliação cruzada** no Fake.br sem nenhum ajuste nele.
- **Ambiente:** scripts rodam pelo Python do WSL. No Windows, o Controle
  Inteligente de Aplicativos bloqueia DLLs do scikit-learn — não desligar (não
  volta sem reset do sistema).
- Throughput medido na RTX 4060 com qwen3.5:9b: ~43 tokens/s de geração;
  8,6 s por artigo na extração completa.

## 7. Revisão dos rótulos da classe falsa (05/11/2026, versão 1.1.0)

Exploratório e posterior aos experimentos, a partir de uma dúvida na revisão do
TCC: como se sabe que cada entrada da classe falsa é falsa?

- **O problema.** O rótulo FALSO é herdado do artigo de checagem
  (`"veredito": "FALSO"` em `web_scraping_limpo.py`); nenhum campo trazia o
  veredito de cada mensagem.
- **Primeira tentativa, abandonada.** Uma varredura de títulos com revisão
  manual (`marcar_rotulo_duvidoso.py`) separou 104 entradas duvidosas, mas errou:
  casos tidos como verdadeiros ou como "não é checagem" tinham selo de "Fake
  news" ou "Exagerado". Não usar `rotulo_duvidoso.json` como referência.
- **O método final.** O Boatos.org fecha os artigos recentes com um selo de
  veredito, e os antigos têm #boato no título. `veredito_checador.py` lê essas
  marcas: 84% das falsas têm veredito explícito do checador, 15% têm palavra de
  desmentido no título (regra ampla, a parte menos segura) e 144 entradas, de 73
  artigos, não têm marca nenhuma.
- **A revisão.** O pesquisador abriu os artigos dessas 144 entradas e localizou o
  veredito no texto do checador (`aplicar_revisao_rotulos.py`). As 3 entradas do
  artigo do Neymar são alucinações, sem veredito a localizar, e já saíam da base.
- **Decisões com o orientador.** O veredito vem do checador, não do pesquisador.
  Meias-verdades (selos Enganoso, Exagerado, Real com erros; 67) ficam FALSO, e a
  base não se declara multiclasse. Mensagens verdadeiras ou em apuração (20) saem;
  conteúdo sem veredito (4) vira NAO_E_CHECAGEM, fora do pareado. Quando a planilha
  divergiu de um veredito explícito do checador (bebê do Ceará, "Anitta é bi",
  caso Gael), prevaleceu o checador.
- **Efeito.** Versão 1.1.0 (DOI 10.5281/zenodo.23982799): 27.871 instâncias,
  pareado 11.823 x 2. Os experimentos não foram refeitos e continuam na versão
  de 28.002. Retreinar o modelo de referência sem as entradas muda a acurácia em
  no máximo 0,40 ponto (`sensibilidade_revisao_rotulos.py`), variação que também
  aparece ao retreinar com mudanças mínimas, sem alterar nenhuma conclusão.
- **Limites.** A regra de título é ampla, e conteúdo patrocinado não tem marca no
  texto: há casos conhecidos que ficaram com o rótulo que o checador publicou.

**Versão 1.1.1 (06/11/2026, DOI 10.5281/zenodo.24019899).** Uma segunda leitura dos
pares entre as classes mostrou entradas da série Autoboataria com veredito "Fato" ou
"Inconclusivo" do checador que a revisão havia mantido como falsas. Saem essas 5 e os 6
erros da auditoria da via html: 27.860 instâncias, pareado 11.816 x 2. Os 43 pares
foram lidos pelo pesquisador e conferidos pelo assistente; nenhuma matéria é negada
pelo checador (11 do mesmo fato, 32 só do mesmo assunto). Retirar as 38 entradas da
revisão muda a acurácia em no máximo 0,30 ponto.
