# -*- coding: utf-8 -*-
"""Pipeline do lado FALSO: raspagem dos portais de checagem e extração por LLM.

  1. main_scraper()  -> raspagem bruta das fontes, salva em dataset_bruto.json
  2. main_nlp()      -> extração por artigo (texto viral, alegação, palavras-chave,
                        justificativa), salva em dataset_tcc_fake_news_final.json

O resto do corpus fica em scripts próprios, nesta ordem:
  corrigir_corpus.py        -> campos derivados, duplicatas e divisão temporal do lado falso
  coletar_verdadeiras.py    -> lado VERDADEIRO (Agência Brasil)
  preparar_verdadeiras.py   -> duplicatas, lides e divisão temporal do lado verdadeiro
  fundir_corpus.py          -> corpus único, com `texto_entrada` e o subconjunto pareado
(A rota antiga do lado verdadeiro via GDELT foi abandonada: a API DOC só cobre
os últimos 3 meses. Está no histórico do git.)

QUAL CAMPO SERVE DE ENTRADA DE TREINO
-------------------------------------
    O campo de entrada é o `texto_entrada` do corpus_fundido.jsonl (viral limpo |
    título + lide). O `ideia_central` atual NÃO serve para o braço refinado: foi
    gerado vendo o artigo do checador e sabendo que a alegação é falsa. O braço
    refinado vai usar um prompt cego, igual nas duas classes.
    texto_bruto_jornalista -> NÃO É ENTRADA. Só rastreabilidade.

O texto_bruto_jornalista é o artigo do site de checagem, isto é, o jornalista
DESMENTINDO o boato -- ele existe para rastrear a origem do rótulo e para
conferir a literalidade do texto viral. Usá-lo como entrada de treino produz
vazamento de rótulo: o artigo contém "é falso que", "boato", "checamos", e o
classificador aprende a reconhecer texto de fact-check, não desinformação.

DIVISÃO DE RESPONSABILIDADE LLM vs. PYTHON
------------------------------------------
O LLM só faz o que exige interpretação de linguagem; toda a montagem do registro
é determinística, feita em Python. Concretamente, o modelo devolve apenas:
    tema, idioma          -> classificação, restrita a enum fechado no schema
    texto_viral_bruto     -> extração literal, conferida contra o artigo original
    claim_refinada        -> geração (resumo de uma frase)
    palavras_chave        -> geração (5-10 termos)
Tudo o mais (url, fonte, título, veredito, texto bruto, flags) o Python já sabe e
anexa sozinho.
"""
import os
import re
import json
import time
import sys
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limpeza_viral import limpar_viral                               # noqa: E402

# Marca os registros produzidos por esta versão da extração -- permite separar
# no corpus o que foi reprocessado com as regras novas (RQ2).
# v2-2026-09: limpeza_viral nas três vias (introdução, versões, lixo).
# v3-2026-10: + recusa a voz do checador (título com "#boato", assinatura,
#             rodapé, narração "diz a mensagem") -- 873 títulos no pareado.
VERSAO_EXTRACAO = "v3-2026-10"

# ==============================================================================
# --- 1. CONFIGURAÇÕES DO MOTOR LLM (OLLAMA LOCAL) ---
# ==============================================================================

OLLAMA_URL = "http://localhost:11434/api/chat"
# Lido de variável de ambiente pra facilitar alternar entre modelos nos testes
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")

# Modelos de raciocínio híbrido (qwen3/qwen3.5) gastam tokens "pensando" antes
# de responder. Para extração estruturada isso é overhead puro -- e chegou a
# devolver content vazio nos testes. Desligado por padrão; se o modelo em uso
# não aceitar o parâmetro, chamar_llm() detecta o 400 e desliga o envio sozinho.
ENVIAR_THINK_FALSE = True

print(f"[*] Motor NLP iniciado com Ollama local ({OLLAMA_MODEL}).")

# ==============================================================================
# --- 2. PARÂMETROS E FONTES ---
# ==============================================================================

# None = sem limite (varredura exaustiva do arquivo da fonte).
# Um número limita quantas matérias NOVAS trazer por execução.
LIMIT_LINKS_POR_FONTE = None
# Teto de páginas por fonte. É só trava de segurança: a varredura para sozinha ao
# detectar página repetida (fim real da paginação). O Boatos.org publica ~25 por
# página, então 2000 páginas cobrem um acervo de ~50 mil matérias.
MAX_PAGINAS = 2000
# Pausa entre downloads de artigo. O Boatos.org é um site independente pequeno, e
# uma varredura exaustiva são milhares de requisições -- vale não sobrecarregá-lo.
PAUSA_ENTRE_ARTIGOS = 0.3
# Grava o acervo bruto a cada N matérias baixadas. Sem isto, uma varredura de
# horas perderia tudo num Ctrl+C ou queda de rede: o salvamento só acontecia no fim.
SALVAR_A_CADA_ARTIGOS = 50
# Página em que a varredura começa. Deixe em 1 no uso normal; ajuste para retomar
# uma varredura longa interrompida sem repercorrer o início do arquivo (o
# varrer_site informa a última página visitada ao terminar).
PAGINA_INICIAL = 1

# Teto do texto do artigo guardado. Era 5000, mas isso truncava artigos longos
# antes do fim do trecho viral, e a verificação de literalidade não encontrava o
# texto -- falso negativo de medição, não falha real de extração (18 dos 19 casos
# checados). O LLM continua recebendo só os primeiros 3800 caracteres, então
# subir este teto não custa tokens; custa alguns MB em disco e devolve a
# capacidade de conferir. Vale para coletas NOVAS; o acervo antigo segue em 5000.
MAX_CHARS_ARTIGO = 15000

# Caminhos resolvidos a partir da localização DESTE arquivo, não do diretório de
# trabalho. Com o projeto reorganizado (script em scripts/, dados em datasets/),
# caminhos relativos simples apontariam para lugares diferentes conforme de onde
# o Python fosse chamado -- e o pipeline criaria um dataset vazio sem avisar.
_RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DADOS = os.path.join(_RAIZ_PROJETO, "datasets")
os.makedirs(PASTA_DADOS, exist_ok=True)

ARQUIVO_BRUTO = os.path.join(PASTA_DADOS, "dataset_bruto.json")
ARQUIVO_FINAL = os.path.join(PASTA_DADOS, "dataset_tcc_fake_news_final.json")


# Estes parâmetros controlam salvamento e amostragem para testes:
SALVAR_A_CADA = 10        # grava os JSONs a cada N artigos processados
LIMITE_ARTIGOS_NLP = None  # ex.: 5 -> processa só 5 artigos (teste rápido); None = todos

# Enums fechados: 'tema' vira `tema_pareamento` (taxonomia.py), que comanda o
# balanceamento com o lado verdadeiro. Tema livre ("Política", "politica",
# "Política/Economia") fragmentaria o mapa. O schema JSON restringe a geração
# a estes valores exatos.
TEMAS_VALIDOS = [
    "Política", "Saúde", "Economia", "Golpes e Fraudes", "Segurança Pública",
    "Ciência e Tecnologia", "Meio Ambiente", "Celebridades", "Religião",
    "Esportes", "Internacional", "Outros"
]
IDIOMAS_VALIDOS = ["pt-BR", "es-ES", "en-US", "outro"]

# Cada fonte ganhou três campos novos, todos vindos da sondagem empírica
#
#   seletor_viral  -- seletor CSS do bloco que contém a MENSAGEM VIRAL transcrita.
#                     Quando existe, o texto viral é obtido de forma determinística
#                     e o LLM não precisa mais gerá-lo (era o campo mais caro em
#                     tokens e o único com risco real de paráfrase).
#                     None = a fonte não marca o viral estruturalmente.
#   ignorar_url    -- regex de URLs a descartar já na varredura.
#   ativa          -- False mantém a fonte documentada sem tentar coletar.
#
# Achados da sondagem:
#   Boatos.org : 12/12 artigos com exatamente 1 <blockquote>, e em 12/12 ele era a
#                mensagem viral (nenhum caso de citação jornalística). Cobertura 100%.
#   G1         : 42 blockquotes em 10 artigos, TODOS declarações de assessoria,
#                especialistas ou falas de políticos sendo checadas -- nunca o
#                conteúdo falso. Usar o blockquote aqui contaminaria a base com o
#                oposto do rótulo, então seletor_viral fica None de propósito.
#   Lupa       : mudou de domínio (lupa.uol.com.br -> agencialupa.org) e de layout;
#                por isso nunca coletou nada. Seletor corrigido pela sondagem.
#   Aos Fatos  : a listagem não traz links de matéria no HTML servido (só nuvem de
#                tags), indicando feed renderizado por JavaScript -- requests+bs4
#                não alcançam. Desativada até migrarmos para navegador headless.
FONTES = [
    {
        "nome": "Boatos.org",
        "url_inicial": "https://www.boatos.org/",
        "seletor_link": "article h2 a",
        "seletor_viral": "blockquote",
        # O <blockquote> só existe nos artigos mais recentes: no arquivo profundo
        # a cobertura cai de 66% para 9%, porque o site trocou de template ao
        # longo dos anos. Mas 87% desses artigos ainda trazem a transcrição no
        # corpo, aberta por um marcador ("Leia a mensagem que circula online:") e
        # encerrada onde começa a análise ("Checagem", "Conclusão").
        # Este par de padrões recupera ~45% do que o blockquote não pega, e roda
        # sobre o texto JÁ SALVO -- não precisa rebaixar nada.
        "marcador_viral_inicio": (
            r"(?:confira|leia|veja)\s+(?:o|a|as|os|algumas)?[^.:;]{0,60}"
            r"(?:texto|mensage|conteúdo|conteudo|transcri|post|publicaç)[^.:;]{0,40}:\s*"),
        "marcador_viral_fim": r"\s(?:Checagem|Conclusão|Análise|Fontes?:)\b",
        # O site republica cada boato em espanhol e inglês: 29% do que foi
        # coletado era não-PT e ia parar no LLM só para ser descartado depois
        # pelo filtro de idioma.
        "ignorar_url": r"/(espanol|english)/",
        # O Boatos.org já categoriza cada artigo, e a categoria está na URL --
        # classificação feita por editor humano, consistente e de graça. Pedir o
        # tema ao LLM produzia distorção grave: ele marcava como "Saúde" qualquer
        # boato que mencionasse morte ou corpo (morte falsa de celebridade, preso
        # por estupro, apreensão de drogas), inflando Saúde para 47% dos registros
        # quando a categoria real é 7,7% do corpus. Como o tema comanda o
        # balanceamento com o lado verdadeiro, isso distorceria o corpus inteiro.
        # 'brasil' e 'lista' são genéricos e ficam de fora: nesses o LLM decide.
        "regex_categoria": r"boatos\.org/([^/]+)/",
        "mapa_tema": {
            "politica":       "Política",
            "saude":          "Saúde",
            "economia":       "Economia",
            "esporte":        "Esportes",
            "religiao":       "Religião",
            "tecnologia":     "Ciência e Tecnologia",
            "ciencia":        "Ciência e Tecnologia",
            "entretenimento": "Celebridades",
            "mundo":          "Internacional",
            "meio-ambiente":  "Meio Ambiente",
        },
        "ativa": True,
    },
    {
        "nome": "G1 Fato ou Fake",
        "url_inicial": "https://g1.globo.com/fato-ou-fake/",
        "seletor_link": ".feed-post-link",
        "seletor_viral": None,
        "ignorar_url": None,
        "ativa": True,
    },
    {
        "nome": "Lupa",
        "url_inicial": "https://www.agencialupa.org/checagem/",
        "seletor_link": "article.feed a.feed-link",
        "seletor_viral": None,
        "ignorar_url": None,
        "ativa": True,
    },
    {
        "nome": "Aos Fatos",
        "url_inicial": "https://www.aosfatos.org/noticias/",
        "seletor_link": "a.card",
        "seletor_viral": None,
        "ignorar_url": None,
        "ativa": False,   # feed via JavaScript -- ver comentário acima
    },
]

# ==============================================================================
# --- 3. ETAPA 1: MOTOR DE VARREDURA E EXTRAÇÃO TEXTUAL (SCRAPER BRUTO) ---
# ==============================================================================

def tema_por_url(url, nome_fonte):
    """Deriva o tema da categoria que a própria fonte atribuiu, lida da URL.
    Retorna None quando a fonte não categoriza, quando a categoria é genérica
    ('brasil', 'lista') ou quando não casa -- nesses casos o LLM decide."""
    fonte = next((f for f in FONTES if f["nome"] == nome_fonte), None)
    if not fonte or not fonte.get("mapa_tema") or not fonte.get("regex_categoria"):
        return None
    m = re.search(fonte["regex_categoria"], url or "")
    if not m:
        return None
    return fonte["mapa_tema"].get(m.group(1).lower())

def extrair_viral_estrutural(soup, seletor):
    """Extrai a mensagem viral pela marcação HTML, antes do get_text() achatar
    tudo. É aqui que se ganha o campo mais caro do pipeline de graça: o texto
    sai literal por construção, sem gastar tokens de geração."""
    if not seletor:
        return None

    blocos = [b.get_text(" ", strip=True) for b in soup.select(seletor)]
    blocos = [b for b in blocos if len(b) >= 40]
    if not blocos:
        return None

    # Havendo mais de um bloco (boato com "versão 1 / versão 2"), junta todos.
    return "\n\n".join(blocos)[:5000]

VIRAL_TEXTUAL_MIN = 40      # abaixo disso é fragmento, não mensagem
VIRAL_TEXTUAL_MAX = 5000

def extrair_viral_textual(texto_artigo, fonte):
    """Segunda via de extração determinística: recorta a transcrição pelo
    marcador textual que abre e pela seção que fecha.

    Existe porque a marcação HTML só cobre os artigos recentes. Roda sobre o
    texto já armazenado, então recupera acervo antigo sem nova requisição.

    É determinístico e literal por construção -- o recorte vem do próprio texto
    do artigo, não de geração. Retorna None quando não há marcador confiável;
    aí sim o LLM assume."""
    ini_pat = (fonte or {}).get("marcador_viral_inicio")
    if not ini_pat or not texto_artigo:
        return None

    m = re.search(ini_pat, texto_artigo, re.I)
    if not m:
        return None

    resto = texto_artigo[m.end():]

    fim_pat = fonte.get("marcador_viral_fim")
    if fim_pat:
        f = re.search(fim_pat, resto)
        if f:
            resto = resto[:f.start()]

    trecho = resto.strip()
    if not (VIRAL_TEXTUAL_MIN <= len(trecho) <= VIRAL_TEXTUAL_MAX):
        return None
    return trecho

def extrair_dados_noticia(url, fonte=None):
    """Acessa a notícia e devolve (titulo, texto, viral, metodo_do_viral).

    Tenta a extração determinística em duas vias, nesta ordem:
      1. marcação HTML (<blockquote>) -- cobre os artigos recentes
      2. marcador textual no corpo    -- cobre boa parte do arquivo antigo
    Só quando as duas falham é que o LLM assume, mais adiante no pipeline."""
    fonte = fonte or {}
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')

        for x in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            x.decompose()

        titulo = "Título não encontrado"
        tag_h1 = soup.find('h1')
        if tag_h1:
            titulo = tag_h1.get_text(strip=True)

        # Captura o viral ANTES do get_text() destruir a marcação.
        viral = extrair_viral_estrutural(soup, fonte.get("seletor_viral"))
        metodo = "html" if viral else None

        texto_limpo = soup.get_text(separator=' ', strip=True)[:MAX_CHARS_ARTIGO]

        if not viral:
            viral = extrair_viral_textual(texto_limpo, fonte)
            metodo = "texto" if viral else None

        return titulo, texto_limpo, viral, metodo
    except Exception as e:
        print(f"      [aviso] Erro ao baixar texto de {url}: {e}")
        return None, None, None, None

def varrer_site(fonte, limite, urls_ignoradas, pagina_inicial=1, ao_salvar=None):
    """Varre uma fonte e baixa as matérias inéditas.

    limite        -- None para varredura exaustiva; número para trazer só N novas.
    pagina_inicial-- retomar de uma página específica, evitando repercorrer o
                     começo do arquivo numa varredura interrompida.
    ao_salvar     -- callback chamado com as matérias novas a cada
                     SALVAR_A_CADA_ARTIGOS. Quando fornecido, a PERSISTÊNCIA É
                     RESPONSABILIDADE DELE: o chamador não deve gravar de novo o
                     que já foi entregue, sob pena de duplicar."""
    if not fonte.get("ativa", True):
        print(f"\n--- {fonte['nome']}: DESATIVADA (ver comentário em FONTES) ---")
        return []

    if limite is None:
        limite = float("inf")

    print(f"\n--- Iniciando Varredura Bruta em: {fonte['nome']} ---")
    if limite == float("inf"):
        print("   [modo exaustivo] sem limite de matérias; para no fim da paginação.")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    padrao_ignorar = re.compile(fonte["ignorar_url"]) if fonte.get("ignorar_url") else None
    links_para_baixar = []
    descartados_idioma = 0
    pagina = pagina_inicial
    urls_pagina_anterior = None

    while len(links_para_baixar) < limite and pagina <= MAX_PAGINAS:
        if fonte['nome'] == "Boatos.org": url = f"{fonte['url_inicial']}page/{pagina}/"
        elif fonte['nome'] == "Aos Fatos": url = f"{fonte['url_inicial']}?page={pagina}"
        else: url = fonte['url_inicial']

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(resp.text, 'html.parser')
            tags = soup.select(fonte['seletor_link'])

            if not tags:
                print(f"   [!] Fim da paginação na página {pagina}.")
                break

            novos_nesta_pagina = 0
            urls_desta_pagina = set()
            for t in tags:
                href = t.get('href', '')
                if not href.startswith('http'):
                    href = requests.compat.urljoin(fonte['url_inicial'], href)
                urls_desta_pagina.add(href)

                # Descarta versões em outro idioma antes de gastar download+LLM.
                if padrao_ignorar and padrao_ignorar.search(href):
                    descartados_idioma += 1
                    continue

                if href not in urls_ignoradas and href not in links_para_baixar:
                    links_para_baixar.append(href)
                    novos_nesta_pagina += 1
                if len(links_para_baixar) >= limite: break

            # Numa varredura exaustiva são centenas de páginas; imprimir todas
            # afogaria o log. Reporta quando achou algo, ou de 25 em 25.
            if novos_nesta_pagina or pagina % 25 == 0:
                print(f"   -> Página {pagina}: {novos_nesta_pagina} links inéditos. "
                      f"(fila: {len(links_para_baixar)})")

            # Proteção contra laço infinito: alguns sites devolvem sempre a mesma
            # página quando o número pedido passa do fim da paginação, e aí o
            # while nunca terminava (0 links novos, mas 'tags' não vazio).
            # Comparar o conjunto de URLs com o da página anterior distingue esse
            # caso de "0 novos porque já foram coletados antes" -- que é normal
            # ao retomar uma coleta e NÃO deve interromper a varredura.
            if urls_pagina_anterior is not None and urls_desta_pagina == urls_pagina_anterior:
                print(f"   [!] Página {pagina} idêntica à anterior: fim real da paginação.")
                break
            urls_pagina_anterior = urls_desta_pagina

            if fonte['nome'] not in ["Boatos.org", "Aos Fatos"]: break

            pagina += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"   [erro] Falha ao ler a página: {e}")
            break

    if pagina > MAX_PAGINAS:
        print(f"   [!] Teto de {MAX_PAGINAS} páginas atingido (MAX_PAGINAS).")

    if descartados_idioma:
        print(f"   [i] {descartados_idioma} links descartados por idioma (padrão {fonte['ignorar_url']!r}).")

    dados_extraidos = []
    com_viral = 0
    if links_para_baixar:
        total = len(links_para_baixar)
        print(f"   [*] Baixando o texto de {total} matérias "
              f"(~{total * (PAUSA_ENTRE_ARTIGOS + 0.7) / 60:.0f} min estimados)...")
        buffer = []
        for i, link in enumerate(links_para_baixar, 1):
            titulo, texto, viral, metodo = extrair_dados_noticia(link, fonte)
            if titulo and texto:
                registro = {
                    "url_origem": link,
                    "fonte": fonte['nome'],
                    "titulo": titulo,
                    "texto_bruto_jornalista": texto,
                    "processado_pela_ia": False
                }
                if viral:
                    registro["texto_viral_estrutural"] = viral
                    registro["viral_metodo"] = metodo   # "html" ou "texto"
                    com_viral += 1
                dados_extraidos.append(registro)
                buffer.append(registro)

            # Persistência incremental: sem isto, interromper uma varredura de
            # horas descartaria tudo o que já foi baixado.
            if ao_salvar and len(buffer) >= SALVAR_A_CADA_ARTIGOS:
                ao_salvar(buffer)
                print(f"       [{i}/{total}] {len(dados_extraidos)} baixadas — salvo.")
                buffer = []
            elif i % 50 == 0:
                print(f"       [{i}/{total}] {len(dados_extraidos)} baixadas.")

            time.sleep(PAUSA_ENTRE_ARTIGOS)

        if ao_salvar and buffer:
            ao_salvar(buffer)

    if dados_extraidos and fonte.get("seletor_viral"):
        print(f"   [i] Viral capturado estruturalmente em {com_viral}/{len(dados_extraidos)} matérias.")
    print(f"   [i] Última página visitada: {pagina} "
          f"(use pagina_inicial={pagina} para retomar daqui).")

    return dados_extraidos

def main_scraper():
    dataset_bruto = []
    urls_conhecidas = set()

    if os.path.exists(ARQUIVO_BRUTO):
        try:
            with open(ARQUIVO_BRUTO, "r", encoding='utf-8') as f:
                dataset_bruto = json.load(f)
                urls_conhecidas = {item['url_origem'] for item in dataset_bruto if 'url_origem' in item}
            print(f"-> Lake de Dados Brutos carregado. {len(urls_conhecidas)} matérias no histórico.")
        except:
            print("-> Aviso: Arquivo bruto corrompido ou vazio. Começando do zero.")

    novos_dados = 0

    def _persistir(novos):
        """Anexa e grava. Chamado durante o download, não só no fim, para que
        uma interrupção não descarte horas de coleta."""
        nonlocal novos_dados
        dataset_bruto.extend(novos)
        urls_conhecidas.update(i['url_origem'] for i in novos)
        novos_dados += len(novos)
        with open(ARQUIVO_BRUTO, "w", encoding='utf-8') as f:
            json.dump(dataset_bruto, f, indent=4, ensure_ascii=False)

    try:
        for fonte in FONTES:
            # A persistência fica por conta do callback -- não estender aqui de
            # novo, senão cada matéria entraria duas vezes.
            varrer_site(fonte, LIMIT_LINKS_POR_FONTE, urls_conhecidas,
                        pagina_inicial=PAGINA_INICIAL, ao_salvar=_persistir)
    except KeyboardInterrupt:
        print("\n[!] Interrompido pelo usuário. O que já foi baixado está salvo.")

    if novos_dados > 0:
        print(f"\n=== EXTRAÇÃO CONCLUÍDA ===")
        print(f"Novas matérias baixadas nesta execução: {novos_dados}")
        print(f"Acervo bruto total: {len(dataset_bruto)} matérias.")
    else:
        print("\n=== Nenhuma matéria nova encontrada. ===")

# ==============================================================================
# --- 4. ETAPA 2: INFERÊNCIA NLP (TEXTO VIRAL BRUTO + IDEIA CENTRAL + KEYWORDS) ---
# ==============================================================================

def chamar_llm(prompt, format_schema=None):
    """Chama o Ollama local (API REST nativa, sem chave/cota).
    Compartilhada pelas etapas de refinamento de falsas e de verdadeiras.

    format_schema: schema JSON opcional (dict) passado no campo "format" do
    Ollama, que força o modelo a responder em JSON estruturado válido (structured
    outputs). Sem isso, modelos mais "conversadores" como o qwen3.5 tendem a
    ignorar formatos de texto customizados (URL:/TEMA:/etc.) e responder em
    prosa livre -- o que quebrava o parser por regex."""
    global ENVIAR_THINK_FALSE

    tentativas = 0
    while tentativas < 3:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.15}
            }
            if format_schema is not None:
                payload["format"] = format_schema
            if ENVIAR_THINK_FALSE:
                payload["think"] = False

            resp = requests.post(OLLAMA_URL, json=payload, timeout=180)

            # Modelo sem suporte a "think" -> desliga o parâmetro e refaz a
            # chamada (não conta como tentativa: nada foi processado ainda).
            if resp.status_code == 400 and ENVIAR_THINK_FALSE:
                print("   [i] Modelo não aceita o parâmetro 'think'. Desativando e repetindo.")
                ENVIAR_THINK_FALSE = False
                continue

            resp.raise_for_status()
            return resp.json()["message"]["content"]

        except requests.exceptions.ConnectionError:
            print("   [!] Não foi possível conectar ao Ollama. Ele está rodando (`ollama serve`)?")
            tentativas += 1
            time.sleep(5)
        except Exception as e:
            print(f"   [!] Erro ao chamar o Ollama: {e}")
            tentativas += 1
            time.sleep(5)

    return None

# Schema de UM artigo por chamada. Note que 'url' NÃO está aqui: ela é
# encanamento, não extração -- o Python já sabe qual artigo enviou e religa o
# resultado por construção. Tirar a URL do escopo do modelo elimina de vez a
# classe de erro "modelo devolve URL truncada/trocada e o registro é perdido".
# O array "boatos" continua existindo para o caso de artigo-lista (um artigo que
# desmente vários boatos), mas agora todos são sabidamente do mesmo artigo.
SCHEMA_BOATOS = {
    "type": "object",
    "properties": {
        "boatos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "idioma": {"type": "string", "enum": IDIOMAS_VALIDOS},
                    "tema": {"type": "string", "enum": TEMAS_VALIDOS},
                    "texto_viral_bruto": {
                        "type": "string",
                        "description": "A mensagem/corrente original transcrita pelo jornalista, copiada na íntegra, sem alterar nada (erros, caixa alta, emojis)."
                    },
                    "claim_refinada": {"type": "string", "description": "A proposição falsa resumida em uma frase neutra."},
                    "palavras_chave": {"type": "array", "items": {"type": "string"}},
                    "justificativa": {
                        "type": "string",
                        "description": ("Por que a alegação não se sustenta, com base na apuração do "
                                        "jornalista: quais evidências, fontes e verificações foram "
                                        "usadas. Sem emitir veredito.")}
                },
                "required": ["idioma", "tema", "texto_viral_bruto", "claim_refinada",
                             "palavras_chave", "justificativa"]
            }
        }
    },
    "required": ["boatos"]
}

def _normalizar(texto):
    """Colapsa espaços e baixa a caixa -- o scraper já normaliza whitespace via
    get_text(separator=' '), então comparar assim evita falso negativo bobo."""
    return re.sub(r'\s+', ' ', texto or "").strip().lower()

def validar_trecho_literal(trecho, texto_fonte):
    """Confirma que o texto viral devolvido pelo modelo REALMENTE está no artigo
    original, em vez de ter sido parafraseado/alucinado.

    Este é o único campo do pipeline onde alucinação contamina o experimento: o
    texto viral é a matéria-prima da base 'bruta'. Como é extração literal (e não
    geração), dá para checar deterministicamente, sem depender do modelo.

    Estratégia: tenta o texto inteiro; se não bater (o modelo pode ter juntado
    trechos separados por corte de parágrafo), exige que pelo menos 2 de 3
    amostras -- início, meio e fim -- apareçam no fonte."""
    t = _normalizar(trecho)
    fonte = _normalizar(texto_fonte)

    if len(t) < 20:
        return False
    if t in fonte:
        return True

    janela = 60
    if len(t) < janela * 3:
        return False

    meio = (len(t) - janela) // 2
    amostras = [t[:janela], t[meio:meio + janela], t[-janela:]]
    return sum(1 for a in amostras if a in fonte) >= 2

def classificar_literalidade(trecho, texto_fonte):
    """Verifica a literalidade e distingue TRÊS estados, não dois.

    A distinção importa: quando o artigo guardado está no teto de caracteres, um
    resultado negativo pode ser só truncamento -- o trecho existe no site, mas
    não na cópia que temos. Reportar isso como 'falhou' seria contar erro de
    medição como erro de extração.

    Retorna (viral_verificado, status):
        True,  "conferido"                   -- achado no artigo
        None,  "inconclusivo_fonte_truncada" -- não achado, mas a fonte está cortada
        False, "falhou"                      -- não achado, com a fonte completa
    """
    if validar_trecho_literal(trecho, texto_fonte):
        return True, "conferido"
    # A cópia do artigo bateu no teto: negativo aqui não é conclusivo.
    if len(texto_fonte or "") >= 5000:
        return None, "inconclusivo_fonte_truncada"
    return False, "falhou"

# Variante do schema usada quando o texto viral já veio do HTML: o LLM só
# classifica e resume, sem reproduzir o trecho longo. É a maior economia do
# pipeline -- o texto viral costuma ser a maior parte dos tokens gerados.
# ------------------------------------------------------------------------------
# JUSTIFICATIVA -- o raciocínio do jornalista sobre por que a alegação não se
# sustenta. Campo pensado para explicabilidade (XAI), seguindo o precedente do
# LIAR-PLUS (Alhindi et al., 2018) e do PUBHEALTH (Kotonya & Toni, 2020).
#
# ATENÇÃO AO USO: a justificativa só existe do lado FALSO, porque nasce do artigo
# de checagem. Usá-la como feature de entrada num classificador de veracidade
# vaza o rótulo -- a simples presença do campo prediz a classe. Usos legítimos:
#   1. alvo de GERAÇÃO (treinar modelo que explica; avaliar com ROUGE)
#   2. evidência em verificação baseada em evidência (arranjo simétrico)
#   3. referência humana para avaliar explicações do modelo (LIME/SHAP/atenção)
#
# O LIAR-PLUS removia sentenças com vocabulário de veredito para separar
# evidência de julgamento. Fazemos o mesmo em filtrar_veredito().
PALAVRAS_VEREDITO = [
    "é falso", "e falso", "é falsa", "é fake", "trata-se de um boato", "é boato",
    "não é verdade", "nao e verdade", "é mentira", "desmentid", "fake news",
    "não procede", "nao procede", "é #fake", "classificamos como", "veredito",
    "checagem do boatos", "boatos.org classific",
]

# Rodapé/boilerplate que vem depois do corpo do artigo. Em regex e sem depender
# de caixa, porque o site varia a grafia ("PS:", "Ps:", "P.S.") e o texto do
# convite muda entre artigos. Sem isto, a justificativa arrasta e-mail de
# contato, link de WhatsApp e botões de rede social -- lixo que nada tem a ver
# com a apuração e que inflaria o campo artificialmente.
RODAPE_ARTIGO = re.compile(
    r"(?:"
    r"P\.?\s?S\.?\s*:"                       # "PS:", "Ps:", "P.S.:"
    r"|Esse artigo é uma sugestão"
    r"|Se você quiser sugerir"
    r"|entre em contato com a gente"
    r"|Siga[- ]nos no"
    r"|Siga o Boatos"
    r"|Checagem do Boatos\.org sobre"
    r"|Leia também"
    r"|Receba as checagens"
    r"|Compartilhe"
    r"|Assine"
    r"|Fontes?\s*:"
    r"|https?://"                            # qualquer link solto já é rodapé
    r")", re.I)

def filtrar_veredito(texto):
    """Remove as sentenças que emitem veredito, deixando só o raciocínio.

    Sem isto, a justificativa carregaria 'é falso que...' e viraria mais um
    caminho de vazamento de rótulo -- exatamente o que o LIAR-PLUS evitou com
    sua lista de palavras proibidas."""
    if not texto:
        return texto
    sentencas = re.split(r"(?<=[.!?])\s+", texto)
    mantidas = [s for s in sentencas
                if not any(p in s.lower() for p in PALAVRAS_VEREDITO)]
    return " ".join(mantidas).strip()

def tirar_viral_da_evidencia(just_bruta, viral):
    """Remove a alegação de dentro da evidência.

    Precisa acontecer AQUI, e não em `_fechar_justificativa()`, por uma razão de
    ordem: naquele ponto o viral ainda não é conhecido nos artigos sem
    <blockquote> -- quem o extrai é o LLM, depois. O resultado media 87% de
    contaminação nesse caminho, com a alegação inteira dentro do campo que
    deveria servir de padrão-ouro de evidência (RQ3.2).

    Também corrige o caminho estrutural: lá `_fechar_justificativa()` retira o
    `texto_viral_estrutural` do artigo, mas em artigo-lista o viral do registro
    é UM bloco, que pode não coincidir com o texto removido."""
    if not just_bruta or not viral or len(viral) < 40:
        return just_bruta
    novo = just_bruta.replace(viral, " ") if viral in just_bruta else just_bruta
    if novo is just_bruta or novo == just_bruta:
        # Espaçamento pode divergir entre o bloco e o corpo do artigo.
        chave = re.sub(r"\s+", "", viral.lower())
        novo = " ".join(s for s in re.split(r"(?<=[.!?])\s+", just_bruta)
                        if len(s) < 40 or re.sub(r"\s+", "", s.lower()) not in chave)
    novo = re.sub(r"\s{2,}", " ", novo).strip()
    return novo if len(novo) >= 150 else None

def _fechar_justificativa(trecho, viral, minimo=200):
    """Limpeza comum a todas as vias: tira o viral (que é a alegação, não a
    apuração), corta o rodapé, remove as sentenças de veredito e normaliza
    espaço. Devolve None se o que sobra é curto demais para ser argumentação."""
    if not trecho:
        return None
    if viral:
        trecho = trecho.replace(viral, " ")
    corte = RODAPE_ARTIGO.search(trecho)
    if corte and corte.start() > 150:
        trecho = trecho[:corte.start()]
    trecho = filtrar_veredito(trecho)
    trecho = re.sub(r"\s{2,}", " ", trecho or "").strip()
    return trecho if len(trecho) >= minimo else None

def extrair_justificativa_bruta(item):
    """Recorta do artigo o trecho onde o jornalista argumenta.

    Cinco vias, em ordem decrescente de precisão do ponto de ancoragem. Medido
    sobre os 12.708 artigos em português da base:

      1. seção "Análise"   -> 30,1%  (o argumento começa exatamente ali)
      2. seção "Checagem"  ->  0,3%  (template mais antigo; quase sempre vem
                                      junto de "Análise", que ganha por vir antes)
      3. após "Boato - <resumo da alegação>" -> 65,8%. Este é o marcador que
         faltava: existe em 96% dos artigos, contra 29% de "Checagem". Pula-se a
         primeira sentença porque ela reafirma a alegação, não a apura.
      4./5. depois ou antes do viral estrutural -> 0,4%
      6. artigo inteiro menos viral e rodapé -> 3,4%

    Cobertura total: 99,9%. A versão anterior ancorava só em "Checagem" e no
    texto DEPOIS do viral, e cobria 37,8% -- porque em 2.778 artigos a
    argumentacao vem ANTES da citação da mensagem, e ali sobravam menos de 150
    caracteres. Os outros 62% caíam no LLM sem trecho delimitado, que é onde
    apareceram as justificativas ausentes.

    Retorna None só quando nem o artigo inteiro rende 200 caracteres (18 casos)."""
    texto = item.get("texto_bruto_jornalista", "")
    if not texto:
        return None
    viral = item.get("texto_viral_estrutural")

    # Vias 1 e 2: cabeçalho de seção explícito.
    for padrao in (r"\bAn[\u00e1a]lise\b", r"\bChecagem\b(?!\s+do\s+Boatos)"):
        m = re.search(padrao, texto)
        if m:
            t = _fechar_justificativa(texto[m.end():], viral)
            if t:
                return t

    # Via 3: depois da linha "Boato - ...", pulando a sentença que só reafirma
    # a alegação. É a via que cobre dois terços do corpus.
    m = re.search(r"\bBoato\s*[\u2013\u2014-]\s*", texto)
    if m:
        resto = texto[m.end():]
        ponto = re.search(r"(?<=[.!?])\s", resto)
        t = _fechar_justificativa(resto[ponto.end():] if ponto else resto, viral)
        if t:
            return t

    # Vias 4 e 5: ancoradas no viral, quando ele foi extraído estruturalmente.
    if viral:
        pos = texto.find(viral[:150])
        if pos >= 0:
            t = (_fechar_justificativa(texto[pos + len(viral):], viral)
                 or _fechar_justificativa(texto[:pos], viral))
            if t:
                return t

    # Via 6: último recurso.
    return _fechar_justificativa(texto, viral)

SCHEMA_BOATOS_SEM_VIRAL = {
    "type": "object",
    "properties": {
        "boatos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    # Qual bloco transcrito corresponde a este boato. Índice inteiro
                    # é muito mais seguro que pedir ao modelo para ecoar o texto (ou
                    # uma URL): é curto, e o Python valida se está na faixa antes de
                    # usar. Sem isso, um artigo-lista com 3 boatos gravava o MESMO
                    # texto viral nos 3 registros -- 26% da base ficou duplicada.
                    "bloco_indice": {"type": "integer",
                                     "description": "Número do BLOCO que contém este boato."},
                    "idioma": {"type": "string", "enum": IDIOMAS_VALIDOS},
                    "tema": {"type": "string", "enum": TEMAS_VALIDOS},
                    "claim_refinada": {"type": "string", "description": "A proposição falsa resumida em uma frase neutra."},
                    "palavras_chave": {"type": "array", "items": {"type": "string"}},
                    "justificativa": {
                        "type": "string",
                        "description": ("Por que a alegação não se sustenta, com base na apuração do "
                                        "jornalista: quais evidências, fontes e verificações foram "
                                        "usadas. Sem emitir veredito.")}
                },
                "required": ["bloco_indice", "idioma", "tema", "claim_refinada",
                             "palavras_chave", "justificativa"]
            }
        }
    },
    "required": ["boatos"]
}

# Devolvido pela via estrutural quando todos os blocos da marcação são lixo.
# Diferente de None (falha do modelo) e de [] (artigo sem boato): significa
# "tente a via do LLM".
SEM_BLOCO_UTIL = object()


def _extrair_com_viral_estrutural(item):
    """Caminho rápido: o viral veio do HTML, então o LLM só classifica e resume.

    Os blocos são apresentados numerados e o modelo devolve o índice de cada um,
    para que artigos que desmentem vários boatos tenham o texto viral CORRETO em
    cada registro em vez do bloco inteiro repetido."""
    # extrair_viral_estrutural() junta os blocos com "\n\n", e get_text(" ") não
    # deixa quebra de linha dentro de um bloco -- então este split é exato.
    brutos = [b.strip() for b in item["texto_viral_estrutural"].split("\n\n") if b.strip()]

    # Limpeza ANTES do LLM ver os blocos. A via determinística também erra:
    # MEDIDO, 262 dos 308 virais degenerados vieram do <blockquote> (embed do
    # Instagram sem legenda, "Confira também: ..."), e 275 traziam a introdução
    # do checador. Bloco que vira lixo sai; se todos saírem, o artigo não tem
    # viral estrutural útil e cai na via do LLM (sentinela SEM_BLOCO_UTIL).
    limpos = [limpar_viral(b) for b in brutos]
    # Voz do checador (título com "#boato", assinatura, narração) também sai:
    # não é o boato, e carrega o rótulo.
    limpos = [l for l in limpos if not l["degenerada"] and not l["checador"]]
    if not limpos:
        return SEM_BLOCO_UTIL
    blocos = [l["texto"] for l in limpos]
    viral_completo = "\n\n".join(blocos)

    if len(blocos) == 1:
        apresentacao = f"[BLOCO 1]\n{blocos[0][:2500]}"
        instrucao_bloco = "Só existe o BLOCO 1, então bloco_indice deve ser 1."
    else:
        apresentacao = "\n\n".join(
            f"[BLOCO {n}]\n{b[:1500]}" for n, b in enumerate(blocos, 1)
        )
        instrucao_bloco = (f"Existem {len(blocos)} blocos numerados. Para cada boato, "
                           f"informe em bloco_indice o número do bloco que o contém.")

    # A justificativa precisa do trecho onde o jornalista ARGUMENTA, não do
    # cabeçalho do artigo. Quando o recorte determinístico existe, ele é o
    # contexto ideal: já é a parte relevante, e economiza tokens de entrada.
    just_bruta = extrair_justificativa_bruta(item)
    if just_bruta:
        contexto = just_bruta[:2500]
        rotulo_ctx = "APURAÇÃO DO JORNALISTA (base para a justificativa)"
    else:
        contexto = item['texto_bruto_jornalista'][:2500]
        rotulo_ctx = "ARTIGO DE CHECAGEM (extraia daqui a apuração)"

    prompt = f"""
    Você é um engenheiro de dados especialista em NLP e desinformação.
    Abaixo estão as MENSAGENS FALSAS que circularam nas redes sociais, transcritas
    por um site de checagem de fatos, seguidas da apuração jornalística.

    Preencha o array "boatos" com um item para CADA afirmação falsa distinta. {instrucao_bloco}
    Cada item:

    1. bloco_indice: o número do BLOCO que contém este boato.
    2. idioma: o idioma em que a mensagem circulava.
    3. tema: a categoria do boato.
    4. claim_refinada: a afirmação falsa resumida em uma única frase neutra e direta.
    5. palavras_chave: de 5 a 10 termos cruciais (pessoas, entidades, temas).
    6. justificativa: em 2 a 4 frases, POR QUE a alegação não se sustenta, segundo a
       apuração abaixo -- quais evidências, fontes consultadas, datas, imagens
       originais ou contradições o jornalista usou.
       REGRA IMPORTANTE: descreva as EVIDÊNCIAS, não o veredito. Não escreva "é
       falso", "é boato", "trata-se de fake news" nem equivalentes. Em vez de
       "é falso que o vídeo seja de 2026", escreva "o vídeo foi publicado
       originalmente em 2019 no canal X, segundo busca reversa de imagem".

    MENSAGENS TRANSCRITAS:
    {apresentacao}

    {rotulo_ctx}:
    {contexto}
    """

    tentativas = 0
    while tentativas < 3:
        resposta = chamar_llm(prompt, format_schema=SCHEMA_BOATOS_SEM_VIRAL)
        if resposta is None:
            return None

        try:
            dados = json.loads(resposta)
        except json.JSONDecodeError as e:
            print(f"   [DEBUG] JSON inválido ({e}): {resposta[:400]}")
            tentativas += 1
            time.sleep(2)
            continue

        boatos = []
        blocos_usados = set()
        for b in dados.get("boatos", []):
            claim = (b.get("claim_refinada") or "").strip()
            if not claim:
                continue

            # Valida o índice antes de confiar nele.
            # 'metodo' distingue as duas vias determinísticas: marcação HTML
            # (blockquote) e marcador textual no corpo do artigo. Ambas são
            # literais; guardá-las separadas permite comparar a fidelidade das
            # duas na análise do corpus.
            metodo = item.get("viral_metodo", "html")
            idx = b.get("bloco_indice")
            if len(blocos) == 1:
                idx = 1
            if isinstance(idx, int) and 1 <= idx <= len(blocos):
                viral_do_boato = blocos[idx - 1]
                limpeza = limpos[idx - 1]
                origem = metodo
            else:
                # Índice inválido: usa o conjunto e marca, para ser auditável.
                idx = "todos"
                viral_do_boato = viral_completo
                limpeza = {"prefixos": [p for l in limpos for p in l["prefixos"]],
                           "versoes": max(l["versoes"] for l in limpos)}
                origem = f"{metodo}_sem_indice"

            # Um bloco, um registro. MEDIDO: em 100 artigos o mesmo viral foi
            # gravado para alegações diferentes (192 registros nesta via). A
            # segunda alegação que aponta para um bloco já usado não tem viral
            # próprio -- sai, em vez de herdar o texto de outra.
            if idx in blocos_usados:
                continue
            blocos_usados.add(idx)

            # A verificação roda TAMBÉM aqui. O texto vem da marcação, então
            # espera-se que passe -- mas marcar True sem checar tornaria a
            # métrica tautológica, e ela é justamente a evidência de que a
            # extração estrutural é superior. Medida, não presumida.
            verificado, status = classificar_literalidade(
                viral_do_boato, item['texto_bruto_jornalista'])

            # Filtro de veredito também na saída do modelo: mesmo instruído a não
            # emitir julgamento, ele às vezes escorrega, e o campo não pode
            # carregar o rótulo dentro de si.
            justificativa = filtrar_veredito((b.get("justificativa") or "").strip())

            boatos.append({
                "idioma": (b.get("idioma") or "pt-BR").strip(),
                "tema": (b.get("tema") or "Outros").strip(),
                "texto_viral_bruto": viral_do_boato,
                "claim_refinada": claim,
                "palavras_chave": [p.strip() for p in b.get("palavras_chave", []) if p and p.strip()],
                "justificativa": justificativa,
                "justificativa_bruta": tirar_viral_da_evidencia(
                    filtrar_veredito(just_bruta) if just_bruta else None, viral_do_boato),
                "justificativa_origem": "recorte+llm" if just_bruta else "llm",
                "viral_verificado": verificado,
                "verificacao_viral": status,
                "origem_viral": origem,
                "blocos_no_artigo": len(blocos),
                "viral_prefixos_removidos": limpeza["prefixos"],
                "viral_versoes": limpeza["versoes"],
            })
        return boatos

    return None

def extrair_boatos_do_artigo(item):
    """Extrai os boatos de UM artigo. Retorna:
      - lista de boatos (pode ser vazia: artigo legítimo sem boato claro)
      - None se o modelo falhou de fato (sem resposta / JSON inválido nas 3 tentativas)

    A distinção importa: vazio = processado com sucesso, pode marcar como feito;
    None = falhou, deve continuar na fila para uma próxima execução.

    Se a raspagem capturou o viral pela marcação, usa o caminho rápido. Se todos
    os blocos da marcação forem lixo, cai no caminho do LLM."""
    if item.get("texto_viral_estrutural"):
        resultado = _extrair_com_viral_estrutural(item)
        if resultado is not SEM_BLOCO_UTIL:
            return resultado

    texto_artigo = item['texto_bruto_jornalista'][:3800]

    # Mesmo recorte determinístico do caminho estrutural. Sem isto, este caminho
    # -- que atende 50,2% do acervo, os artigos sem <blockquote> -- não produzia
    # justificativa nenhuma: o campo não estava no schema nem no prompt.
    just_bruta = extrair_justificativa_bruta(item)
    contexto_apuracao = (f"\n\n    APURAÇÃO DO JORNALISTA (base para a justificativa):\n"
                         f"    {just_bruta[:2500]}") if just_bruta else ""

    prompt = f"""
    Você é um engenheiro de dados especialista em NLP e desinformação.
    Analise o texto jornalístico bruto abaixo, vindo de um site de checagem de fatos.

    Preencha o array "boatos" com um item para CADA afirmação falsa desmentida no artigo
    (a maioria dos artigos desmente apenas uma; artigos-lista desmentem várias). Cada item:

    1. idioma: o idioma em que o boato circulava (pt-BR, es-ES, en-US ou outro).
    2. tema: a categoria do boato.
    3. texto_viral_bruto: a seção onde o jornalista TRANSCREVE a mensagem original das redes
       sociais. Copie ESSE TRECHO LITERALMENTE, palavra por palavra, do texto abaixo --
       mantenha erros de português, maiúsculas, emojis e alarmismo. NÃO reescreva, NÃO
       resuma, NÃO traduza. Regras:
       a) Comece na PRIMEIRA PALAVRA DA MENSAGEM. A apresentação que o jornalista escreve
          antes dela NÃO faz parte: não copie "Leia:", "Leia a mensagem:", "Confira o texto
          circulante:", "Texto:", "Versão 1:" nem rótulos parecidos.
       b) Se o artigo transcreve várias VERSÕES da mesma mensagem ("Versão 1", "Versão 2"...),
          copie SÓ a primeira versão.
       c) Cada item precisa do SEU PRÓPRIO trecho transcrito. Se o artigo desmente várias
          alegações mas transcreve uma única mensagem, gere um único item.
       d) Nunca copie texto do próprio site nem frases do jornalista: chamadas como
          "Confira também", "Assista", "Ver essa foto no Instagram", créditos de imagem, ou
          frases que avaliam a mensagem ("é falso", "boato", "não procede").
       e) NUNCA copie o TÍTULO do artigo de checagem -- ele costuma terminar com "#boato"
          e vem seguido de categoria, data e nome do autor. O título é a frase do
          jornalista resumindo o boato, não a mensagem que circulou. Nem a narração do
          jornalista sobre a mensagem ("diz a mensagem", "de acordo com o texto").
       Se o artigo não transcrever a mensagem original, não gere item para esse boato.
    4. claim_refinada: a afirmação falsa resumida em uma única frase neutra e direta.
    5. palavras_chave: de 5 a 10 termos cruciais do boato (pessoas, entidades, temas).
    6. justificativa: POR QUE a alegação não se sustenta, segundo a apuração do jornalista.
       Descreva as EVIDÊNCIAS e VERIFICAÇÕES: o que foi consultado, quais fontes oficiais
       responderam, que datas ou origens de imagem foram encontradas, que incoerências
       internas o texto tem. NÃO emita veredito -- não escreva "é falso", "é boato",
       "é fake" nem cite o nome do site de checagem. Escreva 2 a 4 frases.

    Se o artigo não desmentir nenhum boato (ex.: é uma página institucional, índice ou
    reportagem comum), devolva "boatos" como array vazio.

    TEXTO DO ARTIGO:
    {texto_artigo}{contexto_apuracao}
    """

    tentativas = 0
    while tentativas < 3:
        texto_resposta = chamar_llm(prompt, format_schema=SCHEMA_BOATOS)

        if texto_resposta is None:
            return None

        try:
            dados = json.loads(texto_resposta)
        except json.JSONDecodeError as e:
            print(f"   [DEBUG] JSON inválido do modelo ({e}). Resposta bruta:")
            print("   " + "-"*70)
            print(texto_resposta[:1500])
            print("   " + "-"*70)
            tentativas += 1
            time.sleep(2)
            continue

        boatos = []
        virais_usados = set()
        for b in dados.get("boatos", []):
            try:
                viral = (b["texto_viral_bruto"] or "").strip()
                claim = (b["claim_refinada"] or "").strip()
            except (KeyError, AttributeError):
                continue

            if not viral or not claim:
                continue

            # Rede de segurança: o prompt já proíbe introdução, versões extras e
            # texto do site, mas o modelo escorrega. As MESMAS regras da via
            # estrutural rodam sobre a saída dele.
            limpeza = limpar_viral(viral)
            if limpeza["degenerada"] or limpeza["checador"]:
                continue
            viral = limpeza["texto"]
            # Um viral, um registro (ver a via estrutural).
            chave_viral = re.sub(r"\s+", " ", viral.lower())
            if chave_viral in virais_usados:
                continue
            virais_usados.add(chave_viral)

            boatos.append({
                "idioma": (b.get("idioma") or "pt-BR").strip(),
                "tema": (b.get("tema") or "Outros").strip(),
                "texto_viral_bruto": viral,
                "claim_refinada": claim,
                "palavras_chave": [p.strip() for p in b.get("palavras_chave", []) if p and p.strip()],
                "justificativa": filtrar_veredito((b.get("justificativa") or "").strip()),
                "justificativa_bruta": tirar_viral_da_evidencia(
                    filtrar_veredito(just_bruta) if just_bruta else None, viral),
                "justificativa_origem": "recorte+llm" if just_bruta else "llm",
                # Proveniência: marcamos em vez de descartar, para você poder
                # medir a taxa de literalidade por modelo e decidir o filtro
                # depois (filtrar aqui esconderia o dado da avaliação).
                **dict(zip(("viral_verificado", "verificacao_viral"),
                           classificar_literalidade(viral, item['texto_bruto_jornalista']))),
                "origem_viral": "llm",
                "viral_prefixos_removidos": limpeza["prefixos"],
                "viral_versoes": limpeza["versoes"],
            })

        # Array vazio é resposta legítima (artigo sem boato), não falha.
        return boatos

    return None

def main_nlp():
    if not os.path.exists(ARQUIVO_BRUTO):
        print("ERRO: 'dataset_bruto.json' não encontrado. Rode o Scraper primeiro.")
        return

    with open(ARQUIVO_BRUTO, "r", encoding='utf-8') as f:
        dados_brutos = json.load(f)

    # O filtro de idioma por URL existe na varredura, mas o acervo coletado ANTES
    # dessa mudança está cheio de versões em espanhol/inglês. Sem repetir o filtro
    # aqui, cada uma consome uma chamada de LLM só para ser descartada depois pelo
    # veredito de idioma -- foi o que aconteceu em 8 dos 10 artigos do último teste.
    padroes_idioma = {f["nome"]: (re.compile(f["ignorar_url"]) if f.get("ignorar_url") else None)
                      for f in FONTES}

    fila = []
    pulados_idioma = 0
    for item in dados_brutos:
        if item.get("processado_pela_ia", False):
            continue
        pad = padroes_idioma.get(item.get("fonte"))
        if pad and pad.search(item.get("url_origem", "")):
            # Marca como processado com zero boatos: é decisão definitiva
            # (a URL não muda de idioma), então não fica reaparecendo na fila.
            item["processado_pela_ia"] = True
            item["boatos_extraidos"] = 0
            item["motivo_descarte"] = "idioma_url"
            pulados_idioma += 1
            continue
        fila.append(item)

    if pulados_idioma:
        with open(ARQUIVO_BRUTO, "w", encoding='utf-8') as f:
            json.dump(dados_brutos, f, indent=4, ensure_ascii=False)
        print(f"-> {pulados_idioma} matérias em outro idioma removidas da fila (sem gastar LLM).")

    print(f"-> {len(fila)} matérias aguardando refinamento NLP.")

    if LIMITE_ARTIGOS_NLP:
        fila = fila[:LIMITE_ARTIGOS_NLP]
        print(f"-> LIMITE_ARTIGOS_NLP ativo: processando apenas {len(fila)} nesta execução.")

    if not fila: return

    dataset_final = []
    if os.path.exists(ARQUIVO_FINAL):
        with open(ARQUIVO_FINAL, "r", encoding='utf-8') as f:
            dataset_final = json.load(f)

    def _salvar():
        with open(ARQUIVO_FINAL, "w", encoding='utf-8') as f:
            json.dump(dataset_final, f, indent=4, ensure_ascii=False)
        with open(ARQUIVO_BRUTO, "w", encoding='utf-8') as f:
            json.dump(dados_brutos, f, indent=4, ensure_ascii=False)

    sucessos_hoje = 0
    falhas = 0
    nao_literais = 0
    descartados_idioma = 0

    for i, materia in enumerate(fila, 1):
        titulo_curto = (materia.get('titulo') or '')[:65]
        print(f"[{i}/{len(fila)}] {titulo_curto}")

        boatos = extrair_boatos_do_artigo(materia)

        if boatos is None:
            # Falha real do modelo: NÃO marca como processado, para a matéria
            # continuar na fila e ser tentada de novo numa próxima execução.
            falhas += 1
            print("   [!] Falhou nas 3 tentativas -- mantida na fila para a próxima execução.")
            continue

        gravados = 0
        for extraido in boatos:
            if "PT" not in extraido['idioma'].upper():
                descartados_idioma += 1
                continue

            if not extraido['viral_verificado']:
                nao_literais += 1

            # Tema: prefere a categoria que a própria fonte atribuiu (editor
            # humano) e cai no LLM só quando ela não existe. Guarda os dois, o
            # que de graça permite medir a concordância do modelo depois.
            tema_url = tema_por_url(materia['url_origem'], materia['fonte'])

            dataset_final.append({
                "titulo_artigo_checagem": materia['titulo'],
                "tema": tema_url or extraido['tema'],
                "tema_llm": extraido['tema'],
                "tema_origem": "url" if tema_url else "llm",
                "idioma": extraido['idioma'],
                "veredito": "FALSO",
                "url_origem_fake": materia['url_origem'],
                "fonte_fake": materia['fonte'],
                # ATENÇÃO: este é o artigo do site de CHECAGEM (o jornalista
                # desmentindo), não a fake news. Serve como rastreabilidade da
                # rotulagem e como fonte para conferir a literalidade do texto
                # viral -- NÃO deve ser usado como entrada de treino: ele contém
                # "é falso que", "boato", "checamos", e um classificador
                # treinado nele aprende a reconhecer artigo de fact-check, não
                # desinformação. A entrada crua correta do lado FALSO é o
                # texto_viral_bruto.
                "texto_bruto_jornalista": materia['texto_bruto_jornalista'],
                "texto_viral_bruto": extraido['texto_viral_bruto'],
                "claim_refinada": extraido['claim_refinada'],
                # Alias unificado com o campo equivalente do lado VERDADEIRO -- usado na base "refinada".
                "ideia_central": extraido['claim_refinada'],
                # --- explicabilidade (XAI) ---
                # NÃO usar como feature de entrada num classificador de veracidade:
                # o campo só existe do lado FALSO, então sua presença prediz a
                # classe. Usos previstos: alvo de geração de explicação, evidência
                # em verificação baseada em evidência, e referência humana para
                # avaliar as explicações do modelo. Ver README.md.
                "justificativa": extraido.get('justificativa', ''),
                "justificativa_bruta": extraido.get('justificativa_bruta'),
                "justificativa_origem": extraido.get('justificativa_origem', 'llm'),
                "palavras_chave": extraido['palavras_chave'],
                # Proveniência: o texto viral foi conferido no artigo original?
                # True / False / None (inconclusivo -- artigo guardado truncado)
                "viral_verificado": extraido['viral_verificado'],
                "verificacao_viral": extraido.get('verificacao_viral', 'nao_avaliado'),
                # "html"             = bloco da marcação, associado por índice
                # "html_sem_indice"  = bloco da marcação, índice inválido -> texto conjunto
                # "llm"              = gerado pelo modelo e conferido por heurística
                "origem_viral": extraido.get('origem_viral', 'llm'),
                "blocos_no_artigo": extraido.get('blocos_no_artigo', 1),
                # Limpeza aplicada ao viral (limpeza_viral.py) e versão da
                # extração -- separam no corpus o que passou pelas regras novas.
                "viral_prefixos_removidos": extraido.get('viral_prefixos_removidos', []),
                "viral_versoes": extraido.get('viral_versoes', 1),
                "extracao_versao": VERSAO_EXTRACAO,
                "modelo_refinamento": OLLAMA_MODEL,
                "pareado": False
            })
            gravados += 1
            sucessos_hoje += 1

        # Marca só ESTA matéria, e só porque a extração concluiu de fato.
        materia['processado_pela_ia'] = True
        # Registra QUANTOS boatos saíram. Zero aqui é um resultado legítimo
        # (artigo sem mensagem viral transcrita, ex.: desmentido de imagem), e é
        # o que distingue "processado e vazio de verdade" de "falhou" -- a falha
        # nem chega aqui: não marca processado e fica na fila.
        materia['boatos_extraidos'] = gravados

        if gravados:
            print(f"   -> {gravados} boato(s) gravado(s).")
        else:
            print("   -> Nenhum boato em português identificado neste artigo.")

        if i % SALVAR_A_CADA == 0 or i == len(fila):
            _salvar()
            print(f"   [salvo] {len(dataset_final)} registros no arquivo final.")

    _salvar()

    print(f"\n=== FASE 2 FINALIZADA ({OLLAMA_MODEL}) ===")
    print(f"-> {sucessos_hoje} instâncias de fake news gravadas.")
    print(f"-> {falhas} artigos falharam e seguem na fila.")
    print(f"-> {descartados_idioma} boatos descartados por não serem em português.")
    if sucessos_hoje:
        taxa = 100 * (sucessos_hoje - nao_literais) / sucessos_hoje
        print(f"-> Literalidade do texto viral: {sucessos_hoje - nao_literais}/{sucessos_hoje} conferidos no artigo original ({taxa:.1f}%).")

# ==============================================================================
# --- 5. ORQUESTRAÇÃO ---
# ==============================================================================
# O lado VERDADEIRO e as etapas seguintes vivem em scripts próprios -- ver o
# cabeçalho deste arquivo.

if __name__ == "__main__":
    main_scraper()
    main_nlp()
