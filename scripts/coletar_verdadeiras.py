# -*- coding: utf-8 -*-
"""Coleta o lado VERDADEIRO na Agência Brasil (CC BY 3.0 BR).

Desenho em três decisões que valem explicação:

1. GRAVA EM ARQUIVO PRÓPRIO (`dataset_verdadeiras.jsonl`), não no dataset final.
   São ~44 h de download; escrever num arquivo de 160 MB a cada 10 registros
   durante dois dias é convite a corromper o lado falso, que está pronto. A
   fusão é uma passada determinística depois, quando tudo estiver no lugar.

2. RETOMÁVEL POR CONSTRUÇÃO. A cada 10 matérias grava, e ao reiniciar pula o
   que já tem. Uma coleta de 44 h vai ser interrompida -- queda de energia,
   reinício, Ctrl+C -- e a pergunta não é se, é quando.

3. QUOTA POR CÉLULA, MAS SEM EXIGIR PARIDADE. Pede-se por (balde, ano) o
   mínimo entre a demanda do lado falso e a oferta disponível. Onde não fecha
   -- Ciência e Tecnologia, que a Agência Brasil não cobre -- simplesmente vem
   menos, e o desbalanceamento fica registrado em vez de travar a coleta.

Fontes das URLs: sitemap (2013-2020, legado) + relatorios/coleta/urls_cdx.json
(2021-2026, via Internet Archive). O sitemap é relido aqui para o script ser
autossuficiente.

    python scripts/coletar_verdadeiras.py --sondar 5   # valida a extração
    python scripts/coletar_verdadeiras.py --limite 50  # amostra cronometrada
    python scripts/coletar_verdadeiras.py              # coleta completa
"""
import os
import re
import sys
import json
import time
import random
import requests
from datetime import datetime
from collections import Counter, defaultdict
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taxonomia                                        # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_FALSAS = os.path.join(_RAIZ, "datasets", "dataset_tcc_fake_news_final.json")
# JSONL, não JSON: uma linha por registro, aberto em modo append.
#
# Com JSON seria preciso reescrever o arquivo inteiro a cada salvamento. A
# 13.900 registros o arquivo passa de 100 MB, e salvar a cada 10 daria ~70 GB
# de escrita ao longo das 40 h para produzir 100 MB de dado -- desgaste
# gratuito de SSD, especialmente num notebook. Em JSONL cada registro custa o
# seu próprio tamanho e nada mais. A conversão para JSON é instantânea no fim.
ARQ_SAIDA = os.path.join(_RAIZ, "datasets", "dataset_verdadeiras.jsonl")
# Descartes PERMANENTES (404/410, corpo curto): sem este registro a retomada
# tentaria de novo as mesmas páginas mortas, 10 s cada, a cada reinício.
ARQ_DESCARTES = os.path.join(_RAIZ, "datasets", "dataset_verdadeiras_descartes.jsonl")
ARQ_TRAVA = os.path.join(_RAIZ, "datasets", ".coletar_verdadeiras.lock")
ARQ_CDX = os.path.join(_RAIZ, "relatorios", "coleta", "urls_cdx.json")
# O sitemap é legado e congelado em 2020: baixá-lo de novo a cada execução só
# custa ~3 min de Crawl-delay e não traz nada novo.
ARQ_SITEMAP = os.path.join(_RAIZ, "relatorios", "coleta", "urls_sitemap.json")

BASE = "https://agenciabrasil.ebc.com.br"
SITEMAP = f"{BASE}/sitemap.xml"
VEICULO = "Agência Brasil"
LICENCA = "CC BY 3.0 BR"

PAUSA = 10                 # Crawl-delay declarado no robots.txt da EBC
# MEDIDO na amostra de 50: 21% das matérias de 2021-2026 (0% das de 2013-2020)
# devolvem "Página temporariamente indisponível" com HTTP 200, de forma
# persistente -- a matéria existiu, o Archive a guardou, o site é que não a
# serve mais. Repor com outra URL trocaria a amostra aleatória por "o que o
# site ainda serve", viés que não dá para descartar. Então o texto vem da
# cópia do Internet Archive, e o registro diz isso em `texto_via`.
PAGINA_INDISPONIVEL = "temporariamente indispon"
ARCHIVE = "https://web.archive.org/web/{ano}id_/{url}"   # id_: HTML original, sem barra
PAUSA_ARCHIVE = 3
TIMEOUT = 45
SALVAR_A_CADA = 10
MAX_TENTATIVAS = 2
MIN_CHARS_CORPO = 400      # abaixo disso é erro, paywall ou página vazia
SEED = 0
UA = {"User-Agent": "Mozilla/5.0 (pesquisa academica TCC; coleta respeitando robots.txt)"}

PADRAO_URL = re.compile(r"/([a-z\-]+)/noticia/(\d{4})-(\d{2})/")
NAO_E_PAGINA = re.compile(r"\.(jpg|jpeg|png|gif|webp|svg|pdf|mp3|mp4|zip|css|js)(\?|$)", re.I)
NAO_MATERIA = re.compile(r"/(foto|fotos|audio|audios|video|videos|galeria|tag|autor"
                         r"|infografico|radioagencia)/", re.I)
LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.I)

_ultimo = [0.0]


def _aguardar():
    dt = time.time() - _ultimo[0]
    if dt < PAUSA:
        time.sleep(PAUSA - dt)
    _ultimo[0] = time.time()


def baixar(url, pausa=PAUSA):
    for n in range(MAX_TENTATIVAS):
        if pausa:
            _aguardar()
        try:
            r = requests.get(url, headers=UA, timeout=TIMEOUT)
        except Exception as e:
            if n == MAX_TENTATIVAS - 1:
                return None, type(e).__name__
            time.sleep(5)
            continue
        if r.status_code == 200:
            # Sem charset no cabeçalho, o requests presume ISO-8859-1 e o
            # texto sai "PÃ¡gina". O site é todo UTF-8.
            if "charset" not in r.headers.get("Content-Type", "").lower():
                r.encoding = "utf-8"
            return r.text, None
        if r.status_code in (404, 410):
            return None, f"HTTP {r.status_code}"     # não insiste em página morta
        if n == MAX_TENTATIVAS - 1:
            return None, f"HTTP {r.status_code}"
        time.sleep(10)
    return None, "esgotou"


_ultimo_archive = [0.0]


def baixar_do_archive(url):
    """Cópia arquivada mais próxima do ano seguinte à publicação -- perto o
    bastante para ser a matéria como saiu, depois de eventuais correções do
    dia. Devolve (html, timestamp_do_snapshot) ou (None, motivo)."""
    dt = time.time() - _ultimo_archive[0]
    if dt < PAUSA_ARCHIVE:
        time.sleep(PAUSA_ARCHIVE - dt)
    _ultimo_archive[0] = time.time()
    ano = int(PADRAO_URL.search(url).group(2)) + 1
    try:
        r = requests.get(ARCHIVE.format(ano=ano, url=url), headers=UA, timeout=90)
    except Exception as e:
        return None, f"archive {type(e).__name__}"
    if r.status_code != 200:
        return None, f"archive HTTP {r.status_code}"
    html = r.content.decode("utf-8", "replace")
    if PAGINA_INDISPONIVEL in html:
        return None, "archive também indisponível"
    m = re.search(r"/web/(\d{14})", r.url)
    return html, (m.group(1) if m else None)


def chave(url):
    u = url.split("?")[0].split("#")[0]
    return re.sub(r"^https?://", "", u).rstrip("/").lower()


# ==========================================================================
# extração
# ==========================================================================
def extrair(html):
    """Título, lide e corpo, por cascata de rotas.

    Cascata em vez de um seletor só porque o site trocou de CMS entre 2020 e
    2021 -- foi o que congelou o sitemap -- e o acervo tem as duas gerações de
    marcação. A rota usada é gravada em cada registro: se uma delas produzir
    texto ruim, dá para isolar e refazer só aquela fatia, sem repetir 44 h.

    As meta tags Open Graph vêm primeiro porque são as mais estáveis: existem
    para o compartilhamento em rede social e sobrevivem a redesenho de tema."""
    s = BeautifulSoup(html, "html.parser")
    rotas = []

    def meta(prop, attr="property"):
        t = s.find("meta", attrs={attr: prop})
        return (t.get("content") or "").strip() if t else ""

    titulo = meta("og:title")
    if titulo:
        rotas.append("og:title")
    else:
        h1 = s.find("h1")
        titulo = h1.get_text(" ", strip=True) if h1 else ""
        rotas.append("h1" if titulo else "sem-titulo")

    lide = meta("og:description") or meta("description", "name")
    rotas.append("og:description" if lide else "sem-lide")

    # Data: a da página é mais confiável que a da URL, que só tem ano-mês.
    data = (meta("article:published_time") or
            meta("article:modified_time") or "")
    if not data:
        t = s.find("time")
        data = (t.get("datetime") or t.get_text(" ", strip=True)) if t else ""

    for x in s(["script", "style", "nav", "footer", "header", "aside", "form",
                "figure", "figcaption"]):
        x.decompose()

    corpo, rota_corpo = "", "nenhuma"
    # `conteudo-noticia` é o contêiner da matéria no tema atual, medido em
    # páginas de 2017 e 2022 -- o CMS atual serve o acervo antigo com o mesmo tema.
    for seletor, nome in (("div.conteudo-noticia", "conteudo-noticia"),
                          ("div[class*='field--name-body']", "drupal-body"),
                          ("article", "article"),
                          ("div[class*='post-content']", "post-content"),
                          ("div[class*='content']", "content")):
        alvo = s.select_one(seletor)
        if alvo:
            txt = " ".join(p.get_text(" ", strip=True) for p in alvo.find_all("p"))
            if len(txt) >= MIN_CHARS_CORPO:
                corpo, rota_corpo = txt, nome
                break
    if not corpo:
        txt = " ".join(p.get_text(" ", strip=True) for p in s.find_all("p"))
        if len(txt) >= MIN_CHARS_CORPO:
            corpo, rota_corpo = txt, "todos-os-p"
    rotas.append(f"corpo:{rota_corpo}")

    corpo = re.sub(r"\s{2,}", " ", corpo).strip()
    # Sem lide explícito, usa a primeira sentença do corpo -- é o que o lide
    # seria de qualquer forma no texto jornalístico.
    if not lide and corpo:
        m = re.search(r"^(.{80,400}?[.!?])\s", corpo)
        lide = m.group(1) if m else corpo[:300]
        rotas[1] = "lide-da-1a-sentenca"

    return {"titulo": titulo, "lide": lide.strip(), "conteudo_real": corpo,
            "data_pagina": data, "rotas_extracao": rotas}


# ==========================================================================
# montagem da fila
# ==========================================================================
def urls_do_sitemap():
    if os.path.exists(ARQ_SITEMAP):
        with open(ARQ_SITEMAP, encoding="utf-8") as f:
            return json.load(f)
    print("   lendo sitemap legado (2013-2020)...")
    raiz, erro = baixar(SITEMAP)
    if not raiz:
        print(f"   [!] sitemap indisponível: {erro}")
        return []
    filhos = [u for u in LOC.findall(raiz) if "sitemap" in u.lower()]
    todas, incompleto = [], False
    for i, f in enumerate(filhos, 1):
        txt, _ = baixar(f)
        if txt:
            todas.extend(LOC.findall(txt))
        else:
            incompleto = True
        print(f"      [{i}/{len(filhos)}] acumulado {len(todas):,}".replace(",", "."))
    # Só guarda o cache se veio inteiro: um cache parcial encolheria a oferta
    # de 2013-2020 em silêncio em todas as execuções seguintes.
    if not incompleto:
        with open(ARQ_SITEMAP, "w", encoding="utf-8") as f:
            json.dump(todas, f)
    return todas


def urls_do_cdx():
    if not os.path.exists(ARQ_CDX):
        print(f"   [!] {ARQ_CDX} não existe -- rode descobrir_cdx.py antes.")
        return []
    with open(ARQ_CDX, encoding="utf-8") as f:
        d = json.load(f)
    return [u for lista in d.values() for u in lista]


def demanda_das_falsas():
    with open(ARQ_FALSAS, encoding="utf-8") as f:
        dados = json.load(f)
    d = Counter()
    for r in dados:
        if (r.get("veredito") or "").upper() != "FALSO":
            continue
        b, dt = r.get("tema_pareamento"), r.get("data_publicacao")
        if b and dt:
            d[(b, int(dt[:4]))] += 1
    return d


def celula_da_url(url):
    m = PADRAO_URL.search(url)
    if not m:
        return None
    balde = taxonomia.pareamento_da_editoria(m.group(1))
    return (balde, int(m.group(2))) if balde else None


def montar_fila(dem, feitas, excluir):
    """Sorteia, por célula, o que falta entre demanda e oferta.

    O sorteio importa: pegar as primeiras da lista traria as mais antigas de
    cada mês ou as que o Archive visitou mais, e nenhuma das duas é uma amostra
    do que a agência publicou.

    Reprodutibilidade: cada célula é ordenada por URL canônica e embaralhada
    com semente própria, então a ordem de leitura do sitemap e da CDX não muda
    a seleção. Uma retomada percorre o MESMO ranking e só pula o que já foi
    feito ou descartado -- por isso a cota é demanda MENOS o já coletado, não
    a demanda inteira de novo.

    Devolve também as reservas de cada célula: quando uma matéria falha, a
    próxima do ranking entra no lugar, em vez de a célula ficar abaixo da cota."""
    brutas = urls_do_sitemap() + urls_do_cdx()
    print(f"   URLs brutas reunidas: {len(brutas):,}".replace(",", "."))

    por_celula = defaultdict(dict)
    for u in brutas:
        if NAO_E_PAGINA.search(u) or NAO_MATERIA.search(u):
            continue
        cel = celula_da_url(u)
        if cel:
            por_celula[cel].setdefault(chave(u), u)

    fila, reservas, deficit = [], {}, Counter()
    for celula, pedido in dem.items():
        falta = pedido - feitas.get(celula, 0)
        if falta <= 0:
            continue
        ranking = [por_celula[celula][k] for k in sorted(por_celula.get(celula, {}))]
        random.Random(f"{SEED}|{celula[0]}|{celula[1]}").shuffle(ranking)
        ranking = [u for u in ranking if chave(u) not in excluir]
        fila.extend(ranking[:falta])
        reservas[celula] = ranking[falta:]
        if len(ranking) < falta:
            deficit[celula] = falta - len(ranking)

    random.Random(SEED).shuffle(fila)   # mistura os anos, para uma interrupção
    return fila, reservas, deficit      # precoce deixar o corpus parcial representativo


def ler_jsonl(caminho):
    if not os.path.exists(caminho):
        return []
    saida = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                saida.append(json.loads(linha))
            except Exception:
                pass          # linha truncada por interrupção: ignora
    return saida


def adquirir_trava():
    """Duas instâncias somadas dobram a taxa real contra a Agência Brasil e
    quebram o Crawl-delay que a seção de coleta declara."""
    try:
        fd = os.open(ARQ_TRAVA, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        print(f"[!] Outra coleta parece estar rodando ({ARQ_TRAVA}).")
        print("    Se tiver certeza de que não há, apague o arquivo e rode de novo.")
        sys.exit(1)
    os.write(fd, f"{os.getpid()} {datetime.now().isoformat()}\n".encode())
    os.close(fd)


# ==========================================================================
def sondar(n):
    """Valida a extração em poucas matérias ANTES de comprometer dois dias."""
    dem = demanda_das_falsas()
    fila, _, _ = montar_fila(dem, Counter(), set())
    print(f"\nSONDA -- {n} matérias de {len(fila):,} na fila\n".replace(",", "."))
    for u in fila[:n]:
        html, erro = baixar(u)
        if not html:
            print(f"  [X] {erro}  {u[:90]}")
            continue
        d = extrair(html)
        print(f"  [{len(d['conteudo_real']):>5}c] {d['titulo'][:76]}")
        print(f"        rotas: {' | '.join(d['rotas_extracao'])}")
        print(f"        lide : {d['lide'][:130]}")
        print(f"        data : {d['data_pagina'][:25]}\n")
    print("Confira se título e lide estão corretos e se o corpo não trouxe menu")
    print("ou rodapé. Estando bom, rode com --limite 50 para cronometrar.")


def main():
    limite = int(sys.argv[sys.argv.index("--limite") + 1]) if "--limite" in sys.argv else None

    adquirir_trava()      # a sonda também acessa o site: vale a mesma regra
    try:
        if "--sondar" in sys.argv:
            sondar(int(sys.argv[sys.argv.index("--sondar") + 1]))
        else:
            coletar(limite)
    finally:
        os.remove(ARQ_TRAVA)


def coletar(limite):
    # Retomada: o que já foi gravado conta contra a cota da célula, e o que foi
    # descartado de vez não é tentado de novo.
    feitas, ja = Counter(), set()
    for r in ler_jsonl(ARQ_SAIDA):
        url = r.get("url_origem_real") or ""
        ja.add(chave(url))
        cel = celula_da_url(url)
        if cel:
            feitas[cel] += 1
    descartadas = {chave(r.get("url", "")) for r in ler_jsonl(ARQ_DESCARTES)}
    n_ja = len(ja)
    print(f"já coletadas: {n_ja:,} · descartadas antes: {len(descartadas):,}"
          .replace(",", "."))

    dem = demanda_das_falsas()
    fila, reservas, deficit = montar_fila(dem, feitas, ja | descartadas)
    print(f"   fila desta execução: {len(fila):,}".replace(",", "."))
    if deficit:
        print(f"\n   células sem oferta suficiente ({sum(deficit.values()):,} instâncias):"
              .replace(",", "."))
        for (b, a), n in sorted(deficit.items(), key=lambda x: -x[1])[:8]:
            print(f"      {b:22s} {a}  faltam {n}")
        print("   -> o corpus sai desbalanceado nessas células, por decisão.")

    if limite:
        fila = fila[:limite]
        print(f"   LIMITE ativo: {len(fila)} nesta execução.")
    if not fila:
        print("\nNada a coletar.")
        return

    print(f"\n   estimativa: {len(fila) * PAUSA / 3600:.1f} h a {PAUSA}s por matéria")
    print("   (Ctrl+C a qualquer momento: a retomada continua de onde parou)\n")

    saida = open(ARQ_SAIDA, "a", encoding="utf-8")
    desc = open(ARQ_DESCARTES, "a", encoding="utf-8")

    def gravar():
        # flush + fsync: sem isso o registro fica no buffer do SO e uma
        # queda de energia leva junto o que a tela já disse estar salvo.
        for arq in (saida, desc):
            arq.flush()
            os.fsync(arq.fileno())

    def repor(url):
        """Põe a próxima do ranking da mesma célula no lugar da que falhou."""
        cel = celula_da_url(url)
        if limite is None and cel and reservas.get(cel):
            fila.append(reservas[cel].pop(0))

    inicio = time.time()
    gravados = 0
    falhas, curtas, indisponiveis, via_archive = 0, 0, 0, 0
    i = 0
    interrompido = False
    try:
        while i < len(fila):
            url = fila[i]
            i += 1
            html, erro = baixar(url)
            if not html:
                falhas += 1
                print(f"[{i}/{len(fila)}] [X] {erro}")
                # Só 404/410 é descarte definitivo; erro de rede pode ser da
                # nossa conexão, e a retomada deve tentar de novo.
                if erro in ("HTTP 404", "HTTP 410"):
                    desc.write(json.dumps({"url": url, "motivo": erro}) + "\n")
                repor(url)
                continue
            via, snapshot = "site", None
            if PAGINA_INDISPONIVEL in html:
                html, snapshot = baixar_do_archive(url)
                if not html:
                    indisponiveis += 1
                    print(f"[{i}/{len(fila)}] [X] indisponível no site; {snapshot}")
                    desc.write(json.dumps({"url": url, "motivo": "indisponivel",
                                           "detalhe": snapshot}, ensure_ascii=False) + "\n")
                    repor(url)
                    continue
                via = "internet_archive"
                via_archive += 1
            d = extrair(html)
            if len(d["conteudo_real"]) < MIN_CHARS_CORPO or not d["titulo"]:
                curtas += 1
                print(f"[{i}/{len(fila)}] [!] corpo curto/sem título, descartada")
                desc.write(json.dumps({"url": url, "motivo": "corpo_curto", "via": via}) + "\n")
                repor(url)
                continue

            m = PADRAO_URL.search(url)
            editoria, ano, mes = m.group(1), m.group(2), m.group(3)
            data = ""
            if d["data_pagina"]:
                mm = re.search(r"(\d{4})-(\d{2})-(\d{2})", d["data_pagina"])
                if mm:
                    data = mm.group(0)
            if not data:
                data = f"{ano}-{mes}-01"      # da URL só sai ano-mês; dia 1 é convenção

            registro = {
                "titulo": d["titulo"],
                "lide": d["lide"],
                # Entrada CRUA do lado verdadeiro na RQ1. O recorte para
                # título+lide é feito no experimento, não aqui: guardar o texto
                # inteiro permite refazer o recorte sem recoletar.
                "conteudo_real": d["conteudo_real"],
                "ideia_central": None,        # preenchido depois, pelo refinamento
                "palavras_chave": [],
                "veredito": "VERDADEIRO",
                "idioma": "pt-BR",
                "url_origem_real": url,
                "fonte_real": VEICULO,
                "licenca": LICENCA,
                "editoria": editoria,
                "tema_pareamento": taxonomia.pareamento_da_editoria(editoria),
                "tema_pareamento_origem": "url_editoria",
                "data_publicacao": data,
                "data_origem": "pagina" if d["data_pagina"] else "url",
                "rotas_extracao": d["rotas_extracao"],
                # Proveniência do TEXTO: a URL é sempre a da agência, mas o
                # HTML pode ter vindo da cópia arquivada (ver PAGINA_INDISPONIVEL).
                "texto_via": via,
                "snapshot_archive": snapshot,
                "pareado": True,
            }
            saida.write(json.dumps(registro, ensure_ascii=False) + "\n")
            gravados += 1
            if gravados % SALVAR_A_CADA == 0:
                gravar()
                decorrido = time.time() - inicio
                resta = (len(fila) - i) * decorrido / i / 3600
                print(f"[{i}/{len(fila)}] [salvo] {n_ja + gravados:,} registros"
                      .replace(",", ".") + f"  | restam ~{resta:.1f} h")
    except KeyboardInterrupt:
        # Em JSONL uma interrupção perde no máximo a linha em escrita, que a
        # retomada ignora -- então aqui só garantimos que o resto vá ao disco.
        print("\n[interrompido] gravando o que já foi baixado...")
        interrompido = True
    finally:
        gravar()
        saida.close()
        desc.close()

    coletadas = ler_jsonl(ARQ_SAIDA)

    print(f"\n=== {'COLETA INTERROMPIDA' if interrompido else 'COLETA FINALIZADA'} ===")
    print(f"-> {len(coletadas):,} registros no total.".replace(",", "."))
    print(f"-> {falhas} falhas de rede/HTTP, {curtas} descartadas por corpo curto, "
          f"{indisponiveis} indisponíveis também no Archive.")
    print(f"-> nesta execução, {via_archive} textos vieram do Internet Archive.")
    print("-> texto_via (total):", dict(Counter(x.get("texto_via", "site") for x in coletadas)))
    print("-> rotas de extração usadas:")
    for r, n in Counter(x for c in coletadas for x in c["rotas_extracao"]).most_common(8):
        print(f"      {r:26s} {n:,}".replace(",", "."))
    print(f"\nArquivo: {ARQ_SAIDA}")
    print("Próximo: refinar (ideia_central) e fundir com o lado falso.")


if __name__ == "__main__":
    main()
