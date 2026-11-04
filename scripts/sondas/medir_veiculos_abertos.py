# -*- coding: utf-8 -*-
"""Inventaria os sitemaps de veículos de licença aberta e compara com a demanda.

O que responde: existe oferta de notícia verdadeira, por assunto E por ano, para
parear as ~15,8 mil instâncias falsas? E onde ela falta?

Por que sitemap e não paginação: a listagem por seção devolve 13 matérias por
página, então chegar a 2013 custaria milhares de requisições com Crawl-delay de
10s. O índice de sitemap entrega centenas de milhares de URLs em ~16 requisições
-- e cada URL já traz a editoria e o ano-mês, então tema e data saem sem abrir a
matéria. Mesmo princípio que resolveu o tema inflado no lado falso: ler o rótulo
que a fonte já publica em vez de inferir.

NÃO baixa matéria e NÃO grava no dataset. Só lê sitemaps e conta.

    python scripts/medir_veiculos_abertos.py
    python scripts/medir_veiculos_abertos.py --veiculo "Agência Brasil"
"""
import os
import re
import io
import sys
import gzip
import json
import time
import requests
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import taxonomia                                        # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARQ_FINAL = os.path.join(_RAIZ, "datasets", "dataset_tcc_fake_news_final.json")
SAIDA = os.path.join(_RAIZ, "relatorios", "coleta", "oferta_veiculos_abertos.json")

UA = {"User-Agent": "Mozilla/5.0 (pesquisa academica TCC; contato via GitHub)"}
PAUSA_PADRAO = 10          # o robots.txt da Agência Brasil pede Crawl-delay: 10
TIMEOUT = 60
MAX_SITEMAPS = 40          # trava de segurança contra índice gigante

# --------------------------------------------------------------------------
# Veículos de licença aberta ou de órgão público -- a restrição não é de
# qualidade, é JURÍDICA: redistribuir texto integral de notícia protegida num
# corpus público do Zenodo não se sustenta. Agência Brasil é CC-BY; agências
# legislativas são conteúdo público de reprodução permitida.
#
# `padrao` extrai (editoria, ano, mes) da URL. Onde a editoria não existe na URL
# -- caso das agências legislativas, que são monotemáticas -- `editoria_fixa`
# assume. Isso é honesto: a fonte É de um assunto só.
#
# Os padrões da Agência Brasil foram verificados; os demais são HIPÓTESE. Quando
# não casam, o script despeja exemplos de URL para o padrão ser corrigido -- que
# é como os seletores do Boatos.org e da Lupa foram acertados.
# --------------------------------------------------------------------------
VEICULOS = [
    {"nome": "Agência Brasil", "base": "https://agenciabrasil.ebc.com.br",
     "sitemap": "https://agenciabrasil.ebc.com.br/sitemap.xml",
     "licenca": "CC BY 3.0 BR",
     "padrao": r"/([a-z\-]+)/noticia/(\d{4})-(\d{2})/",
     "editoria_fixa": None,
     # O sitemap da Agência Brasil é legado: para em 2020. A listagem por
     # seção, porém, serve conteúdo atual e pagina fundo (verificado:
     # /economia?page=300 devolve 2024-04). É a única rota para 2021-2026.
     # 'geral' fica de fora de propósito: são 42 mil matérias que caem no balde
     # "Outros", cuja demanda é 31 -- paginar isso seria gastar um terço do
     # esforço pelo balde menos necessário.
     "paginacao": {"padrao": "{base}/{editoria}?page={n}",
                   "editorias": ["politica", "economia", "saude", "internacional",
                                 "justica", "meio-ambiente", "educacao", "cultura",
                                 "direitos-humanos", "esportes"]}},
    {"nome": "Agência Senado", "base": "https://www12.senado.leg.br",
     "sitemap": "https://www12.senado.leg.br/sitemap.xml.gz",
     "licenca": "conteúdo público (Lei 9.610 art. 8º)",
     "padrao": r"/noticias/materias/(\d{4})/(\d{2})/",
     "editoria_fixa": "politica"},
    # A Câmara NÃO declara sitemap no robots.txt e /sitemap.xml dá 404. Fica
    # desativada: sem índice, só sobraria paginação, e ela é monotemática
    # (Política) -- balde que o Senado já cobre. Não vale o custo.
    {"nome": "Agência Câmara", "base": "https://www.camara.leg.br",
     "sitemap": None, "ativa": False,
     "licenca": "conteúdo público",
     "padrao": r"/noticias/(\d{4})/(\d{2})/",
     "editoria_fixa": "politica"},
    {"nome": "Agência Gov", "base": "https://agenciagov.ebc.com.br",
     "sitemap": "https://agenciagov.ebc.com.br/sitemap.xml.gz",
     "licenca": "conteúdo público",
     "padrao": r"/([a-z\-]+)/(\d{4})-(\d{2})/",
     "editoria_fixa": None},
]

# URLs que NÃO são matéria. O sitemap da Agência Brasil mistura galeria de foto,
# áudio e vídeo com notícia -- contar tudo junto superestimaria a oferta.
NAO_MATERIA = re.compile(r"/(foto|fotos|audio|audios|video|videos|galeria|tag|autor"
                         r"|infografico|radioagencia)/", re.I)

_ultimo = defaultdict(float)


def _aguardar(host, pausa):
    dt = time.time() - _ultimo[host]
    if dt < pausa:
        time.sleep(pausa - dt)
    _ultimo[host] = time.time()


def baixar(url, pausa):
    host = url.split("/")[2]
    _aguardar(host, pausa)
    try:
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
    except Exception as e:
        return None, f"{type(e).__name__}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    dados = r.content
    # Sitemap comprimido é comum; alguns servidores não marcam o Content-Type.
    if url.endswith(".gz") or dados[:2] == b"\x1f\x8b":
        try:
            dados = gzip.decompress(dados)
        except Exception:
            pass
    return dados.decode("utf-8", "ignore"), None


def crawl_delay(base):
    """Respeita o Crawl-delay declarado. Não é gentileza: é o que a seção de
    coleta do TCC vai declarar, e é o que evita bloqueio no meio da coleta."""
    txt, _ = baixar(base.rstrip("/") + "/robots.txt", 1)
    if not txt:
        return PAUSA_PADRAO
    m = re.search(r"(?im)^\s*crawl-delay:\s*(\d+)", txt)
    return max(int(m.group(1)), 1) if m else PAUSA_PADRAO


LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.I)


def coletar_urls(veiculo, pausa):
    """Segue o índice de sitemap até as URLs finais. Devolve (urls, erros)."""
    raiz, erro = baixar(veiculo["sitemap"], pausa)
    if not raiz:
        return [], [f"sitemap raiz: {erro}"]

    locs = LOC.findall(raiz)
    erros = []
    # Índice de sitemap aponta para outros sitemaps; sitemap comum já traz as
    # URLs das páginas. A distinção é pela tag, mas na prática basta ver se os
    # alvos também são sitemaps.
    filhos = [u for u in locs if "sitemap" in u.lower()]
    if not filhos:
        return [u for u in locs], erros

    print(f"   índice com {len(filhos)} sitemaps filhos "
          f"(~{min(len(filhos), MAX_SITEMAPS) * pausa / 60:.0f} min a {pausa}s)")
    urls = []
    for i, f in enumerate(filhos[:MAX_SITEMAPS], 1):
        txt, erro = baixar(f, pausa)
        if not txt:
            erros.append(f"{f}: {erro}")
            continue
        achadas = LOC.findall(txt)
        urls.extend(achadas)
        print(f"      [{i}/{min(len(filhos), MAX_SITEMAPS)}] {len(achadas):,} URLs"
              .replace(",", "."))
    return urls, erros


def classificar(urls, veiculo):
    """Extrai (balde de pareamento, ano) de cada URL. Devolve contagens e o que
    não casou, para o padrão poder ser corrigido."""
    padrao = re.compile(veiculo["padrao"])
    fixa = veiculo.get("editoria_fixa")
    matriz = Counter()
    editorias = Counter()
    nao_casou, nao_materia, sem_balde = [], 0, Counter()

    for u in urls:
        if NAO_MATERIA.search(u):
            nao_materia += 1
            continue
        m = padrao.search(u)
        if not m:
            if len(nao_casou) < 8:
                nao_casou.append(u)
            continue
        g = m.groups()
        if fixa:
            editoria, ano = fixa, g[0]
        else:
            editoria, ano = g[0], g[1]
        if not (ano.isdigit() and 2000 < int(ano) < 2100):
            continue
        editorias[editoria] += 1
        balde = taxonomia.pareamento_da_editoria(editoria)
        if not balde:
            # Editoria desconhecida é DESCARTADA, não jogada em "Outros":
            # absorver encheria o balde genérico com qualquer coisa.
            sem_balde[editoria] += 1
            continue
        matriz[(balde, int(ano))] += 1
    return matriz, editorias, nao_casou, nao_materia, sem_balde


def sondar_paginacao(veiculo, pausa, amostra, ate_ano):
    """Mede o CUSTO da paginação antes de pagar por ele.

    Paginar a Agência Brasil inteira até 2021 são milhares de requisições a 10s.
    Em vez de descobrir isso depois de meia noite rodando, busca-se uma amostra
    curta por editoria, vê-se a que data ela chegou, e extrapola-se linearmente
    quantas páginas faltariam. Erro de extrapolação existe -- o ritmo de
    publicação varia ao longo dos anos -- mas a ordem de grandeza basta para
    decidir entre paginar e encolher o conjunto."""
    cfg = veiculo.get("paginacao")
    if not cfg:
        print("   (sem paginação configurada)")
        return {}
    padrao = re.compile(veiculo["padrao"])
    print(f"\n   SONDA DE PAGINAÇÃO -- {amostra} páginas por editoria, alvo {ate_ano}")
    print(f"   {'editoria':20s} {'URLs':>6s} {'mais antiga':>12s} {'pág/ano':>8s} {'até ' + str(ate_ano):>12s}")
    plano = {}
    for ed in cfg["editorias"]:
        datas, urls_ed = [], []
        for n in range(1, amostra + 1):
            url = cfg["padrao"].format(base=veiculo["base"], editoria=ed, n=n)
            txt, err = baixar(url, pausa)
            if not txt:
                break
            achadas = [u for u in re.findall(r'href="([^"]+)"', txt)
                       if padrao.search(u) and not NAO_MATERIA.search(u)]
            urls_ed.extend(achadas)
            for u in achadas:
                m = padrao.search(u)
                if m:
                    datas.append(f"{m.group(2)}-{m.group(3)}")
        if not datas:
            print(f"   {ed:20s} {'0':>6s} {'-':>12s} {'-':>8s} {'falhou':>12s}")
            continue
        antiga = min(datas)
        ano_ant, mes_ant = int(antiga[:4]), int(antiga[5:7])
        # meses percorridos pela amostra -> páginas por ano
        from datetime import date
        hoje = date.today()
        meses = max(1, (hoje.year - ano_ant) * 12 + (hoje.month - mes_ant))
        pag_ano = amostra * 12 / meses
        faltam = int(pag_ano * max(0, hoje.year - ate_ano + 1))
        plano[ed] = faltam
        print(f"   {ed:20s} {len(urls_ed):>6d} {antiga:>12s} {pag_ano:>8.0f} {faltam:>12,}"
              .replace(",", "."))
    if plano:
        tot = sum(plano.values())
        print(f"\n   TOTAL estimado: {tot:,} páginas".replace(",", ".") +
              f"  ->  {tot * pausa / 3600:.1f} h só de descoberta, a {pausa}s")
    return plano


def demanda_do_corpus():
    with open(ARQ_FINAL, encoding="utf-8") as f:
        dataset = json.load(f)
    d = Counter()
    sem_campo = 0
    for r in dataset:
        if (r.get("veredito") or "").upper() != "FALSO":
            continue
        balde = r.get("tema_pareamento")
        data = r.get("data_publicacao")
        if not balde or not data:
            sem_campo += 1
            continue
        d[(balde, int(data[:4]))] += 1
    return d, sem_campo


def main():
    alvo = None
    if "--veiculo" in sys.argv:
        alvo = sys.argv[sys.argv.index("--veiculo") + 1]
    paginar = "--paginar" in sys.argv
    amostra = int(sys.argv[sys.argv.index("--amostra") + 1]) if "--amostra" in sys.argv else 15
    ate_ano = int(sys.argv[sys.argv.index("--ate") + 1]) if "--ate" in sys.argv else 2021

    if paginar:
        # Modo separado: não inventaria sitemap, só estima o custo da paginação.
        for v in VEICULOS:
            if alvo and v["nome"] != alvo:
                continue
            if not v.get("paginacao"):
                continue
            print(f"\n{'=' * 78}\nVEÍCULO: {v['nome']}\n{'=' * 78}")
            sondar_paginacao(v, crawl_delay(v["base"]), amostra, ate_ano)
        return

    demanda, sem_campo = demanda_do_corpus()
    total_dem = sum(demanda.values())
    print("=" * 78)
    print(f"  DEMANDA: {total_dem:,} instâncias falsas com balde e data"
          .replace(",", "."))
    if sem_campo:
        print(f"  ({sem_campo} sem tema_pareamento ou data -- rode corrigir_corpus.py)")
    print("=" * 78)

    oferta_total = Counter()
    relatorio = []
    for v in VEICULOS:
        if alvo and v["nome"] != alvo:
            continue
        if v.get("ativa") is False and not alvo:
            print(f"\n[ pulado ] {v['nome']}: {v.get('motivo', 'sem sitemap disponível')}")
            continue
        print(f"\n{'=' * 78}\nVEÍCULO: {v['nome']}   [{v['licenca']}]\n{'=' * 78}")
        pausa = crawl_delay(v["base"])
        print(f"   Crawl-delay respeitado: {pausa}s")
        urls, erros = coletar_urls(v, pausa)
        if not urls:
            print(f"   [X] nenhuma URL. {erros}")
            relatorio.append({"veiculo": v["nome"], "erro": erros})
            continue

        matriz, editorias, nao_casou, nao_materia, sem_balde = classificar(urls, v)
        casadas = sum(matriz.values())
        print(f"   URLs no sitemap ......... {len(urls):,}".replace(",", "."))
        print(f"   descartadas (foto/vídeo). {nao_materia:,}".replace(",", "."))
        print(f"   com editoria e ano ...... {casadas:,}".replace(",", "."))
        if editorias:
            print("   editorias:", dict(editorias.most_common(8)))
        if sem_balde:
            print("   editorias sem balde (descartadas):", dict(sem_balde.most_common(6)))
        if nao_casou:
            print(f"   [!] {len(nao_casou)} exemplos que não casaram com o padrão:")
            for u in nao_casou[:5]:
                print("       ", u[:110])
            print("       -> me mande estas linhas que eu corrijo o padrão.")
        if casadas:
            anos = sorted({a for _, a in matriz})
            print(f"   alcance temporal: {anos[0]} a {anos[-1]}")
        oferta_total.update(matriz)
        relatorio.append({"veiculo": v["nome"], "licenca": v["licenca"],
                          "urls": len(urls), "materias": casadas,
                          "editorias": dict(editorias),
                          "sem_balde": dict(sem_balde),
                          "nao_casou_exemplos": nao_casou,
                          "matriz": {f"{b}|{a}": n for (b, a), n in matriz.items()}})

    # --- confronto oferta x demanda ---------------------------------------
    print("\n" + "=" * 78)
    print("  OFERTA x DEMANDA POR ANO")
    print("=" * 78)
    print(f"  {'ano':>5s} {'demanda':>9s} {'oferta':>9s} {'cobertura':>10s}")
    anos = sorted({a for _, a in demanda} | {a for _, a in oferta_total})
    falta_ano = []
    for ano in anos:
        d = sum(n for (b, a), n in demanda.items() if a == ano)
        o = sum(n for (b, a), n in oferta_total.items() if a == ano)
        cob = 100 * o / d if d else float("inf")
        marca = "" if (not d or o >= d) else "  <-- FALTA"
        if d and o < d:
            falta_ano.append((ano, d - o))
        print(f"  {ano:>5d} {d:>9,} {o:>9,} {cob:>9.0f}%{marca}"
              .replace(",", "."))

    print("\n" + "=" * 78)
    print("  CÉLULAS DESCOBERTAS (balde x ano onde a oferta não fecha)")
    print("=" * 78)
    buracos = [((b, a), n - oferta_total.get((b, a), 0))
               for (b, a), n in demanda.items()
               if oferta_total.get((b, a), 0) < n]
    buracos.sort(key=lambda x: -x[1])
    if not buracos:
        print("  nenhuma: a oferta cobre toda a demanda.")
    for (b, a), falta in buracos[:20]:
        print(f"  {b:22s} {a}  faltam {falta:,}".replace(",", "."))
    if len(buracos) > 20:
        print(f"  ... e mais {len(buracos) - 20} células")

    total_of = sum(oferta_total.values())
    cobrivel = sum(min(n, oferta_total.get(k, 0)) for k, n in demanda.items())
    print("\n" + "=" * 78)
    print(f"  demanda total ........... {total_dem:,}".replace(",", "."))
    print(f"  oferta total ............ {total_of:,}".replace(",", "."))
    print(f"  pareável célula a célula. {cobrivel:,} "
          f"({100 * cobrivel / max(1, total_dem):.1f}%)".replace(",", "."))
    print("\n  Leitura: 'oferta total' maior que a demanda não basta -- o que")
    print("  importa é a última linha, porque o pareamento é célula a célula.")
    print("  Se ela ficar bem abaixo de 100%, ou acrescentamos veículos, ou o")
    print("  conjunto balanceado encolhe para o período/assunto que fecha.")

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump({"demanda": {f"{b}|{a}": n for (b, a), n in demanda.items()},
                   "veiculos": relatorio}, f, indent=2, ensure_ascii=False)
    print(f"\n  Detalhe em {SAIDA}")


if __name__ == "__main__":
    main()
