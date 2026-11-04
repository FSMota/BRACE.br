# -*- coding: utf-8 -*-
"""Descobre URLs históricas pela API CDX do Internet Archive.

Por que isto existe: o sitemap da Agência Brasil é legado e para em 2020, mas
54,5% da demanda está em 2021-2026. A única rota que restava era paginar a
listagem por seção -- 4.176 requisições, 11,6 h, num CMS que já devolveu
resposta inconsistente na página 301.

A CDX é um índice do que o Internet Archive já rastreou. Ela lista URLs por
prefixo e período sem tocar no site de origem, então a descoberta deixa de
depender da paginação e deixa de consumir a cota de educação do veículo. As URLs
da Agência Brasil trazem editoria e ano-mês, então continuam entregando tema e
data sem abrir a matéria.

O que ela NÃO faz: baixar a matéria. Isso segue vindo do site do veículo, com o
Crawl-delay respeitado -- são as 44 h que nenhuma rota de descoberta elimina.

    python scripts/descobrir_cdx.py --smoke        # teste de 2 minutos
    python scripts/descobrir_cdx.py                # varredura 2021-2026
"""
import os
import re
import sys
import json
import time
import requests
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taxonomia                                        # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_FINAL = os.path.join(_RAIZ, "datasets", "dataset_tcc_fake_news_final.json")
SAIDA = os.path.join(_RAIZ, "relatorios", "coleta", "urls_cdx.json")

CDX = "http://web.archive.org/cdx/search/cdx"
DOMINIO = "agenciabrasil.ebc.com.br"
UA = {"User-Agent": "Mozilla/5.0 (pesquisa academica TCC)"}

# 'geral' fora: 42 mil matérias que caem no balde "Outros", cuja demanda é 31.
#
# Os quatro últimos são CANDIDATOS ao slug da editoria de ciência. A primeira
# varredura zerou "Ciência e Tecnologia" nos seis anos -- 1.034 das 1.057
# células descobertas -- não por falta de oferta, mas porque nenhum slug de
# ciência estava na lista. O sitemap mostrou 'pesquisaeinovacao', então o slug
# existe; qual grafia a URL usa em cada época é o que estes candidatos medem.
# Os que voltarem zerados custam 6 consultas cada e resolvem a dúvida.
EDITORIAS = ["politica", "economia", "saude", "internacional", "justica",
             "meio-ambiente", "educacao", "cultura", "direitos-humanos", "esportes",
             "pesquisa-e-inovacao", "ciencia-e-tecnologia", "ciencia", "tecnologia"]
ANOS = list(range(2021, 2027))

PAUSA = 3          # a CDX é um serviço público e sem chave; não convém apertar
TIMEOUT = 180      # consultas de prefixo largo demoram
TENTATIVAS = 3

PADRAO_URL = re.compile(r"/([a-z\-]+)/noticia/(\d{4})-(\d{2})/")
# A CDX indexa TUDO que foi rastreado, inclusive a imagem que ilustra a matéria
# -- e ela mora sob o mesmo caminho, então casa com o padrão acima. Sem este
# filtro a oferta seria contada com o dobro do tamanho real.
NAO_E_PAGINA = re.compile(r"\.(jpg|jpeg|png|gif|webp|svg|pdf|mp3|mp4|zip|css|js)(\?|$)", re.I)


def chave_canonica(url):
    """Identidade da matéria, para deduplicar de verdade.

    O `collapse=urlkey` da CDX só funde linhas ADJACENTES de mesma chave, e a
    chave inclui query string -- então http/https, barra final e `?utm_...`
    escapam e a mesma matéria volta duas ou três vezes. No smoke test as URLs
    já vieram em pares. Sem esta normalização a oferta seria contada com o
    dobro do tamanho, e a decisão de coletar ou encolher sairia de um número
    inflado."""
    u = url.split("?")[0].split("#")[0]
    u = re.sub(r"^https?://", "", u).rstrip("/").lower()
    return u
_ultimo = [0.0]


def _aguardar():
    dt = time.time() - _ultimo[0]
    if dt < PAUSA:
        time.sleep(PAUSA - dt)
    _ultimo[0] = time.time()


def consultar(prefixo, limite=None):
    """Uma consulta CDX por prefixo de URL.

    `collapse=urlkey` remove as capturas repetidas da mesma página -- sem isso a
    mesma matéria volta uma vez por visita do rastreador, e a contagem infla em
    uma ordem de grandeza. `fl=original` pede só a URL: é tudo que precisamos,
    porque a data está dentro dela."""
    params = {"url": f"{DOMINIO}/{prefixo}", "matchType": "prefix",
              "output": "json", "fl": "original", "collapse": "urlkey",
              "filter": "statuscode:200"}
    if limite:
        params["limit"] = str(limite)
    for n in range(TENTATIVAS):
        _aguardar()
        try:
            r = requests.get(CDX, params=params, timeout=TIMEOUT, headers=UA)
        except Exception as e:
            if n == TENTATIVAS - 1:
                return None, f"{type(e).__name__}"
            time.sleep(10 * (n + 1))
            continue
        if r.status_code != 200:
            if n == TENTATIVAS - 1:
                return None, f"HTTP {r.status_code}"
            time.sleep(15 * (n + 1))
            continue
        txt = r.text.strip()
        if not txt:
            return [], None                     # vazio é resposta legítima
        try:
            linhas = json.loads(txt)
        except json.JSONDecodeError:
            return None, "resposta não-JSON"
        # A primeira linha é cabeçalho quando há resultado.
        return [l[0] for l in linhas[1:]] if len(linhas) > 1 else [], None
    return None, "esgotou tentativas"


def smoke():
    """Teste curto: a CDX responde para este domínio, e devolve matéria?"""
    print("SMOKE TEST -- uma editoria, um ano, 25 URLs\n")
    urls, erro = consultar("economia/noticia/2023", limite=25)
    if urls is None:
        print(f"   [X] FALHOU: {erro}")
        print("   A CDX não respondeu. Sem ela, resta a paginação (11,6 h).")
        return False
    if not urls:
        print("   [!] respondeu, mas sem resultados para este prefixo.")
        print("   Pode ser que o Archive não tenha rastreado essa seção.")
        return False
    print(f"   [OK] {len(urls)} URLs devolvidas. Exemplos:")
    for u in urls[:5]:
        print("      ", u[:105])
    casou = sum(1 for u in urls
                if PADRAO_URL.search(u) and not NAO_E_PAGINA.search(u))
    imagens = sum(1 for u in urls if NAO_E_PAGINA.search(u))
    if imagens:
        print(f"   ({imagens} eram imagem/anexo e foram descartadas)")
    unicas = len({chave_canonica(u) for u in urls if not NAO_E_PAGINA.search(u)})
    print(f"\n   casam com o padrão editoria/noticia/AAAA-MM: {casou}/{len(urls)}")
    print(f"   URLs ÚNICAS após normalizar: {unicas} "
          f"(duplicatas da CDX: {casou - unicas})")
    if casou < len(urls) * 0.5:
        print("   [!] taxa baixa -- o prefixo está trazendo página que não é matéria.")
    print("\n   Rode sem --smoke para a varredura completa de 2021-2026.")
    return True


def demanda():
    with open(ARQ_FINAL, encoding="utf-8") as f:
        dados = json.load(f)
    d = Counter()
    for r in dados:
        if (r.get("veredito") or "").upper() != "FALSO":
            continue
        b, dt = r.get("tema_pareamento"), r.get("data_publicacao")
        if b and dt and int(dt[:4]) in ANOS:
            d[(b, int(dt[:4]))] += 1
    return d


def main():
    if "--smoke" in sys.argv:
        smoke()
        return

    dem = demanda()
    total_dem = sum(dem.values())
    print("=" * 74)
    print(f"  DEMANDA {ANOS[0]}-{ANOS[-1]}: {total_dem:,} instâncias".replace(",", "."))
    print(f"  consultas a fazer: {len(EDITORIAS) * len(ANOS)} "
          f"(~{len(EDITORIAS) * len(ANOS) * PAUSA / 60:.0f} min a {PAUSA}s)")
    print("=" * 74)

    oferta, todas, falhas = Counter(), {}, []
    for ed in EDITORIAS:
        linha = []
        for ano in ANOS:
            urls, erro = consultar(f"{ed}/noticia/{ano}")
            if urls is None:
                falhas.append((ed, ano, erro))
                linha.append("ERRO")
                continue
            bons, vistas = [], set()
            for u in urls:
                if NAO_E_PAGINA.search(u):
                    continue
                m = PADRAO_URL.search(u)
                if not m:
                    continue
                ck = chave_canonica(u)
                if ck in vistas:
                    continue
                vistas.add(ck)
                balde = taxonomia.pareamento_da_editoria(m.group(1))
                if balde:
                    oferta[(balde, int(m.group(2)))] += 1
                    bons.append(u)
            brutas = len(urls)
            if brutas and len(bons) < brutas * 0.7:
                linha_extra = f" ({brutas} brutas -> {len(bons)} únicas)"
            else:
                linha_extra = ""
            todas[f"{ed}|{ano}"] = bons
            linha.append(str(len(bons)))
        print(f"  {ed:20s} " + "  ".join(f"{v:>7s}" for v in linha))
    print(f"  {'':20s} " + "  ".join(f"{a:>7d}" for a in ANOS))

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(todas, f, indent=1, ensure_ascii=False)

    print("\n" + "=" * 74)
    print("  OFERTA x DEMANDA POR CÉLULA (balde x ano)")
    print("=" * 74)
    buracos = []
    for (b, a), n in sorted(dem.items(), key=lambda x: -x[1]):
        o = oferta.get((b, a), 0)
        marca = "" if o >= n else f"   FALTAM {n - o}"
        if o < n:
            buracos.append((b, a, n - o))
        print(f"  {b:22s} {a}  pedido {n:>5,}  oferta {o:>6,}{marca}"
              .replace(",", "."))

    cobrivel = sum(min(n, oferta.get(k, 0)) for k, n in dem.items())
    print("\n" + "=" * 74)
    print(f"  pareável célula a célula em {ANOS[0]}-{ANOS[-1]}: {cobrivel:,}/{total_dem:,} "
          f"({100 * cobrivel / max(1, total_dem):.1f}%)".replace(",", "."))
    if falhas:
        print(f"  consultas que falharam: {len(falhas)} -> {falhas[:5]}")
    print(f"\n  URLs salvas em {SAIDA}")
    print("  Somadas às do sitemap (2014-2020), fecham o pareamento completo.")


if __name__ == "__main__":
    main()
