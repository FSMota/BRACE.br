# -*- coding: utf-8 -*-
"""Pareamento datado × não datado (controle temporal).
Pré-registro: `relatorios/PREREGISTRO_PAREAMENTO_DATADO.md`.

    python scripts/pareamento_datado.py --sortear    # sorteia falsas, variante D e o ranking da variante N
    python scripts/pareamento_datado.py --coletar    # coleta as verdadeiras de N que faltam (retomável, ~10 s cada)

Regra final da variante N (fixada aqui, antes da coleta): para cada tema, as
primeiras URLs do ranking, na ordem, que estejam no corpus ou na coleta nova,
até completar o pedido do tema. Erro de rede passageiro não descarta a URL: a
retomada tenta de novo, e a regra final continua sendo a ordem do ranking.

A análise (modelos e medidas T1–T5) fica num modo à parte, commitado antes de
rodar, depois que a coleta terminar.

Nada aqui escreve em `dataset_verdadeiras.jsonl` nem no corpus: as verdadeiras
novas vão para `datasets/verdadeiras_nao_datadas.jsonl`, e a base do TCC não muda.
"""
import os
import re
import sys
import json
import random
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coletar_verdadeiras as cv                                # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
PASTA = os.path.join(_RAIZ, "relatorios", "pareamento_datado")
ARQ_AMOSTRA = os.path.join(PASTA, "amostra.json")
ARQ_COLETA = os.path.join(_RAIZ, "datasets", "verdadeiras_nao_datadas.jsonl")
ARQ_DESCARTES = os.path.join(_RAIZ, "datasets", "verdadeiras_nao_datadas_descartes.jsonl")
ANOS = range(2014, 2021)
N_FALSAS = 2500
SEED = 0
PARTES = (("treino", 0.8), ("validacao", 0.1), ("teste", 0.1))


def ano(r):
    d = (r.get("data_publicacao") or "")[:4]
    return int(d) if d.isdigit() else None


def url_verdadeira(r):
    return r.get("url_origem_real") or r.get("url") or ""


def dividir(ids, semente):
    """80/10/10 por ordem de sorteio: devolve {id: parte}."""
    ids = sorted(ids)
    random.Random(semente).shuffle(ids)
    n, saida, ini = len(ids), {}, 0
    for i, (parte, frac) in enumerate(PARTES):
        fim = n if i == len(PARTES) - 1 else ini + round(frac * n)
        for x in ids[ini:fim]:
            saida[x] = parte
        ini = fim
    return saida


def sortear():
    if os.path.exists(ARQ_AMOSTRA):
        sys.exit(f"{ARQ_AMOSTRA} já existe: o sorteio é feito uma vez só.")
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    falsas = sorted((r for r in corpus if r["veredito"] == "FALSO" and r["subconjunto_pareado"] and ano(r) in ANOS),
                    key=lambda r: r["id"])
    verd_par = [r for r in corpus if r["veredito"] == "VERDADEIRO" and r["subconjunto_pareado"] and ano(r) in ANOS]
    print(f"falsas pareadas 2014–2020: {len(falsas)} | verdadeiras pareadas 2014–2020: {len(verd_par)}")

    escolhidas = random.Random(SEED).sample(falsas, N_FALSAS)
    parte_falsa = dividir([r["id"] for r in escolhidas], f"{SEED}|falsas")

    # Variante D: mesma célula tema × ano.
    por_celula = defaultdict(list)
    for r in verd_par:
        por_celula[(r["tema_pareamento"], ano(r))].append(r["id"])
    pedido_d = Counter((r["tema_pareamento"], ano(r)) for r in escolhidas)
    ids_d = []
    for cel in sorted(pedido_d, key=str):
        cands = sorted(por_celula[cel])
        if len(cands) < pedido_d[cel]:
            sys.exit(f"célula {cel} sem verdadeiras suficientes ({len(cands)} < {pedido_d[cel]})")
        ids_d += random.Random(f"{SEED}|D|{cel[0]}|{cel[1]}").sample(cands, pedido_d[cel])
    parte_d = dividir(ids_d, f"{SEED}|D")

    # Variante N: por tema, sem olhar o ano, entre as URLs do sitemap de 2014–2020.
    por_tema = defaultdict(dict)
    for u in cv.urls_do_sitemap():
        if cv.NAO_E_PAGINA.search(u) or cv.NAO_MATERIA.search(u):
            continue
        cel = cv.celula_da_url(u)
        if cel and cel[1] in ANOS:
            por_tema[cel[0]].setdefault(cv.chave(u), u)
    pedido_n = Counter(r["tema_pareamento"] for r in escolhidas)
    url_para_id = {cv.chave(url_verdadeira(r)): r["id"] for r in corpus if r["veredito"] == "VERDADEIRO"}
    rankings = {}
    for tema in sorted(pedido_n):
        ranking = [por_tema[tema][k] for k in sorted(por_tema.get(tema, {}))]
        random.Random(f"{SEED}|N|{tema}").shuffle(ranking)
        if len(ranking) < pedido_n[tema]:
            sys.exit(f"tema {tema} sem URLs suficientes ({len(ranking)} < {pedido_n[tema]})")
        rankings[tema] = ranking

    reaproveitadas = sum(cv.chave(u) in url_para_id for t, rk in rankings.items() for u in rk[:pedido_n[t]])
    anos_n = Counter(cv.celula_da_url(u)[1] for t, rk in rankings.items() for u in rk[:pedido_n[t]])
    os.makedirs(PASTA, exist_ok=True)
    amostra = {
        "semente": SEED, "anos": [min(ANOS), max(ANOS)], "n_falsas": N_FALSAS,
        "falsas": {r["id"]: parte_falsa[r["id"]] for r in escolhidas},
        "D_verdadeiras": parte_d,
        "N_pedido_por_tema": dict(pedido_n),
        "N_rankings": rankings,
        "N_reaproveitaveis_entre_as_primeiras": reaproveitadas,
        "N_anos_das_primeiras": dict(sorted(anos_n.items())),
        "D_anos": dict(sorted(Counter(ano(r) for r in escolhidas).items())),
        "regra_N": "para cada tema, percorre o ranking: URL já no corpus é reaproveitada; as outras são "
                   "coletadas; falha é substituída pela próxima do ranking, até completar o pedido do tema",
    }
    with open(ARQ_AMOSTRA, "w", encoding="utf-8") as f:
        json.dump(amostra, f, ensure_ascii=False, indent=1)
    print(f"falsas: {N_FALSAS} | partes {dict(Counter(parte_falsa.values()))}")
    print(f"D: {len(ids_d)} verdadeiras | anos das falsas (= de D): {amostra['D_anos']}")
    print(f"N: pedido por tema {dict(pedido_n)}")
    print(f"N: anos das primeiras do ranking: {amostra['N_anos_das_primeiras']}")
    print(f"N: {reaproveitadas} já estão no corpus; a coletar ≈ {N_FALSAS - reaproveitadas} "
          f"(~{(N_FALSAS - reaproveitadas) * cv.PAUSA / 3600:.1f} h, sem contar substituições)")
    print(f"salvo em {ARQ_AMOSTRA}")


def coletar():
    """Percorre os rankings de N; coleta o que não está no corpus nem foi coletado/descartado aqui."""
    amostra = json.load(open(ARQ_AMOSTRA, encoding="utf-8"))
    corpus_urls = {cv.chave(url_verdadeira(json.loads(l))) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()}
    feitas = {cv.chave(r["url_origem_real"]) for r in cv.ler_jsonl(ARQ_COLETA)}
    descartadas = {cv.chave(r["url"]) for r in cv.ler_jsonl(ARQ_DESCARTES)} | \
                  {cv.chave(r.get("url", "")) for r in cv.ler_jsonl(cv.ARQ_DESCARTES)}
    saida, desc = open(ARQ_COLETA, "a", encoding="utf-8"), open(ARQ_DESCARTES, "a", encoding="utf-8")
    try:
        for tema, ranking in amostra["N_rankings"].items():
            pedido, obtidas = amostra["N_pedido_por_tema"][tema], 0
            for url in ranking:
                if obtidas >= pedido:
                    break
                k = cv.chave(url)
                if k in corpus_urls or k in feitas:
                    obtidas += 1
                    continue
                if k in descartadas:
                    continue
                html, erro = cv.baixar(url)
                via, snapshot = "site", None
                if html and cv.PAGINA_INDISPONIVEL in html:
                    html, snapshot = cv.baixar_do_archive(url)
                    via = "internet_archive"
                if not html:
                    if erro in ("HTTP 404", "HTTP 410") or via == "internet_archive":
                        desc.write(json.dumps({"url": url, "motivo": erro or "indisponivel"}) + "\n")
                    print(f"[{tema}] [X] {erro or snapshot}")
                    continue
                d = cv.extrair(html)
                if len(d["conteudo_real"]) < cv.MIN_CHARS_CORPO or not d["titulo"]:
                    desc.write(json.dumps({"url": url, "motivo": "corpo_curto", "via": via}) + "\n")
                    continue
                m = cv.PADRAO_URL.search(url)
                editoria = m.group(1)
                data = f"{m.group(2)}-{m.group(3)}-01"
                if d["data_pagina"]:
                    mm = re.search(r"(\d{4})-(\d{2})-(\d{2})", d["data_pagina"])
                    data = mm.group(0) if mm else data
                saida.write(json.dumps({
                    "titulo": d["titulo"], "lide": d["lide"], "conteudo_real": d["conteudo_real"],
                    "veredito": "VERDADEIRO", "url_origem_real": url, "fonte_real": cv.VEICULO,
                    "licenca": cv.LICENCA, "editoria": editoria,
                    "tema_pareamento": cv.taxonomia.pareamento_da_editoria(editoria),
                    "data_publicacao": data, "data_origem": "pagina" if d["data_pagina"] else "url",
                    "rotas_extracao": d["rotas_extracao"], "texto_via": via, "snapshot_archive": snapshot,
                    "variante": "N"}, ensure_ascii=False) + "\n")
                saida.flush()
                os.fsync(saida.fileno())
                feitas.add(k)
                obtidas += 1
                print(f"[{tema}] {obtidas}/{pedido}")
            if obtidas < pedido:
                print(f"[!] {tema}: ranking esgotado com {obtidas}/{pedido}")
    except KeyboardInterrupt:
        print("\n[interrompido] a retomada continua de onde parou")
    finally:
        for f in (saida, desc):
            f.flush()
            os.fsync(f.fileno())
            f.close()
    print(f"coletadas nesta base: {len(cv.ler_jsonl(ARQ_COLETA))}")


if __name__ == "__main__":
    if "--sortear" in sys.argv:
        sortear()
    elif "--coletar" in sys.argv:
        coletar()
    else:
        print(__doc__)
