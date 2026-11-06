# -*- coding: utf-8 -*-
"""Busca sinais de que a data de publicação de um artigo de checagem é posterior à original.

    python scripts/datas_republicadas.py

Exploratório, só leitura de datasets/corpus_fundido.jsonl. Grava
relatorios/resultado_datas_republicadas.json. Três sinais, por artigo do Boatos.org:

  S1 lote: muitos artigos com a mesma data exata. Republicação em massa (migração
     de site) deixa picos num único dia.
  S2 defasagem: o ano mais recente citado no texto é pelo menos 2 anos anterior à
     data da página. É só um indício: um artigo novo pode tratar de um caso antigo.
  S3 publicação incorporada: o texto traz a assinatura de um tweet com data (como
     "November 24, 2023"), e a data mais recente é pelo menos 1 ano anterior à
     página. É o sinal mais direto, mas só existe em parte dos artigos.
"""
import os
import re
import json
import statistics
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_datas_republicadas.json")
RE_ANO = re.compile(r"\b(20[0-2]\d)\b")
MESES = "January|February|March|April|May|June|July|August|September|October|November|December"
RE_TWEET = re.compile(r"\((?:@\w+)\)\s+(?:" + MESES + r")\s+\d{1,2},\s+(20\d\d)")
RE_DATA_PAGINA = re.compile(r"\b\d{2}/\d{2}/20\d\d\b")
LIMIAR_LOTE = 10


def main():
    arts = {}
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["veredito"] == "FALSO" and r["fonte"] == "Boatos.org" and len(r.get("data_publicacao") or "") >= 10:
            arts.setdefault(r["url"], r)
    por_dia = Counter(r["data_publicacao"][:10] for r in arts.values())
    lotes = {d: n for d, n in por_dia.items() if n >= LIMIAR_LOTE}
    por_ano = defaultdict(lambda: Counter())
    casos = []
    for url, r in arts.items():
        ano = int(r["data_publicacao"][:4])
        texto = (r.get("texto_bruto_jornalista") or "").split(" Ps:")[0]
        texto = RE_DATA_PAGINA.sub(" ", texto)          # a data da própria página vem no texto
        anos = [int(a) for a in RE_ANO.findall(texto) if int(a) <= ano]
        tw = [int(a) for a in RE_TWEET.findall(texto)]
        c = por_ano[ano]
        c["artigos"] += 1
        c["em_lote"] += r["data_publicacao"][:10] in lotes
        if anos:
            c["cita_ano"] += 1
            if max(anos) <= ano - 2:
                c["defasagem"] += 1
                casos.append({"url": url, "titulo": r.get("titulo_artigo_checagem"), "data": r["data_publicacao"][:10],
                              "ano_mais_recente": max(anos), "sinal": "defasagem"})
        if tw:
            c["com_tweet"] += 1
            if max(tw) <= ano - 1:
                c["tweet_antigo"] += 1
                casos.append({"url": url, "titulo": r.get("titulo_artigo_checagem"), "data": r["data_publicacao"][:10],
                              "ano_mais_recente": max(tw), "sinal": "tweet"})
    print(f"artigos: {len(arts)} | mediana por dia: {statistics.median(por_dia.values())} | dias com >= {LIMIAR_LOTE} artigos: {len(lotes)}")
    print("maiores lotes:", sorted(lotes.items(), key=lambda x: -x[1])[:15])
    em_lote = sum(n for n in lotes.values())
    print(f"artigos em lotes: {em_lote} ({100 * em_lote / len(arts):.1f}%)")
    # só a defasagem cruza um corte da divisão se a data real for de outra parte
    for corte in ("2023-07-25", "2024-08-05"):
        depois = [d for d in lotes if corte < d <= f"{int(corte[:4]) + 1}{corte[4:]}"]
        print(f"lotes até 1 ano depois do corte {corte}: {len(depois)} dias, {sum(lotes[d] for d in depois)} artigos")
    print("\nano  artigos em_lote  cita_ano defasagem  %def  com_tweet tweet_antigo")
    for a in sorted(por_ano):
        c = por_ano[a]
        pct = 100 * c["defasagem"] / c["cita_ano"] if c["cita_ano"] else 0
        print(f"{a} {c['artigos']:7d} {c['em_lote']:7d} {c['cita_ano']:9d} {c['defasagem']:9d} {pct:5.1f} {c['com_tweet']:10d} {c['tweet_antigo']:12d}")
    json.dump({"artigos": len(arts), "limiar_lote": LIMIAR_LOTE, "lotes": lotes,
               "por_ano": {str(a): dict(c) for a, c in sorted(por_ano.items())}, "casos": casos},
              open(ARQ_SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", ARQ_SAIDA)


if __name__ == "__main__":
    main()
