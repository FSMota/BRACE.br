# -*- coding: utf-8 -*-
"""Atraso entre o post viral incorporado na checagem e a data da checagem (exploratório).
Pré-registro: `relatorios/PREREGISTRO_ATRASO_CHECAGEM.md`.

    python scripts/atraso_checagem.py --medir       # distribuição do atraso
    python scripts/atraso_checagem.py --planilha    # 30 artigos para validação humana, sem datas
    python scripts/atraso_checagem.py --apurar      # depois da validação

A data do post vem da assinatura do embed do X/Twitter dentro do bloco da
mensagem viral: "— Nome (@perfil) May 27, 2024". Usa-se a mais antiga do bloco.
"""
import os
import re
import sys
import json
import random
from datetime import date
from collections import Counter

import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_BRUTO = os.path.join(_RAIZ, "datasets", "dataset_bruto.json")
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
PASTA = os.path.join(_RAIZ, "relatorios", "atraso_checagem")
ARQ_MEDIDA = os.path.join(PASTA, "resultado_atraso.json")
ARQ_AMOSTRA = os.path.join(PASTA, "amostra_validacao.json")
ARQ_PLANILHA = os.path.join(PASTA, "planilha_validacao.xlsx")
MESES = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
         "October", "November", "December"]
EMBED = re.compile(r"\(@\w+\)\s+(" + "|".join(MESES) + r")\s+(\d{1,2}),\s+(20\d\d)")
CATEGORIAS = ["a_propria_mensagem", "outro_post", "nao_da_para_saber"]
N_VALIDACAO = 30
SEED = 0


def artigos():
    d = json.load(open(ARQ_BRUTO, encoding="utf-8"))
    d = d if isinstance(d, list) else list(d.values())
    datas = {}
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        r = json.loads(l)
        if r["veredito"] == "FALSO":
            datas.setdefault((r.get("url") or "").rstrip("/"), r["data_publicacao"])
    saida, sem_corpus = [], 0
    for a in d:
        if a.get("fonte") != "Boatos.org":
            continue
        viral = a.get("texto_viral_estrutural") or ""
        posts = [date(int(y), MESES.index(m) + 1, int(dd)) for m, dd, y in EMBED.findall(viral)]
        if not posts:
            continue
        chave = (a.get("url_origem") or "").rstrip("/")
        if chave not in datas:
            sem_corpus += 1
            continue
        checagem = date.fromisoformat(datas[chave][:10])
        saida.append({"url": a["url_origem"], "data_post": min(posts).isoformat(), "data_checagem": checagem.isoformat(),
                      "atraso_dias": (checagem - min(posts)).days, "n_posts": len(posts), "viral": viral})
    return saida, sem_corpus


def medir():
    arts, sem_corpus = artigos()
    a = np.array([x["atraso_dias"] for x in arts])
    res = {"n": len(arts), "sem_data_no_corpus": sem_corpus,
           "mediana": float(np.median(a)), "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75)),
           "pct_ate_7": round(100 * float((a <= 7).mean()), 1), "pct_ate_30": round(100 * float((a <= 30).mean()), 1),
           "pct_ate_365": round(100 * float((a <= 365).mean()), 1), "pct_negativo": round(100 * float((a < 0).mean()), 1),
           "pct_mesmo_dia_ou_seguinte": round(100 * float(((a >= 0) & (a <= 1)).mean()), 1),
           "faixas": dict(Counter("negativo" if x < 0 else "0-1" if x <= 1 else "2-7" if x <= 7 else "8-30" if x <= 30
                                  else "31-365" if x <= 365 else ">365" for x in a))}
    ok = res["mediana"] <= 30
    res["leitura"] = ("próxima da circulação na escala do pareamento" if ok and res["pct_ate_365"] >= 90 else
                      "perto na maioria, com cauda de recirculação" if ok else "não representa bem a circulação")
    res["confronto"] = {"T1": [res["mediana"], res["mediana"] <= 7], "T2": [res["pct_ate_30"], res["pct_ate_30"] >= 75],
                        "T3": [res["pct_ate_365"], res["pct_ate_365"] >= 90], "T4": [res["pct_negativo"], res["pct_negativo"] <= 10]}
    os.makedirs(PASTA, exist_ok=True)
    with open(ARQ_MEDIDA, "w", encoding="utf-8") as f:
        json.dump(res | {"itens": [{k: v for k, v in x.items() if k != "viral"} for x in arts]}, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def planilha():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    if os.path.exists(ARQ_PLANILHA) and "--sobrescrever" not in sys.argv:
        sys.exit(f"[!] {ARQ_PLANILHA} já existe -- pode ter anotações.")
    arts, _ = artigos()
    amostra = random.Random(SEED).sample(sorted(arts, key=lambda x: x["url"]), N_VALIDACAO)
    os.makedirs(PASTA, exist_ok=True)
    with open(ARQ_AMOSTRA, "w", encoding="utf-8") as f:
        json.dump([{"n": i, "url": x["url"]} for i, x in enumerate(amostra, start=1)], f, indent=1, ensure_ascii=False)
    wb = Workbook()
    ws = wb.active
    ws.title = "validacao"
    ws.append(["n", "link da checagem", "bloco da mensagem viral (sem datas)", "o post incorporado é…", "observacao"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for i, x in enumerate(amostra, start=1):
        sem_datas = EMBED.sub("(@perfil) [data oculta]", x["viral"])        # cegamento: nenhuma data visível
        ws.append([i, x["url"], sem_datas, None, None])
    for linha in ws.iter_rows(min_row=2):
        for c in linha:
            c.alignment = Alignment(wrap_text=c.column == 3, vertical="top")
    for col, larg in zip("ABCDE", (5, 40, 90, 22, 30)):
        ws.column_dimensions[col].width = larg
    dv = DataValidation(type="list", formula1='"' + ",".join(CATEGORIAS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"D2:D{N_VALIDACAO + 1}")
    wb.save(ARQ_PLANILHA)
    print(f"planilha de validação com {N_VALIDACAO} artigos -> {ARQ_PLANILHA}")


def apurar():
    from openpyxl import load_workbook
    med = {x["url"]: x for x in json.load(open(ARQ_MEDIDA, encoding="utf-8"))["itens"]}
    ws = load_workbook(ARQ_PLANILHA)["validacao"]
    linhas = [[c.value for c in r] for r in ws.iter_rows(min_row=2) if r[0].value not in (None, "")]
    if any(l[3] not in CATEGORIAS for l in linhas):
        sys.exit("há linhas sem categoria válida")
    cont = Counter(l[3] for l in linhas)
    proprias = np.array([med[l[1]]["atraso_dias"] for l in linhas if l[3] == "a_propria_mensagem"])
    res = {"contagem": dict(cont), "pct_propria_mensagem": round(100 * cont["a_propria_mensagem"] / len(linhas), 1),
           "atraso_nas_proprias": {"n": len(proprias), "mediana": float(np.median(proprias)) if len(proprias) else None,
                                   "pct_ate_30": round(100 * float((proprias <= 30).mean()), 1) if len(proprias) else None}}
    res["confronto_T5"] = [res["pct_propria_mensagem"], res["pct_propria_mensagem"] >= 60]

    # Desvio declarado (26/10/2026, decidido durante a anotação, antes de qualquer
    # resultado): em blocos com mais de um post, a data mais antiga pode ser a de
    # uma evidência anterior ao boato. Versão corrigida: usa a data do post que o
    # anotador indicou como a mensagem ("mensagem = post N" na observação); sem
    # indicação, assume o post 1 (suposição declarada).
    arts = {a["url"]: a for a in artigos()[0]}
    corrigidos, suposicoes, detalhes = [], 0, []
    for l in linhas:
        if l[3] != "a_propria_mensagem":
            continue
        a = arts[l[1]]
        posts = [date(int(y), MESES.index(m) + 1, int(dd)) for m, dd, y in EMBED.findall(a["viral"])]
        if len(posts) == 1:
            k = 1
        else:
            m = re.search(r"post\s*(\d+)", str(l[4] or ""), re.I)
            if m:
                k = int(m.group(1))
            elif "primeira parte" in str(l[4] or "").lower():
                k = 1
            else:
                k, suposicoes = 1, suposicoes + 1
        checagem = date.fromisoformat(a["data_checagem"])
        atraso = (checagem - posts[k - 1]).days
        corrigidos.append(atraso)
        if len(posts) > 1:
            detalhes.append({"n": l[0], "posts": len(posts), "post_usado": k, "atraso_original": a["atraso_dias"],
                             "atraso_corrigido": atraso})
    c = np.array(corrigidos)
    res["atraso_nas_proprias_corrigido"] = {
        "n": len(c), "mediana": float(np.median(c)), "pct_ate_7": round(100 * float((c <= 7).mean()), 1),
        "pct_ate_30": round(100 * float((c <= 30).mean()), 1), "pct_ate_365": round(100 * float((c <= 365).mean()), 1),
        "blocos_com_varios_posts": detalhes, "suposicoes_post_1": suposicoes}
    with open(os.path.join(PASTA, "resultado_validacao.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"--medir": medir, "--planilha": planilha, "--apurar": apurar}.get(
        next((a for a in sys.argv[1:] if a.startswith("--")), ""), lambda: print(__doc__))()
