# -*- coding: utf-8 -*-
"""Auditoria dirigida da via html em 2016 e 2026: sorteia a amostra e gera a planilha.

    python scripts/anotacao_html_epoca.py --sortear SAIDA.xlsx
    python scripts/anotacao_html_epoca.py --apurar PLANILHA.xlsx

Motivo: 2016 e 2026 concentram quase metade da via html, a de mais erros na anotação
das 200 falsas, e essa anotação não olhou esses anos separadamente. Sorteia 50
entradas de cada ano, sem as já lidas naquela anotação. A planilha embaralha os
dois anos e não mostra ano nem id; a chave fica em
relatorios/anotacao_html/amostra.json, que não deve ser aberta antes do fim da
leitura. Categorias e definições são as da anotação das 200 falsas.
"""
import os
import sys
import json
import math
import random
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anotacao_manual import CATEGORIAS, DEFINICOES, NAO_BOATO                 # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_LIDAS = os.path.join(_RAIZ, "relatorios", "anotacao", "amostra_meta.json")
PASTA = os.path.join(_RAIZ, "relatorios", "anotacao_html")
ARQ_AMOSTRA = os.path.join(PASTA, "amostra.json")
ARQ_RESULTADO = os.path.join(PASTA, "resultado.json")
ANOS, N_POR_ANO, SEED = ("2016", "2026"), 50, 20261106


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - m), 1), round(100 * (c + m), 1)]


def sortear(saida):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    assert not os.path.exists(ARQ_AMOSTRA), "a amostra já existe: não sortear de novo"
    meta = json.load(open(ARQ_LIDAS, encoding="utf-8"))
    lidas = {i["id"] if isinstance(i, dict) else i for i in (meta.get("itens") or meta.get("amostra"))}
    pop = {a: [] for a in ANOS}
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        a = (r.get("data_publicacao") or "")[:4]
        if r["veredito"] == "FALSO" and r.get("origem_viral") == "html" and a in pop and r["id"] not in lidas:
            pop[a].append(r)
    rng = random.Random(SEED)
    linhas = []
    for a in ANOS:
        linhas += [(a, r) for r in rng.sample(sorted(pop[a], key=lambda x: x["id"]), N_POR_ANO)]
    rng.shuffle(linhas)
    os.makedirs(PASTA, exist_ok=True)
    json.dump({"semente": SEED, "populacao": {a: len(v) for a, v in pop.items()}, "n_por_ano": N_POR_ANO,
               "itens": [{"n": k + 1, "id": r["id"], "ano": a, "url": r.get("url")} for k, (a, r) in enumerate(linhas)]},
              open(ARQ_AMOSTRA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    wb = Workbook()
    ws = wb.active
    ws.title = "Anotação"
    cab = ["N", "Texto da entrada", "Link do artigo", "Categoria", "Observação"]
    for k, (c, w) in enumerate(zip(cab, (6, 90, 45, 16, 40)), 1):
        x = ws.cell(row=1, column=k, value=c)
        x.font = Font(bold=True, color="FFFFFF")
        x.fill = PatternFill("solid", fgColor="1D5C69")
        ws.column_dimensions[x.column_letter].width = w
    for k, (a, r) in enumerate(linhas, 1):
        ws.append([k, r["texto_entrada"], r.get("url"), "", ""])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "A2"
    dv = DataValidation(type="list", formula1='"' + ",".join(CATEGORIAS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"D2:D{ws.max_row}")
    d = wb.create_sheet("Categorias")
    d.append(["Categoria", "Definição", "Exemplo"])
    for c in d[1]:
        c.font = Font(bold=True)
    for linha in DEFINICOES:
        d.append(list(linha))
    d.column_dimensions["A"].width = 14
    d.column_dimensions["B"].width = 90
    d.column_dimensions["C"].width = 70
    for row in d.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
    wb.save(saida)
    print("população:", {a: len(v) for a, v in pop.items()}, "| planilha:", saida)


def apurar(caminho):
    from openpyxl import load_workbook
    ws = load_workbook(caminho)["Anotação"]
    cat = {row[0].value: (row[3].value or "").strip().lower() for row in ws.iter_rows(min_row=2)}
    am = json.load(open(ARQ_AMOSTRA, encoding="utf-8"))
    faltam = [n for n, c in cat.items() if not c]
    assert not faltam, f"linhas sem categoria: {faltam}"
    res = {"por_ano": {}}
    for a in ANOS:
        itens = [i for i in am["itens"] if i["ano"] == a]
        cont = Counter(cat[i["n"]] for i in itens)
        k = sum(cont[c] for c in NAO_BOATO)
        n = len(itens) - cont["duvida"]
        res["por_ano"][a] = {"categorias": dict(cont), "nao_e_a_mensagem": k, "n_validos": n,
                             "pct": round(100 * k / n, 1), "ic95": wilson(k, n)}
        print(a, res["por_ano"][a])
    json.dump(res, open(ARQ_RESULTADO, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", ARQ_RESULTADO)


if __name__ == "__main__":
    if "--sortear" in sys.argv:
        sortear(sys.argv[sys.argv.index("--sortear") + 1])
    elif "--apurar" in sys.argv:
        apurar(sys.argv[sys.argv.index("--apurar") + 1])
    else:
        print(__doc__)
