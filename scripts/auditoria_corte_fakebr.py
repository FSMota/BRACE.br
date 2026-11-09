# -*- coding: utf-8 -*-
"""O corte do Fake.br preserva a alegação falsa? (exploratório, posterior aos experimentos)

    python scripts/auditoria_corte_fakebr.py --planilha SAIDA.xlsx   # sorteia e gera a planilha
    python scripts/auditoria_corte_fakebr.py --apurar ANOTADA.xlsx   # conta as respostas

Na avaliação externa, cada texto do Fake.br foi cortado no primeiro fim de frase a
partir do 250º caractere (ou no 300º), nas duas classes. O rótulo do Fake.br vale para
a notícia inteira: se a parte falsa estiver depois do corte, o texto avaliado fica com
o rótulo falso sem conter a falsidade. Esta auditoria sorteia 50 notícias falsas e
mostra, para leitura humana, o trecho mantido, a parte cortada e o início da notícia
verdadeira pareada pelo próprio Fake.br (por semelhança de assunto, nem sempre o mesmo
fato), como apoio para identificar o que a falsa afirma de errado.
"""
import os
import re
import sys
import json
import random
import argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avaliacao_cruzada import carregar_fakebr, CORTE_MIN, CORTE_SEM_FIM   # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_auditoria_corte_fakebr.json")
N_AMOSTRA = 50
SEMENTE = 20261109
TAM_VERDADEIRA = 700
OPCOES = ["sim", "parcial", "não", "dúvida"]


def cortar(bruto):
    """Mesmo corte de avaliacao_cruzada.entrada_texto, mas devolve as duas partes
    sem a normalização de fonte, para leitura."""
    t = re.sub(r"[\r\n\t]+", " ", bruto.replace("﻿", "")).strip()
    t = re.sub(r" {2,}", " ", t)
    m = re.search(r"[.!?]", t[CORTE_MIN:])
    i = CORTE_MIN + m.end() if m else CORTE_SEM_FIM
    return t[:i].strip(), t[i:].strip()


def planilha(saida):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    pares, _ = carregar_fakebr()
    amostra = sorted(random.Random(SEMENTE).sample(pares, N_AMOSTRA), key=lambda p: p["n"])
    wb = Workbook()
    ins = wb.active
    ins.title = "instrucoes"
    linhas = [
        "Auditoria do corte do Fake.br",
        "",
        "Cada linha é uma notícia FALSA do Fake.br, sorteada (50 de 3.600, semente fixa).",
        "Coluna C: o trecho que os modelos receberam (corte no 1º fim de frase depois do 250º caractere).",
        "Coluna D: o restante da notícia, que foi cortado.",
        "Coluna E: o início da notícia VERDADEIRA que o Fake.br pareou com essa falsa.",
        "  O pareamento do Fake.br é por semelhança de assunto e nem sempre trata do mesmo fato;",
        "  use-a só como apoio para identificar o que a falsa afirma de errado.",
        "",
        "Pergunta (coluna F): a alegação falsa principal da notícia está no trecho mantido (coluna C)?",
        "  sim      - a afirmação falsa central já aparece no trecho mantido.",
        "  parcial  - o trecho mantido apresenta o assunto, mas a afirmação falsa central",
        "             só aparece, ou só se completa, na parte cortada.",
        "  não      - o trecho mantido não contém a afirmação falsa (só contexto, cabeçalho etc.).",
        "  dúvida   - não dá para decidir, por exemplo porque não se identifica o que é falso.",
        "",
        "Coluna G: comentário livre (opcional), útil nos casos parcial, não e dúvida.",
    ]
    for i, l in enumerate(linhas, 1):
        ins.cell(i, 1, l)
    ins["A1"].font = Font(bold=True, size=13)
    ins.column_dimensions["A"].width = 110

    ws = wb.create_sheet("anotacao")
    cab = ["id", "n_fakebr", "trecho mantido (entrada dos modelos)", "parte cortada",
           "início da verdadeira pareada", "alegação no trecho mantido?", "comentário"]
    ws.append(cab)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for k, p in enumerate(amostra, 1):
        mantido, cortado = cortar(p["fake"]["bruto"])
        verd = re.sub(r"\s+", " ", p["true"]["bruto"].replace("﻿", "")).strip()
        if len(verd) > TAM_VERDADEIRA:
            verd = verd[:TAM_VERDADEIRA].rsplit(" ", 1)[0] + " [...]"
        ws.append([k, p["n"], mantido, cortado or "(nada foi cortado)", verd, "", ""])
    larg = {"A": 5, "B": 9, "C": 45, "D": 70, "E": 50, "F": 16, "G": 30}
    for col, w in larg.items():
        ws.column_dimensions[col].width = w
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(wrap_text=True, vertical="top")
    dv = DataValidation(type="list", formula1='"' + ",".join(OPCOES) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"F2:F{N_AMOSTRA + 1}")
    ws.freeze_panes = "C2"
    wb.save(saida)
    print(f"planilha: {saida} ({N_AMOSTRA} notícias, semente {SEMENTE})")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (c - h, c + h)


def apurar(arq):
    from openpyxl import load_workbook
    ws = load_workbook(arq)["anotacao"]
    resp = [(r[0], r[1], (r[5] or "").strip().lower()) for r in ws.iter_rows(min_row=2, values_only=True) if r[0]]
    cont = Counter(x[2] or "(vazio)" for x in resp)
    validas = [x for x in resp if x[2] in ("sim", "parcial", "não")]
    n = len(validas)
    k = sum(x[2] == "sim" for x in validas)
    lo, hi = wilson(k, n)
    res = {"respostas": dict(cont), "decididas": n, "sim": k,
           "proporcao_sim": k / n if n else None, "ic95_wilson": [lo, hi],
           "nao_ou_parcial": [x[1] for x in validas if x[2] != "sim"]}
    json.dump(res, open(ARQ_SAIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--planilha")
    ap.add_argument("--apurar")
    a = ap.parse_args()
    if a.planilha:
        planilha(a.planilha)
    elif a.apurar:
        apurar(a.apurar)
    else:
        ap.print_help()
