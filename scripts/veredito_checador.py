# -*- coding: utf-8 -*-
"""Lê o veredito que o próprio checador publicou em cada artigo da classe falsa.

    python scripts/veredito_checador.py              # gera relatórios/marcacoes/veredito_checador.json
    python scripts/veredito_checador.py --planilha   # e também a planilha de revisão

Por que existe: o rótulo FALSO é herdado do artigo de checagem. Este script
substitui a varredura de títulos (marcar_rotulo_duvidoso.py) por uma leitura
do campo que o checador usa para dar o veredito, sempre que ele existe.

Fontes do veredito, em ordem de confiança:
  1. selo: o Boatos.org fecha os artigos recentes com um selo de veredito
     ("Fake news ❌", "Golpe ⚠", "Enganoso 😈", "Exagerado 🗣", "Real com erros 🤔",
     "Boato sem comprovação 👎", "Em apuração ❓"...). É o veredito do checador.
  2. hashtag: G1 (#FAKE / #FATO) e os títulos antigos do Boatos.org (#boato).
  3. título: os artigos antigos do Boatos.org não têm selo; o título é a única
     indicação do veredito. Classificado por regra, com os casos sem marca de
     desmentido listados para revisão humana.

Mapeamento para a classe da base:
  FALSO .......... Fake news, Falso, Golpe, Promoção falsa, Boato sem comprovação, #boato, #FAKE
  MEIA_VERDADE ... Enganoso, Exagerado, Real com erros
  NAO_VERIFICADO . Em apuração
  REVISAR ........ sem selo nem hashtag e sem marca de desmentido no título

Conteúdo que não é checagem (patrocinado) não tem marca no texto do Boatos.org;
não há como detectá-lo por regra. Os casos conhecidos entram em REVISAR quando
não têm selo; os que têm selo de "Fake news" ficam como o checador publicou.
"""
import os
import re
import sys
import json
from collections import Counter

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "marcacoes", "veredito_checador.json")
ARQ_PLANILHA = os.path.join(_RAIZ, "relatorios", "marcacoes", "veredito_checador_revisar.xlsx")

SELOS = {  # rótulo do selo (minúsculas) -> classe
    "fake news": "FALSO", "falso": "FALSO", "golpe": "FALSO", "promoção falsa": "FALSO",
    "boato sem comprovação": "FALSO",
    "enganoso": "MEIA_VERDADE", "exagerado": "MEIA_VERDADE", "real com erros": "MEIA_VERDADE",
    "em apuração": "NAO_VERIFICADO",
}
EMO = r"[☀-➿\U0001F300-\U0001FAFF]"
RE_SELO = re.compile(r"(" + "|".join(re.escape(s) for s in sorted(SELOS, key=len, reverse=True)) + r")\s*️?\s*" + EMO, re.I)
RE_HASHTAG = re.compile(r"#(boato|fake|fato)\b", re.I)
DESMENTE = re.compile(
    r"\bboatos?\b|fals[oa]s?\b|falsamente|fake|mentir|não é (real|verdade|verídic)|nao e verdade|\bnão\b|\bnem\b|\bnunca\b|"
    r"montage|montad|manipulad|editad|adulterad|golpe|invent|\bia\b|inteligência artificial|encena|simula|sátira|piada|"
    r"descontextualiz|fora de contexto|engan|distorc|falseia|deturp|sem provas|sem fundamento|absurd|lend[áa]|teoria|"
    r"conspira|exager|requentad|volta a circular|ressurge|desmentid|leviana|errad|erro|equivoc|irreal|fictíci|alarm|"
    r"balela|lorota|hoax|pegadinha|farsa|falácia|mito|vírus|não procede|caça-clique|carece|sem lastro|ilação",
    re.I)


def veredito(r):
    """Devolve (classe, fonte_do_veredito, texto_do_veredito)."""
    texto = (r.get("texto_bruto_jornalista") or "").split(" Ps:")[0]
    selos = list(RE_SELO.finditer(texto[-800:]))
    if selos:
        s = selos[-1].group(1)
        return SELOS[s.lower()], "selo", s
    titulo = r.get("titulo_artigo_checagem") or ""
    h = RE_HASHTAG.findall(titulo)
    if h:
        tag = h[-1].lower()
        return ("VERDADEIRO_CHECADOR" if tag == "fato" else "FALSO"), "hashtag", "#" + h[-1]
    if DESMENTE.search(titulo):
        return "FALSO", "titulo", titulo
    return "REVISAR", "nenhuma", titulo


def main():
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    itens = []
    for r in corpus:
        if r["veredito"] != "FALSO":
            continue
        classe, fonte, txt = veredito(r)
        itens.append({"id": r["id"], "classe": classe, "fonte_do_veredito": fonte, "veredito_checador": txt,
                      "fonte": r["fonte"], "titulo_checagem": r.get("titulo_artigo_checagem"), "url": r.get("url"),
                      "split": r["split"], "subconjunto_pareado": r["subconjunto_pareado"],
                      "texto_entrada": r["texto_entrada"]})
    cont = Counter((i["classe"], i["fonte_do_veredito"]) for i in itens)
    selos = Counter(i["veredito_checador"] for i in itens if i["fonte_do_veredito"] == "selo")
    artigos = Counter(i["fonte_do_veredito"] for i in {i["url"]: i for i in itens}.values())
    res = {"n_falsas": len(itens), "artigos_por_fonte_do_veredito": dict(artigos),
           "por_classe_e_fonte": {f"{a}|{b}": n for (a, b), n in sorted(cont.items())},
           "selos": dict(selos.most_common()),
           "itens": [{k: v for k, v in i.items() if k != "texto_entrada"} for i in itens]}
    os.makedirs(os.path.dirname(ARQ_SAIDA), exist_ok=True)
    json.dump(res, open(ARQ_SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{len(itens)} falsas | artigos por fonte do veredito: {dict(artigos)}")
    for (a, b), n in sorted(cont.items()):
        print(f"  {a:20s} {b:8s} {n:6d}")
    print("selos:", dict(selos.most_common()))
    if "--planilha" in sys.argv:
        planilha([i for i in itens if i["classe"] != "FALSO"])


def planilha(itens):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "Revisar"
    cols = ["Classe proposta", "Fonte do veredito", "Veredito do checador", "Título da checagem",
            "Texto da entrada", "Link", "Parte", "No pareado", "Id", "Decisão", "Observação"]
    larg = [18, 14, 22, 48, 70, 40, 10, 10, 16, 18, 30]
    ws.append(cols)
    for k, c in enumerate(ws[1]):
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="1D5C69")
        ws.column_dimensions[c.column_letter].width = larg[k]
    ordem = {"NAO_VERIFICADO": 0, "MEIA_VERDADE": 1, "VERDADEIRO_CHECADOR": 2, "REVISAR": 3}
    for i in sorted(itens, key=lambda i: (ordem.get(i["classe"], 9), i["titulo_checagem"] or "")):
        ws.append([i["classe"], i["fonte_do_veredito"],
                   i["veredito_checador"] if i["fonte_do_veredito"] != "nenhuma" else "",
                   i["titulo_checagem"], i["texto_entrada"], i["url"], i["split"],
                   "sim" if i["subconjunto_pareado"] else "não", i["id"], "", ""])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = ws.dimensions
    dv = DataValidation(type="list", allow_blank=True,
                        formula1='"FALSO,MEIA_VERDADE,NAO_VERIFICADO,NAO_E_CHECAGEM,remover"')
    ws.add_data_validation(dv)
    dv.add(f"J2:J{ws.max_row}")
    wb.save(ARQ_PLANILHA)
    print("planilha:", ARQ_PLANILHA, f"({len(itens)} linhas)")


if __name__ == "__main__":
    main()
