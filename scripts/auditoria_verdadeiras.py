# -*- coding: utf-8 -*-
"""Mede o que dá para medir na classe verdadeira sem substituir o checador (exploratório).

    python scripts/auditoria_verdadeiras.py                     # as três medidas e o resultado
    python scripts/auditoria_verdadeiras.py --planilha SAIDA    # e a planilha de pares para leitura

O rótulo VERDADEIRO vem da fonte, e nenhuma matéria foi verificada uma a uma:
  1. correções publicadas pela própria agência, no corpo das matérias coletadas;
     é um piso: só conta os erros que ela admitiu, e a versão coletada já está corrigida;
  2. fala atribuída na entrada ("disse", "segundo"...): o que é verdadeiro é a fala ter
     ocorrido, e não necessariamente o seu conteúdo;
  3. verdadeiras do pareado parecidas com alguma falsa (TF-IDF, cosseno >= 0,3), para
     leitura humana: o checador desmentiu o que a agência publicou?
"""
import os
import re
import sys
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_VERD = os.path.join(_RAIZ, "datasets", "dataset_verdadeiras.jsonl")
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
MARCAS = os.path.join(_RAIZ, "relatorios", "marcacoes")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_auditoria_verdadeiras.json")
LIMIAR = 0.3
NOTA_CORRECAO = re.compile(
    r"(mat[ée]ria|texto|reportagem|not[ií]cia|informa[çc][ãa]o)\s+(foi\s+)?(alterad[ao]|corrigid[ao]|atualizad[ao]|retificad[ao])"
    r"[^.]{0,80}(corre[çc][ãa]o|corrigir|informa[çc][ãa]o|grafia|erro)"
    r"|\bdiferentemente\s+do\s+(que\s+foi\s+)?(informado|publicado|divulgado)"
    r"|\berrata\b|\bcorre[çc][ãa]o:|\btexto\s+corrigido\b|\bmat[ée]ria\s+corrigida\b", re.I)
CORRECAO = re.compile(r"corre[çc][ãa]o|corrigi|diferentemente|errata", re.I)   # sem isso, é só acréscimo ou ajuste
FALA = re.compile(r"\b(disse|diz|afirmou|afirma|declarou|segundo|de acordo com|defendeu|criticou|avaliou|informou|anunciou)\b", re.I)


def ids(nome):
    return {i["id"] for i in json.load(open(os.path.join(MARCAS, nome + ".json"), encoding="utf-8"))["itens"]}


def main():
    regs = [json.loads(l) for l in open(ARQ_VERD, encoding="utf-8") if l.strip()]
    correcoes = []
    for r in regs:
        t = r.get("conteudo_real") or ""
        m = NOTA_CORRECAO.search(t)
        if m:
            correcoes.append({"url": r["url_origem_real"], "pareado": bool(r.get("pareado")),
                              "trecho": t[max(0, m.start() - 60):m.end() + 160].replace("\n", " ")})
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    revisao = json.load(open(os.path.join(MARCAS, "revisao_rotulos.json"), encoding="utf-8"))["decisoes"]
    fora = ids("entrada_inventada") | {i for i, v in revisao.items() if v["decisao"] != "FALSO"}
    verd = [r for r in corpus if r["veredito"] == "VERDADEIRO" and r["subconjunto_pareado"]]
    fals = [r for r in corpus if r["veredito"] == "FALSO" and r["id"] not in fora]
    fala = sum(bool(FALA.search(r["texto_entrada"])) for r in verd)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vec.fit([r["texto_entrada"] for r in verd + fals])
    S = vec.transform([r["texto_entrada"] for r in verd]) @ vec.transform([r["texto_entrada"] for r in fals]).T
    mx = np.asarray(S.max(axis=1).todense()).ravel()
    arg = np.asarray(S.argmax(axis=1)).ravel()
    pares = [{"similaridade": round(float(mx[i]), 3), "verdadeira": verd[i], "falsa": fals[arg[i]]}
             for i in np.argsort(-mx) if mx[i] >= LIMIAR]
    reais = [c for c in correcoes if CORRECAO.search(c["trecho"])]
    res = {"notas_de_alteracao": len(correcoes),
           "correcoes": {"n": len(reais), "total": len(regs), "pct": round(100 * len(reais) / len(regs), 2), "casos": reais},
           "fala_atribuida": {"n": fala, "total": len(verd), "pct": round(100 * fala / len(verd), 1)},
           "pares_parecidos": {"limiar": LIMIAR, "n": len(pares),
                               "itens": [{"similaridade": p["similaridade"], "id_v": p["verdadeira"]["id"], "id_f": p["falsa"]["id"]} for p in pares]}}
    json.dump(res, open(ARQ_SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"notas de alteração: {len(correcoes)} | correções: {len(reais)} de {len(regs)} ({res['correcoes']['pct']}%)")
    print(f"fala atribuída: {fala} de {len(verd)} ({res['fala_atribuida']['pct']}%)")
    print(f"pares com similaridade >= {LIMIAR}: {len(pares)}")
    if "--planilha" in sys.argv:
        planilha(pares, sys.argv[sys.argv.index("--planilha") + 1])


def planilha(pares, saida):
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ws = wb.active
    ws.title = "Pares"
    cab = ["N", "Similaridade", "Verdadeira (Agência Brasil)", "Link da matéria", "Falsa (mensagem desmentida)",
           "Título da checagem", "Link da checagem", "Relação", "Observação"]
    for k, (c, w) in enumerate(zip(cab, (5, 11, 60, 30, 60, 42, 30, 26, 34)), 1):
        x = ws.cell(row=1, column=k, value=c)
        x.font = Font(bold=True, color="FFFFFF")
        x.fill = PatternFill("solid", fgColor="1D5C69")
        ws.column_dimensions[x.column_letter].width = w
    for k, p in enumerate(pares, 1):
        v, f = p["verdadeira"], p["falsa"]
        ws.append([k, p["similaridade"], v["texto_entrada"], v.get("url"), f["texto_entrada"],
                   f.get("titulo_artigo_checagem"), f.get("url"), "", ""])
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)
    ws.freeze_panes = "C2"
    dv = DataValidation(type="list", allow_blank=True,
                        formula1='"checador desmente a materia,mesmo fato: detalhe difere,so mesmo assunto,duvida"')
    ws.add_data_validation(dv)
    dv.add(f"H2:H{ws.max_row}")
    wb.save(saida)
    print("planilha:", saida)


if __name__ == "__main__":
    main()
