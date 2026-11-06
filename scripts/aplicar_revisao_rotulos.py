# -*- coding: utf-8 -*-
"""Converte a revisão humana dos rótulos (planilha) em decisões por id.

    python scripts/aplicar_revisao_rotulos.py CAMINHO/veredito_checador_revisar.xlsx

Entrada: a planilha gerada por veredito_checador.py --planilha, com a coluna "Decisão" preenchida pelo pesquisador (05/11/2026), que abriu cada artigo no portal e localizou o veredito no texto do checador. Saída:
relatorios/marcacoes/revisao_rotulos.json, lida por exportar_publicacao.py.

Regras fixadas com o orientador:
  - o veredito vem do checador: quando há selo ou veredito explícito do
    checador, ele prevalece sobre a leitura do pesquisador (CHECADOR_PREVALECE);
  - meias-verdades (selos Enganoso, Exagerado, Real com erros) ficam FALSO;
  - mensagens verdadeiras e não verificadas saem da base;
  - conteúdo publicado no formato de checagem sem veredito vira NAO_E_CHECAGEM.
"""
import os
import sys
import json
from collections import Counter

from openpyxl import load_workbook

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "marcacoes", "revisao_rotulos.json")
MAPA = {"falso": "FALSO", "verdadeiro": "REMOVER", "nao_verificado": "REMOVER",
        "nao_e_checagem": "NAO_E_CHECAGEM", "remover": "REMOVER", "meia_verdade": "FALSO"}
# conflitos entre a planilha e o veredito explícito do checador: vale o checador
CHECADOR_PREVALECE = {
    "Fa0c288abac83": ("FALSO", "artigo abre com 'Boato –' (bebê do Ceará)"),
    "F583782a1bbb8": ("FALSO", "entrada traz o veredito do checador: 'Anitta é bi – Boato'"),
    "F974af4649497": ("REMOVER", "selo 'Em apuração ❓': o checador não deu veredito"),
}


def main():
    ws = load_workbook(sys.argv[1]).active
    cab = [c.value for c in ws[1]]
    decisoes = {}
    for linha in ws.iter_rows(min_row=2):
        d = dict(zip(cab, [c.value for c in linha]))
        bruto = (d["Decisão"] or "").strip()
        if not bruto:
            continue
        decisoes[d["Id"]] = {"decisao": MAPA[bruto.lower()], "planilha": bruto,
                             "observacao": d.get("Observação"), "origem": "pesquisador"}
    for i, (dec, motivo) in CHECADOR_PREVALECE.items():
        decisoes[i] = {"decisao": dec, "planilha": decisoes.get(i, {}).get("planilha"),
                       "observacao": motivo, "origem": "checador"}
    json.dump({"fonte": os.path.basename(sys.argv[1]), "n": len(decisoes),
               "por_decisao": dict(Counter(v["decisao"] for v in decisoes.values())),
               "decisoes": decisoes}, open(ARQ_SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(len(decisoes), "decisões ->", ARQ_SAIDA, dict(Counter(v["decisao"] for v in decisoes.values())))


if __name__ == "__main__":
    main()
