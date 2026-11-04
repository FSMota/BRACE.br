# -*- coding: utf-8 -*-
"""Apuração das afirmações A1–A8 da matriz de modelos.
Pré-registro: `relatorios/PREREGISTRO_MATRIZ_MODELOS.md` (§2 e §3).

    python scripts/apurar_matriz.py

Lê as predições gravadas em `relatorios/matriz/pred_<rep>_<clf>_<braço>.npz`
(e `pred_Ba_<braço>.npz`, do BERTimbau ajustado). Funciona com a matriz
incompleta: as contagens saem como "X de N disponíveis", e a leitura do §3
("vale em X de 13") só é aplicada quando os 13 modelos estiverem presentes.

Critérios por modelo (§2):
  A1  bruto − refinado no Fake.br (nossa direção) > 0, McNemar exato, Holm < 0,05
  A2  o mesmo no nosso teste
  A3  bruto: nosso teste − Fake.br ≥ 10 pp
  A4  rótulos embaralhados entre 45% e 55% (controle refeito, 3 sementes; o original também é relatado)
  A5  P+LR bruto no Fake.br a até 3 pp da mediana dos modelos
  A6  bruto: Fake.br → nosso teste entre 65% e 80%
  A7  bruto − refinado em Fake.br → nosso teste > 0, McNemar exato, Holm < 0,05
  A8  bruto: (Fake.br → nosso) − (nosso → Fake.br) > 0
Holm: sobre os testes de cada afirmação, entre os modelos disponíveis.
"""
import os
import sys
import json
import glob

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparativo_bracos import bal, pct, mcnemar, holm                     # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
PASTA = os.path.join(_RAIZ, "relatorios", "matriz")
ARQ_SAIDA = os.path.join(PASTA, "apuracao.json")
N_TOTAL = 13
ALFA = 0.05


def leitura(x, n):
    if n < N_TOTAL:
        return f"parcial ({n} de {N_TOTAL} modelos): leitura do §3 só com os 13"
    return "robusta" if x >= 10 else "dependente do modelo" if x >= 5 else "não se sustenta"


def main():
    rot = {}
    for l in open(ARQ_DADOS, encoding="utf-8"):
        r = json.loads(l)
        rot[r["id"]] = 1 if r["veredito"] == "FALSO" else 0
    modelos = {}
    for caminho in sorted(glob.glob(os.path.join(PASTA, "pred_*_bruto.npz"))):
        nome = os.path.basename(caminho)[5:-10]                      # pred_<nome>_bruto.npz
        ref = caminho.replace("_bruto.npz", "_refinado.npz")
        if os.path.exists(ref):
            modelos[nome] = (np.load(caminho), np.load(ref))
    controles = {}
    for c in glob.glob(os.path.join(PASTA, "controle_*.json")):
        controles.update(json.load(open(c, encoding="utf-8")))
    print(f"modelos com os dois braços: {len(modelos)} de {N_TOTAL}: {', '.join(modelos)}")

    linhas, testes = {}, {"A1": [], "A2": [], "A7": []}
    for nome, (b, r) in modelos.items():
        y_te = np.array([rot[i] for i in b["teste_ids"]])
        y_fb = np.array([rot[i] for i in b["fakebr_ids"]])
        assert list(b["teste_ids"]) == list(r["teste_ids"]) and list(b["fakebr_ids"]) == list(r["fakebr_ids"])
        L = {"nosso_teste": pct(bal(y_te, b["nosso_teste"])), "nosso_fakebr": pct(bal(y_fb, b["nosso_fakebr"])),
             "fakebr_nosso": pct(bal(y_te, b["fakebr_nosso"])),
             "ref_nosso_teste": pct(bal(y_te, r["nosso_teste"])), "ref_nosso_fakebr": pct(bal(y_fb, r["nosso_fakebr"])),
             "ref_fakebr_nosso": pct(bal(y_te, r["fakebr_nosso"]))}
        if "fakebr_fakebr" in b.files:
            L["fakebr_fakebr"] = pct(bal(y_fb, b["fakebr_fakebr"]))
        for a, chave, y in (("A1", "nosso_fakebr", y_fb), ("A2", "nosso_teste", y_te), ("A7", "fakebr_nosso", y_te)):
            so_b, so_r, p = mcnemar(b[chave] == y, r[chave] == y)
            L[a] = {"dif_pp": round(L[chave] - L["ref_" + chave], 2), "so_bruto": so_b, "so_refinado": so_r, "p": p}
            testes[a].append((nome, p))
        L["A3"] = {"queda_pp": round(L["nosso_teste"] - L["nosso_fakebr"], 2)}
        L["A3"]["vale"] = L["A3"]["queda_pp"] >= 10
        L["A6"] = {"valor": L["fakebr_nosso"], "vale": 65 <= L["fakebr_nosso"] <= 80}
        L["A8"] = {"dif_pp": round(L["fakebr_nosso"] - L["nosso_fakebr"], 2)}
        L["A8"]["vale"] = L["A8"]["dif_pp"] > 0
        ctl = controles.get(f"{nome.replace('_', '+')}|bruto")
        L["A4"] = ({"acuracias": ctl["acuracias"], "vale": ctl["dentro_45_55"]} if ctl else {"vale": None})
        if "embaralhado_teste" in b.files:
            L["A4"]["original_semente0_grade_na_validacao"] = pct(bal(y_te, b["embaralhado_teste"]))
        linhas[nome] = L

    for a, lista in testes.items():
        corr = holm([p for _, p in lista]) if lista else []
        for (nome, _), pc in zip(lista, corr):
            linhas[nome][a]["p_holm"] = pc
            linhas[nome][a]["vale"] = linhas[nome][a]["dif_pp"] > 0 and pc < ALFA

    mediana = float(np.median([L["nosso_fakebr"] for L in linhas.values()])) if linhas else None
    if "P_LR" in linhas:
        linhas["P_LR"]["A5"] = {"mediana": round(mediana, 2), "vale": abs(linhas["P_LR"]["nosso_fakebr"] - mediana) <= 3}

    n = len(linhas)
    contagem = {}
    for a in ("A1", "A2", "A3", "A4", "A6", "A7", "A8"):
        vals = [L[a]["vale"] for L in linhas.values() if L[a].get("vale") is not None]
        contagem[a] = {"vale_em": sum(vals), "de": len(vals),
                       "leitura": ("controle: esperado 12 de 12" if a == "A4" else leitura(sum(vals), len(vals)))}
    if "P_LR" in linhas:
        contagem["A5"] = {"vale": linhas["P_LR"]["A5"]["vale"], "mediana": linhas["P_LR"]["A5"]["mediana"],
                          "leitura": "parcial" if n < N_TOTAL else ("sim" if linhas["P_LR"]["A5"]["vale"] else "não")}
    for nome, L in linhas.items():
        print(f"  {nome:8s} A1 {L['A1']['dif_pp']:+5.1f} (Holm {L['A1']['p_holm']:.3g}) | A2 {L['A2']['dif_pp']:+5.1f} | "
              f"A3 {L['A3']['queda_pp']:5.1f} | A4 {L['A4'].get('acuracias')} | A6 {L['A6']['valor']:5.1f} | "
              f"A7 {L['A7']['dif_pp']:+5.1f} (Holm {L['A7']['p_holm']:.3g}) | A8 {L['A8']['dif_pp']:+5.1f}")
    for a, c in contagem.items():
        print(f"  {a}: {c}")
    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        json.dump({"n_modelos": n, "contagem": contagem, "por_modelo": linhas}, f, indent=2, ensure_ascii=False,
                  default=float)
    print(f"salvo em {ARQ_SAIDA}")


if __name__ == "__main__":
    main()
