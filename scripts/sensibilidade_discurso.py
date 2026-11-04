# -*- coding: utf-8 -*-
"""Sensibilidade: as métricas de teste com e sem as checagens de discurso (G1 e Lupa).
Pré-registro: `relatorios/PREREGISTRO_SENSIBILIDADE_DISCURSO.md`.

    python scripts/sensibilidade_discurso.py

Usa as predições já gravadas da matriz (`relatorios/matriz/pred_*.npz`); nenhum
modelo é retreinado, porque nenhuma instância marcada está no treino ou na
validação. As afirmações A2, A3, A6, A7 e A8 são recontadas sem as instâncias,
com os mesmos critérios de `apurar_matriz.py` e os 11 modelos válidos.
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
ARQ_MARCAS = os.path.join(_RAIZ, "relatorios", "marcacoes", "checagem_de_discurso.json")
PASTA = os.path.join(_RAIZ, "relatorios", "matriz")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_sensibilidade_discurso.json")
EXCLUIDOS_CONTROLE = {"Bc_NB", "Ol_NB"}                  # adendo 2 da matriz
PREVISOES = {"D1": 0.5, "D2": (-0.3, 0.3)}


def main():
    rot = {json.loads(l)["id"]: (1 if json.loads(l)["veredito"] == "FALSO" else 0)
           for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()}
    marcas = {i["id"] for i in json.load(open(ARQ_MARCAS, encoding="utf-8"))["itens"] if i["subconjunto_pareado"]}
    res, mudancas = {"n_marcadas_no_teste": 0, "por_modelo": {}}, []
    contagem = {a: {"com": 0, "sem": 0} for a in ("A2", "A3", "A6", "A7", "A8")}
    testes = {"A2": [], "A7": []}
    for caminho in sorted(glob.glob(os.path.join(PASTA, "pred_*_bruto.npz"))):
        nome = os.path.basename(caminho)[5:-10]
        b, r = np.load(caminho), np.load(caminho.replace("_bruto", "_refinado"))
        ids = list(b["teste_ids"])
        y = np.array([rot[i] for i in ids])
        manter = np.array([i not in marcas for i in ids])
        res["n_marcadas_no_teste"] = int((~manter).sum())
        y_fb = np.array([rot[i] for i in b["fakebr_ids"]])
        L = {}
        for braco, z in (("bruto", b), ("refinado", r)):
            for chave in ("nosso_teste", "fakebr_nosso"):
                com, sem = pct(bal(y, z[chave])), pct(bal(y[manter], z[chave][manter]))
                L[f"{braco}|{chave}"] = {"com": com, "sem": sem, "dif_pp": round(sem - com, 2)}
                mudancas.append((f"{nome}|{braco}|{chave}", round(sem - com, 2)))
        nfb = pct(bal(y_fb, b["nosso_fakebr"]))
        res["por_modelo"][nome] = L
        if nome in EXCLUIDOS_CONTROLE:
            continue
        for versao, m in (("com", np.ones_like(manter)), ("sem", manter)):
            m = m.astype(bool)
            bt = pct(bal(y[m], b["nosso_teste"][m]))
            bfn = pct(bal(y[m], b["fakebr_nosso"][m]))
            contagem["A3"][versao] += (bt - nfb) >= 10
            contagem["A6"][versao] += 65 <= bfn <= 80
            contagem["A8"][versao] += (bfn - nfb) > 0
            for a, chave in (("A2", "nosso_teste"), ("A7", "fakebr_nosso")):
                so_b, so_r, p = mcnemar(b[chave][m] == y[m], r[chave][m] == y[m])
                dif = pct(bal(y[m], b[chave][m])) - pct(bal(y[m], r[chave][m]))
                testes[a].append((versao, dif, p))
    for a, lista in testes.items():
        for versao in ("com", "sem"):
            sub = [(d, p) for v, d, p in lista if v == versao]
            corr = holm([p for _, p in sub])
            contagem[a][versao] = int(sum(d > 0 and pc < 0.05 for (d, _), pc in zip(sub, corr)))
    contagem = {a: {k: int(v) for k, v in c.items()} for a, c in contagem.items()}
    maior = max(mudancas, key=lambda x: abs(x[1]))
    ref = res["por_modelo"]["P_LR"]["bruto|nosso_teste"]["dif_pp"]
    res["afirmacoes_11_validos"] = contagem
    res["maior_variacao"] = {"onde": maior[0], "pp": maior[1]}
    res["referencia_P_LR_bruto_nosso_teste_pp"] = ref
    muda = any(c["com"] != c["sem"] for c in contagem.values())
    res["leitura"] = ("não afetam as conclusões" if abs(maior[1]) <= 0.5 and not muda
                      else "relatar os números também sem elas e discutir a mudança")
    res["confronto"] = {"D1": [maior[1], abs(maior[1]) <= PREVISOES["D1"]],
                        "D2": [ref, PREVISOES["D2"][0] <= ref <= PREVISOES["D2"][1]],
                        "D3": ["mudou" if muda else "nenhuma mudou", not muda]}
    print(f"instâncias marcadas no teste: {res['n_marcadas_no_teste']}")
    for nome, L in res["por_modelo"].items():
        print(f"  {nome:7s} " + " | ".join(f"{k} {v['com']:.2f}→{v['sem']:.2f} ({v['dif_pp']:+.2f})" for k, v in L.items()))
    print(f"afirmações (com → sem): {contagem}")
    print(f"maior variação: {res['maior_variacao']} | referência: {ref:+.2f} pp | leitura: {res['leitura']}")
    print(f"confronto: {res['confronto']}")
    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)


if __name__ == "__main__":
    main()
