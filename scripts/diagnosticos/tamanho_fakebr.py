# -*- coding: utf-8 -*-
"""Quanto o tamanho do texto separa as classes do Fake.br, completo e cortado (exploratório).

    python scripts/diagnosticos/tamanho_fakebr.py

Mesmos atributos da checagem C1 (comparativo_bracos.py): número de palavras e log do
número de caracteres, padronizados, numa regressão logística. Validação cruzada em 5
dobras com os pares do Fake.br (falsa n e verdadeira n) sempre na mesma dobra.
Mede o texto completo e o texto com o corte da avaliação externa.
"""
import os
import re
import sys
import math
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS)
from avaliacao_cruzada import carregar_fakebr, entrada_texto   # noqa: E402

ARQ_SAIDA = os.path.join(os.path.dirname(_SCRIPTS), "relatorios", "resultado_tamanho_fakebr.json")


def atributos(textos):
    return np.array([[len(re.findall(r"\w+", t)), math.log1p(len(t))] for t in textos])


def completo(bruto):
    return re.sub(r"\s+", " ", bruto.replace("﻿", "")).strip()


def medir(textos, y, grupos):
    X = atributos(textos)
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, grupos):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return balanced_accuracy_score(y, pred)


def main():
    pares, _ = carregar_fakebr()
    y = np.array([1] * len(pares) + [0] * len(pares))          # 1 = falsa
    grupos = np.array([p["n"] for p in pares] * 2)
    res = {}
    for nome, f in (("completo", completo), ("cortado", entrada_texto)):
        textos = [f(p["fake"]["bruto"]) for p in pares] + [f(p["true"]["bruto"]) for p in pares]
        med_f = float(np.median([len(t) for t in textos[:len(pares)]]))
        med_v = float(np.median([len(t) for t in textos[len(pares):]]))
        res[nome] = {"acuracia_balanceada": round(100 * medir(textos, y, grupos), 2),
                     "mediana_caracteres_falsas": med_f, "mediana_caracteres_verdadeiras": med_v}
    res["n_pares"] = len(pares)
    json.dump(res, open(ARQ_SAIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
