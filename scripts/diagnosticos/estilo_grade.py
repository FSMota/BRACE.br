# -*- coding: utf-8 -*-
"""Modelo só de forma com busca em grade (exploratório, posterior aos experimentos).

    python scripts/diagnosticos/estilo_grade.py

O modelo só de forma (modelo_estilo.py) usou os valores padrão do scikit-learn,
enquanto o modelo de referência teve o C escolhido por busca em grade na validação.
Aqui os dois classificadores do modelo só de forma passam pela mesma regra: a grade
é fixada de antemão, cada combinação é treinada no treino do pareado e fica a de
maior acurácia balanceada na validação; o teste e o Fake.br só são usados no fim.
A primeira linha reproduz os valores padrão, para conferir que nada mais mudou.
"""
import os
import sys
import json
import itertools

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SCRIPTS)
from modelo_estilo import atributos, ARQ_CORPUS, SEED          # noqa: E402
from avaliacao_cruzada import carregar_fakebr, entrada_texto    # noqa: E402

ARQ_SAIDA = os.path.join(os.path.dirname(_SCRIPTS), "relatorios", "resultado_estilo_grade.json")
GRADE_LR = [0.01, 0.1, 1, 10, 100]
GRADE_GB = list(itertools.product([0.03, 0.1, 0.3], [15, 31, 63], [100, 300]))  # taxa, folhas, iterações


def main():
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    par = [r for r in corpus if r["subconjunto_pareado"]]
    dados = {}
    for s in ("treino", "validacao", "teste"):
        rs = [r for r in par if r["split"] == s]
        dados[s] = (np.array([atributos(r["texto_entrada"]) for r in rs]),
                    np.array([1 if r["veredito"] == "FALSO" else 0 for r in rs]))
    pares, _ = carregar_fakebr()
    tf = [entrada_texto(p["fake"]["bruto"]) for p in pares]
    tv = [entrada_texto(p["true"]["bruto"]) for p in pares]
    dados["fakebr"] = (np.array([atributos(t) for t in tf + tv]), np.array([1] * len(tf) + [0] * len(tv)))
    Xtr, ytr = dados["treino"]
    esc = StandardScaler().fit(Xtr)

    def avaliar(m, escala):
        prep = esc.transform if escala else (lambda X: X)
        return {s: round(100 * balanced_accuracy_score(dados[s][1], m.predict(prep(dados[s][0]))), 2)
                for s in ("validacao", "teste", "fakebr")}

    res = {"n": {s: int(len(dados[s][1])) for s in dados}}
    res["padrao"] = {
        "regressao_logistica": avaliar(LogisticRegression(max_iter=2000).fit(esc.transform(Xtr), ytr), True),
        "gradient_boosting": avaliar(HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr), False),
    }
    lr = []
    for c in GRADE_LR:
        m = LogisticRegression(C=c, max_iter=5000).fit(esc.transform(Xtr), ytr)
        lr.append({"C": c, **avaliar(m, True)})
    gb = []
    for taxa, folhas, it in GRADE_GB:
        m = HistGradientBoostingClassifier(learning_rate=taxa, max_leaf_nodes=folhas, max_iter=it,
                                           random_state=SEED).fit(Xtr, ytr)
        gb.append({"learning_rate": taxa, "max_leaf_nodes": folhas, "max_iter": it, **avaliar(m, False)})
    res["grade"] = {"regressao_logistica": lr, "gradient_boosting": gb}
    res["escolhido"] = {"regressao_logistica": max(lr, key=lambda x: x["validacao"]),
                        "gradient_boosting": max(gb, key=lambda x: x["validacao"])}
    json.dump(res, open(ARQ_SAIDA, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: res[k] for k in ("n", "padrao", "escolhido")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
