# -*- coding: utf-8 -*-
"""A data de publicação ainda separa as classes no pareado? (exploratório)

    python scripts/diagnosticos/atalho_data.py

O pareamento equilibra as classes por tema e por ano, mas não dentro do ano. Um
classificador (regressão logística, C = 10) recebe um único atributo categórico
e é avaliado de duas formas:
  - validação cruzada aleatória em 5 dobras, no pareado inteiro: treino e teste
    vêm dos mesmos períodos;
  - divisão temporal dos experimentos: o teste é de um período que o treino não
    viu, e as categorias novas não trazem informação.

Registrada em 05/11/2026. Substitui uma medida anterior (57,7% e 60,8%), feita
de forma avulsa, sem script guardado; esta reproduz os valores dela com folga de
menos de 0,1 ponto.
"""
import os
import json

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_atalho_data.json")
SEMENTE, C = 0, 10.0
ATRIBUTOS = {"tema": lambda r: r["tema_pareamento"],
             "ano": lambda r: r["data_publicacao"][:4],
             "ano_mes": lambda r: r["data_publicacao"][:7],
             "data_completa": lambda r: r["data_publicacao"][:10]}


def rotulos(regs):
    return np.array([r["veredito"] == "FALSO" for r in regs], int)


def modelo():
    return LogisticRegression(C=C, max_iter=2000)


def main():
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    par = [r for r in corpus if r["subconjunto_pareado"] and len(r.get("data_publicacao") or "") >= 10]
    y = rotulos(par)
    treino = [r for r in par if r["split"] == "treino"]
    teste = [r for r in par if r["split"] == "teste"]
    dobras = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)
    res = {"n_pareado": len(par), "classificador": f"regressão logística, C={C:g}", "semente": SEMENTE,
           "n_treino": len(treino), "n_teste": len(teste), "atributos": {}}
    for nome, f in ATRIBUTOS.items():
        X = DictVectorizer().fit_transform([{"a": f(r)} for r in par])
        cv = balanced_accuracy_score(y, cross_val_predict(modelo(), X, y, cv=dobras))
        v = DictVectorizer()
        Xtr, Xte = v.fit_transform([{"a": f(r)} for r in treino]), v.transform([{"a": f(r)} for r in teste])
        tp = balanced_accuracy_score(rotulos(teste), modelo().fit(Xtr, rotulos(treino)).predict(Xte))
        res["atributos"][nome] = {"validacao_cruzada_aleatoria": round(100 * cv, 2), "divisao_temporal": round(100 * tp, 2)}
        print(f"{nome:15s} validação cruzada {100 * cv:6.2f}% | divisão temporal {100 * tp:6.2f}%")
    json.dump(res, open(ARQ_SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", ARQ_SAIDA)


if __name__ == "__main__":
    main()
