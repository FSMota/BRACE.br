# -*- coding: utf-8 -*-
"""Sensibilidade: o modelo de referência com e sem as falsas de rótulo duvidoso.

    python scripts/marcar_rotulo_duvidoso.py      # antes, gera a lista
    python scripts/sensibilidade_rotulo_duvidoso.py

Exploratório, não pré-registrado. Modelo de referência (TF-IDF de palavras +
regressão logística, C = 10, o valor escolhido na validação), texto bruto.
Três cenários, para todas as marcadas e só para as categorias de risco:
  1. como nos experimentos (referência);
  2. mesmo modelo, teste sem as marcadas;
  3. treino e teste sem as marcadas (modelo retreinado).
O Fake.br é avaliado nos cenários 1 e 3. Como as marcadas são todas falsas,
o teste sem elas fica levemente desbalanceado; a acurácia balanceada já
compensa isso.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparativo_bracos import rep_tfidf, treinar, avaliar, pct          # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_MARCAS = os.path.join(_RAIZ, "relatorios", "marcacoes", "rotulo_duvidoso.json")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_sensibilidade_rotulo_duvidoso.json")
C_REFERENCIA = 10.0


def main():
    regs = [json.loads(l) for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()]
    por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste", "fakebr")}
    marcas = json.load(open(ARQ_MARCAS, encoding="utf-8"))
    conjuntos = {"todas": {i["id"] for i in marcas["itens"]},
                 "risco": {i["id"] for i in marcas["itens"] if i["risco"]}}
    rep = rep_tfidf("texto_entrada")
    ref = treinar(rep, por, C=C_REFERENCIA)
    base_int, base_fb = pct(avaliar(ref, por["teste"])["bal"]), pct(avaliar(ref, por["fakebr"])["bal"])
    res = {"modelo": "TF-IDF de palavras + LR, C=10, texto bruto",
           "referencia": {"teste": base_int, "fakebr": base_fb}, "cenarios": {}}
    print(f"referência: teste {base_int:.2f}% | Fake.br {base_fb:.2f}%")
    for nome, ids in conjuntos.items():
        n = {s: sum(r["id"] in ids for r in por[s]) for s in ("treino", "validacao", "teste")}
        teste_sem = [r for r in por["teste"] if r["id"] not in ids]
        so_teste = pct(avaliar(ref, teste_sem)["bal"])
        sem = {s: [r for r in por[s] if r["id"] not in ids] for s in ("treino", "validacao")}
        sem["teste"], sem["fakebr"] = teste_sem, por["fakebr"]
        novo = treinar(rep, sem, C=C_REFERENCIA)
        re_int, re_fb = pct(avaliar(novo, teste_sem)["bal"]), pct(avaliar(novo, por["fakebr"])["bal"])
        res["cenarios"][nome] = {
            "marcadas_no_pareado": n,
            "teste_sem_marcadas": {"teste": so_teste, "dif_pp": round(so_teste - base_int, 2)},
            "retreinado_sem_marcadas": {"teste": re_int, "dif_teste_pp": round(re_int - base_int, 2),
                                        "fakebr": re_fb, "dif_fakebr_pp": round(re_fb - base_fb, 2)}}
        print(f"{nome:6s} marcadas no pareado {n} | teste sem elas {so_teste:.2f}% ({so_teste - base_int:+.2f}) | "
              f"retreinado: teste {re_int:.2f}% ({re_int - base_int:+.2f}), Fake.br {re_fb:.2f}% ({re_fb - base_fb:+.2f})")
    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(f"-> {ARQ_SAIDA}")


if __name__ == "__main__":
    main()
