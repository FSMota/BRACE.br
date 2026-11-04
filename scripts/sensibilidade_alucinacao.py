# -*- coding: utf-8 -*-
"""Sensibilidade: o modelo de referência com e sem as entradas inventadas pelo LLM.
Pré-registro: `relatorios/PREREGISTRO_ALUCINACAO.md` (§3 e adendo 2).

    python scripts/sensibilidade_alucinacao.py

Marca as falsas da via LLM com cobertura lexical < 0,5 (`relatorios/alucinacao/
resultado_cobertura.json`) em `relatorios/marcacoes/entrada_inventada.json` e
refaz o modelo de referência (TF-IDF + LR, mesma grade e mesma escolha na
validação) sem elas no treino, na validação e no teste. Diferente da checagem
de discurso, as marcadas também estão no treino, então o modelo é retreinado.
"""
import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparativo_bracos import rotulos, bal, pct                           # noqa: E402
from matriz_modelos import carregar, nossa_direcao, BRACOS                  # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_COBERTURA = os.path.join(_RAIZ, "relatorios", "alucinacao", "resultado_cobertura.json")
ARQ_MARCAS = os.path.join(_RAIZ, "relatorios", "marcacoes", "entrada_inventada.json")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "alucinacao", "resultado_sensibilidade.json")


def marcar():
    itens = [x for x in json.load(open(ARQ_COBERTURA, encoding="utf-8"))["itens"] if x["faixa"] == "baixa"]
    marcas = {"marcacao": "entrada_inventada",
              "motivo": "falsa da via llm cujo texto não existe no artigo de checagem (cobertura lexical < 0,5; "
                        "30 de 30 confirmados como inventados na validação no site ao vivo)",
              "n": len(itens), "fonte_cortada": sum(x["fonte_cortada"] for x in itens),
              "itens": [{k: x[k] for k in ("id", "url", "cobertura", "fonte_cortada", "pareado")} for x in itens]}
    os.makedirs(os.path.dirname(ARQ_MARCAS), exist_ok=True)
    with open(ARQ_MARCAS, "w", encoding="utf-8") as f:
        json.dump(marcas, f, indent=1, ensure_ascii=False)
    return {x["id"] for x in itens}


def main():
    marcas = marcar()
    por = carregar(False)
    sem = {s: [r for r in v if s == "fakebr" or r["id"] not in marcas] for s, v in por.items()}
    res = {"n_marcadas": len(marcas),
           "removidas_por_split": {s: len(por[s]) - len(sem[s]) for s in ("treino", "validacao", "teste")}}
    for braco, campo in BRACOS.items():
        for versao, p in (("com", por), ("sem", sem)):
            h, prever = nossa_direcao("P", "LR", p, campo)
            res[f"{braco}|{versao}"] = {"C": h, "nosso_teste": pct(bal(rotulos(p["teste"]), prever(p["teste"]))),
                                        "nosso_fakebr": pct(bal(rotulos(p["fakebr"]), prever(p["fakebr"])))}
        for m in ("nosso_teste", "nosso_fakebr"):
            res[f"{braco}|dif_pp|{m}"] = round(res[f"{braco}|sem"][m] - res[f"{braco}|com"][m], 2)
    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False, default=float)
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
