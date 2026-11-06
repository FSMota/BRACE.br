# -*- coding: utf-8 -*-
"""Sensibilidade: o modelo de referência sem as entradas que a revisão de rótulos tirou da tarefa binária.

    python scripts/sensibilidade_revisao_rotulos.py

Exploratório, não pré-registrado. Modelo de referência (TF-IDF de palavras +
regressão logística, C = 10), retreinado sem as entradas que a revisão
(relatorios/marcacoes/revisao_rotulos.json) retirou da base ou marcou como
NAO_E_CHECAGEM. Substitui sensibilidade_rotulo_duvidoso.py, que usava a
varredura de títulos, depois abandonada em favor do veredito do checador.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparativo_bracos import rep_tfidf, treinar, avaliar, pct          # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_REVISAO = os.path.join(_RAIZ, "relatorios", "marcacoes", "revisao_rotulos.json")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_sensibilidade_revisao_rotulos.json")
C_REFERENCIA = 10.0
BRACOS = {"bruto": "texto_entrada", "refinado": "ideia_central"}


def main():
    regs = [json.loads(l) for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()]
    por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste", "fakebr")}
    decisoes = json.load(open(ARQ_REVISAO, encoding="utf-8"))["decisoes"]
    ids = {i for i, v in decisoes.items() if v["decisao"] != "FALSO"}
    sem = {s: [r for r in por[s] if r["id"] not in ids] for s in ("treino", "validacao", "teste")}
    sem["fakebr"] = por["fakebr"]
    res = {"modelo": "TF-IDF de palavras + LR, C=10", "fora_da_tarefa_binaria": len(ids),
           "retiradas_do_pareado": {s: len(por[s]) - len(sem[s]) for s in ("treino", "validacao", "teste")},
           "bracos": {}}
    for nome, campo in BRACOS.items():
        rep = rep_tfidf(campo)
        ref, novo = treinar(rep, por, C=C_REFERENCIA), treinar(rep, sem, C=C_REFERENCIA)
        a_t, a_f = pct(avaliar(ref, por["teste"])["bal"]), pct(avaliar(ref, por["fakebr"])["bal"])
        d_t, d_f = pct(avaliar(novo, sem["teste"])["bal"]), pct(avaliar(novo, por["fakebr"])["bal"])
        res["bracos"][nome] = {"teste": {"antes": a_t, "depois": d_t, "dif_pp": round(d_t - a_t, 2)},
                               "fakebr": {"antes": a_f, "depois": d_f, "dif_pp": round(d_f - a_f, 2)}}
        print(f"{nome:8s} teste {a_t:.2f} -> {d_t:.2f} ({d_t - a_t:+.2f}) | Fake.br {a_f:.2f} -> {d_f:.2f} ({d_f - a_f:+.2f})")
    print("retiradas do pareado:", res["retiradas_do_pareado"])
    json.dump(res, open(ARQ_SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", ARQ_SAIDA)


if __name__ == "__main__":
    main()
