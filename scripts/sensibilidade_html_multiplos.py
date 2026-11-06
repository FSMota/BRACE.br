# -*- coding: utf-8 -*-
"""Sensibilidade: o modelo de referência sem as entradas de maior risco da via html.

    python scripts/sensibilidade_html_multiplos.py

Exploratório, não pré-registrado. A auditoria da via html
(relatorios/anotacao_html/) encontrou erros só em artigos com vários registros.
Modelo de referência (TF-IDF de palavras + regressão logística, C = 10) retreinado
em dois cenários: sem as 6 entradas erradas confirmadas e sem todas as entradas da
via html vindas de artigos com vários registros. Como só saem falsas, as classes
ficam levemente desbalanceadas; a acurácia balanceada compensa isso.
"""
import os
import sys
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparativo_bracos import rep_tfidf, treinar, avaliar, pct          # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_AMOSTRA = os.path.join(_RAIZ, "relatorios", "anotacao_html", "amostra.json")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_sensibilidade_html_multiplos.json")
BRACOS = {"bruto": "texto_entrada", "refinado": "ideia_central"}
ERROS_N = (15, 72, 81, 89, 92, 93)                # linhas da planilha anotadas como lixo ou desmentido


def main():
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    falsas = [r for r in corpus if r["veredito"] == "FALSO"]
    registros = Counter(r["url"] for r in falsas)
    multi = {r["id"] for r in falsas if r.get("origem_viral") == "html" and registros[r["url"]] > 1}
    erros = {i["id"] for i in json.load(open(ARQ_AMOSTRA, encoding="utf-8"))["itens"] if i["n"] in ERROS_N}
    assert len(erros) == 6 and erros <= multi
    regs = [json.loads(l) for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()]
    por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste", "fakebr")}
    res = {"modelo": "TF-IDF de palavras + LR, C=10", "html_multiplos_no_corpus": len(multi), "cenarios": {}}
    for cen, ids in (("sem_6_erros", erros), ("sem_html_multiplos", multi)):
        sem = {s: [r for r in por[s] if r["id"] not in ids] for s in ("treino", "validacao", "teste")}
        sem["fakebr"] = por["fakebr"]
        res["cenarios"][cen] = {"retiradas_do_pareado": {s: len(por[s]) - len(sem[s]) for s in sem if s != "fakebr"}}
        print(cen, res["cenarios"][cen]["retiradas_do_pareado"])
        for nome, campo in BRACOS.items():
            rep = rep_tfidf(campo)
            ref, novo = treinar(rep, por, C=10.0), treinar(rep, sem, C=10.0)
            a_t, a_f = pct(avaliar(ref, por["teste"])["bal"]), pct(avaliar(ref, por["fakebr"])["bal"])
            d_t, d_f = pct(avaliar(novo, sem["teste"])["bal"]), pct(avaliar(novo, por["fakebr"])["bal"])
            res["cenarios"][cen][nome] = {"teste": [a_t, d_t, round(d_t - a_t, 2)], "fakebr": [a_f, d_f, round(d_f - a_f, 2)]}
            print(f"  {nome:8s} teste {a_t:.2f} -> {d_t:.2f} ({d_t - a_t:+.2f}) | Fake.br {a_f:.2f} -> {d_f:.2f} ({d_f - a_f:+.2f})")
    json.dump(res, open(ARQ_SAIDA, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("->", ARQ_SAIDA)


if __name__ == "__main__":
    main()
