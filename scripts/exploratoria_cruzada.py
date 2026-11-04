# -*- coding: utf-8 -*-
"""Análises EXPLORATÓRIAS da avaliação cruzada, com testes de hipótese.

    python scripts/exploratoria_cruzada.py

EXPLORATÓRIO: decidido DEPOIS de ver o resultado da avaliação cruzada (adendo 1
do relatorios/PREREGISTRO_AVALIACAO_CRUZADA.md). Os testes abaixo estão
especificados antes de rodar -- hipóteses, alfa e correção --, mas a escolha de
QUAIS testes fazer foi influenciada pelo resultado. Relatar como exploratório.

Alfa = 0,05. T2, T3 e T4 são a família de testes desta análise: p-valores
corrigidos por Holm-Bonferroni. T1 é conferência e fica fora da família.

T1  O acerto no Fake.br é maior que o acaso?
    H0: acurácia balanceada = 50%.  Teste: IC 95% por bootstrap (2.000
    reamostragens de PARES do Fake.br -- cada par tem uma falsa e uma
    verdadeira, então a acurácia do par já é balanceada).

T2  O alvo político sustenta a transferência?                    [principal]
    Modelo M0 = o do pré-registro. Modelo M1 = idêntico, mas treinado e avaliado
    com os termos políticos apagados (lula, dilma, pt, petista, petralha,
    lulista, rousseff, "luiz inácio"), no corpus interno E no Fake.br.
    H0: M0 e M1 erram os mesmos textos na mesma proporção.
    Teste: McNemar exato (binomial sobre os textos em que só um dos dois acerta).
    Se o confundidor for político: M1 cai de forma significativa no Fake.br.

T3  O modelo acerta diferente nos textos que citam o PT?
    Fake.br, modelo M0: acurácia balanceada nos textos COM menção política
    contra SEM menção. H0: sem diferença. Teste: permutação (2.000) do rótulo
    "com menção". Diagnóstico junto: se o modelo aprendeu "cita o PT = falso",
    a revocação de VERDADEIRO cai nos textos com menção -- o G1 também cobre
    Lula e Dilma.

T4  Título contra texto, no MESMO conjunto (os 946 pares da entrada B).
    H0: entrada A e entrada B erram na mesma proporção nesses pares.
    Teste: McNemar exato.
"""
import os
import re
import sys
import json
import time
import random
from collections import Counter

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avaliacao_cruzada import carregar_fakebr, entrada_texto, titulo, C_FIXO   # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_exploratoria_cruzada.json")

ALFA = 0.05
N_BOOT = 2000
N_PERM = 2000
SEED = 0
POLITICO = re.compile(r"\b(?:lula|lulista\w*|luiz\s+in[áa]cio|dilma|rousseff|pt|petista\w*|petralha\w*)\b",
                      re.I)
T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def apagar_politico(t):
    return re.sub(r"\s{2,}", " ", POLITICO.sub(" ", t))


def treinar(treino, transforma=lambda t: t):
    vet = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vet.fit_transform([transforma(r["texto_entrada"]) for r in treino])
    y = [1 if r["veredito"] == "FALSO" else 0 for r in treino]
    return vet, LogisticRegression(C=C_FIXO, max_iter=2000).fit(X, y)


def prever(vet, modelo, textos, transforma=lambda t: t):
    return modelo.predict(vet.transform([transforma(t) for t in textos]))


def mcnemar(acerto_a, acerto_b):
    b = int(np.sum(acerto_a & ~acerto_b))       # só A acerta
    c = int(np.sum(~acerto_a & acerto_b))       # só B acerta
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def ic_bootstrap_pares(acerto_pares, rng):
    """acerto_pares: matriz (n_pares, 2) de acertos [falsa, verdadeira]."""
    n = len(acerto_pares)
    medias = [acerto_pares[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def holm(pvalores):
    """Devolve p-valores corrigidos por Holm-Bonferroni, na ordem de entrada."""
    ordem = sorted(range(len(pvalores)), key=lambda i: pvalores[i])
    m, corrigidos, maior = len(pvalores), [0.0] * len(pvalores), 0.0
    for pos, i in enumerate(ordem):
        maior = max(maior, min(1.0, (m - pos) * pvalores[i]))
        corrigidos[i] = maior
    return corrigidos


def main():
    rng = np.random.default_rng(SEED)
    log("ANÁLISE EXPLORATÓRIA -- avaliação cruzada (não pré-registrada)")
    log(f"alfa {ALFA} | família T2-T4 corrigida por Holm | bootstrap {N_BOOT} | permutação {N_PERM}")
    log("")

    with open(ARQ_CORPUS, encoding="utf-8") as f:
        corpus = [json.loads(l) for l in f if l.strip()]
    par = [r for r in corpus if r["subconjunto_pareado"]]
    treino = [r for r in par if r["split"] == "treino"]
    teste = [r for r in par if r["split"] == "teste"]

    log("prevalência de menção política (lula, dilma, pt...):")
    for nome, regs in (("treino interno", treino), ("teste interno", teste)):
        c = Counter((r["veredito"], bool(POLITICO.search(r["texto_entrada"]))) for r in regs)
        f_tot = c[("FALSO", True)] + c[("FALSO", False)]
        v_tot = c[("VERDADEIRO", True)] + c[("VERDADEIRO", False)]
        log(f"  {nome:16s} falsas {100 * c[('FALSO', True)] / f_tot:5.1f}%   "
            f"verdadeiras {100 * c[('VERDADEIRO', True)] / v_tot:5.1f}%")

    log("carregando o Fake.br...")
    pares, _ = carregar_fakebr()
    textos_f = [entrada_texto(p["fake"]["bruto"]) for p in pares]
    textos_v = [entrada_texto(p["true"]["bruto"]) for p in pares]
    for nome, ts in (("Fake.br falsas", textos_f), ("Fake.br verdadeiras", textos_v)):
        log(f"  {nome:16s} {100 * sum(bool(POLITICO.search(t)) for t in ts) / len(ts):5.1f}% com menção")
    textos = textos_f + textos_v
    y = np.array([1] * len(textos_f) + [0] * len(textos_v))

    # ---------------- M0 e M1 ----------------
    log("")
    log("treinando M0 (pré-registro) e M1 (termos políticos apagados)...")
    vet0, m0 = treinar(treino)
    vet1, m1 = treinar(treino, apagar_politico)
    yt = [1 if r["veredito"] == "FALSO" else 0 for r in teste]
    int0 = balanced_accuracy_score(yt, prever(vet0, m0, [r["texto_entrada"] for r in teste]))
    int1 = balanced_accuracy_score(yt, prever(vet1, m1, [r["texto_entrada"] for r in teste], apagar_politico))
    p0 = prever(vet0, m0, textos)
    p1 = prever(vet1, m1, textos, apagar_politico)
    ac0, ac1 = (p0 == y), (p1 == y)
    cruz0, cruz1 = balanced_accuracy_score(y, p0), balanced_accuracy_score(y, p1)
    log(f"  M0: interno {int0:.2%} | Fake.br {cruz0:.2%}")
    log(f"  M1: interno {int1:.2%} | Fake.br {cruz1:.2%}")

    n = len(pares)
    pares0 = np.stack([ac0[:n], ac0[n:]], axis=1).astype(float)
    pares1 = np.stack([ac1[:n], ac1[n:]], axis=1).astype(float)

    # ---------------- T1 ----------------
    log("")
    log("T1 -- acerto no Fake.br maior que o acaso? (H0: 50%)")
    ic0 = ic_bootstrap_pares(pares0, rng)
    log(f"  M0 {cruz0:.2%}  IC95% [{ic0[0]:.2%}, {ic0[1]:.2%}]  -> "
        f"{'rejeita H0' if ic0[0] > 0.5 else 'não rejeita H0'}")

    # ---------------- T2 ----------------
    log("")
    log("T2 -- apagar os termos políticos muda o acerto no Fake.br? (McNemar, M0 x M1)")
    b, c, p_t2 = mcnemar(ac0, ac1)
    dif = pares0.mean(axis=1) - pares1.mean(axis=1)
    boots = [dif[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    ic_dif = (float(np.percentile(boots, 2.5)) * 100, float(np.percentile(boots, 97.5)) * 100)
    log(f"  só M0 acerta: {b} | só M1 acerta: {c} | p = {p_t2:.4g}")
    log(f"  diferença M0 - M1: {100 * (cruz0 - cruz1):+.2f} pp  IC95% [{ic_dif[0]:+.2f}, {ic_dif[1]:+.2f}] pp")

    # ---------------- T3 ----------------
    log("")
    log("T3 -- M0 no Fake.br: textos COM menção política x SEM menção (permutação)")
    mencao = np.array([bool(POLITICO.search(t)) for t in textos])

    def bal(mascara):
        return balanced_accuracy_score(y[mascara], p0[mascara])
    obs = bal(mencao) - bal(~mencao)
    rng_p = random.Random(SEED)
    idx = list(range(len(mencao)))
    extremos = 0
    for _ in range(N_PERM):
        rng_p.shuffle(idx)
        perm = mencao[idx]
        if abs(bal(perm) - bal(~perm)) >= abs(obs):
            extremos += 1
    p_t3 = (extremos + 1) / (N_PERM + 1)
    for nome, msk in (("com menção", mencao), ("sem menção", ~mencao)):
        rf = recall_score(y[msk], p0[msk], pos_label=1)
        rv = recall_score(y[msk], p0[msk], pos_label=0)
        log(f"  {nome}: acurácia bal. {bal(msk):.2%} | revocação FALSO {rf:.2%} | "
            f"revocação VERDADEIRO {rv:.2%} | n falsas {int(np.sum(msk & (y == 1)))}, "
            f"verdadeiras {int(np.sum(msk & (y == 0)))}")
    log(f"  diferença (com - sem): {100 * obs:+.2f} pp | p = {p_t3:.4g}")

    # ---------------- T4 ----------------
    log("")
    log("T4 -- título x texto no MESMO conjunto (pares com título dos dois lados; McNemar)")
    sel = [i for i, p in enumerate(pares) if titulo(p["fake"]["bruto"]) and titulo(p["true"]["bruto"])]
    tit = [titulo(pares[i]["fake"]["bruto"]) for i in sel] + [titulo(pares[i]["true"]["bruto"]) for i in sel]
    txt = [textos_f[i] for i in sel] + [textos_v[i] for i in sel]
    ys = np.array([1] * len(sel) + [0] * len(sel))
    pa, pb = prever(vet0, m0, txt), prever(vet0, m0, tit)
    acc_a, acc_b = balanced_accuracy_score(ys, pa), balanced_accuracy_score(ys, pb)
    bb, cc, p_t4 = mcnemar(pa == ys, pb == ys)
    log(f"  {len(sel)} pares | texto {acc_a:.2%} | título {acc_b:.2%} | "
        f"diferença {100 * (acc_b - acc_a):+.2f} pp")
    log(f"  só o texto acerta: {bb} | só o título acerta: {cc} | p = {p_t4:.4g}")
    log("  o que a regra de título capturou nas VERDADEIRAS (5 exemplos):")
    for i in sel[:5]:
        log(f"    {titulo(pares[i]['true']['bruto'])[:110]!r}")

    # ---------------- correção ----------------
    brutos = [p_t2, p_t3, p_t4]
    corr = holm(brutos)
    log("")
    log("=" * 72)
    log("RESUMO (T2-T4 com correção de Holm; alfa 0,05)")
    log("=" * 72)
    log(f"  T1 acaso        M0 {cruz0:.2%} IC95% [{ic0[0]:.2%}, {ic0[1]:.2%}]    "
        f"{'SIGNIFICATIVO' if ic0[0] > 0.5 else 'não significativo'}")
    for nome, pb_, pc in (("T2 político (M0xM1)", p_t2, corr[0]), ("T3 com x sem menção", p_t3, corr[1]),
                          ("T4 título x texto", p_t4, corr[2])):
        log(f"  {nome:20s} p = {pb_:.4g} | p Holm = {pc:.4g}   "
            f"{'SIGNIFICATIVO' if pc < ALFA else 'não significativo'}")
    log("=" * 72)

    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump({"exploratorio": True, "alfa": ALFA,
                   "T1": {"M0_fakebr": round(cruz0, 4), "ic95": [round(ic0[0], 4), round(ic0[1], 4)]},
                   "T2": {"M0_interno": round(int0, 4), "M1_interno": round(int1, 4),
                          "M0_fakebr": round(cruz0, 4), "M1_fakebr": round(cruz1, 4),
                          "so_M0": b, "so_M1": c, "p": p_t2, "p_holm": corr[0],
                          "ic95_dif_pp": [round(ic_dif[0], 2), round(ic_dif[1], 2)]},
                   "T3": {"dif_pp": round(100 * obs, 2), "p": p_t3, "p_holm": corr[1]},
                   "T4": {"pares": len(sel), "texto": round(acc_a, 4), "titulo": round(acc_b, 4),
                          "p": p_t4, "p_holm": corr[2]}},
                  f, indent=2, ensure_ascii=False)
    log(f"resultados salvos em relatorios/{os.path.basename(SAIDA)}")


if __name__ == "__main__":
    main()
