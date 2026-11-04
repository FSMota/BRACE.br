# -*- coding: utf-8 -*-
"""Avaliação inversa: treinar no Fake.br e testar na nossa base, fechando a
matriz 2 × 2 (treino × teste) dos braços bruto e refinado.
Pré-registro: `relatorios/PREREGISTRO_AVALIACAO_INVERSA.md`.

    python scripts/avaliacao_inversa.py --fumaca   # confere o código numa amostra pequena; não mostra métricas
    python scripts/avaliacao_inversa.py            # a rodada pré-registrada

Células:
  Fake.br → Fake.br   5 dobras por par (o falso n e o verdadeiro n juntos), predições fora da dobra
  Fake.br → nosso     treino nos 7.200, teste no `teste` da nossa base (e, sensibilidade, todo o pareado)
  nosso → Fake.br     refeita aqui, só para conferir a referência e medir a queda nos mesmos itens
  nosso (7.200) → Fake.br   controle de tamanho: 3.600 + 3.600 do nosso `treino`

Nada da nossa base é usado para escolher o C do modelo do Fake.br, e nada do
Fake.br é usado para escolher o C do nosso.
"""
import os
import re
import sys
import json
import random

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparativo_bracos import (rep_tfidf, treinar, avaliar, rotulos, bal, indices_bootstrap,   # noqa: E402
                                bal_bootstrap, ic, mcnemar, pct, VALORES_C, SEED)

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_RESULTADO = os.path.join(_RAIZ, "relatorios", "resultado_avaliacao_inversa.json")
BRACOS = {"bruto": "texto_entrada", "refinado": "ideia_central"}
N_DOBRAS = 5
N_TAMANHO = 3600                                   # por classe, o tamanho do Fake.br
REFERENCIA = {"bruto": {"interno": 91.19, "fakebr": 73.51}, "refinado": {"interno": 81.73, "fakebr": 66.31}}
PREVISOES = {"I1": (55, 68), "I2": 15, "I3": (80, 92), "I4": (-3, 5), "I5": 3, "I6": 0.5}


def par(reg):
    return int(reg["id"].rsplit("-", 1)[1])


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]|\d|_", " ", (t or "").lower())).strip()


def ajustar_lr(rep, treino, C):
    transformar = rep(treino)
    m = LogisticRegression(C=C, max_iter=2000).fit(transformar(treino), rotulos(treino))
    return {"transformar": transformar, "modelo": m, "C": C}


def dobras_por_par(fakebr):
    pares = sorted({par(r) for r in fakebr})
    random.Random(SEED).shuffle(pares)
    dobra_de = {p: i % N_DOBRAS for i, p in enumerate(pares)}
    return [dobra_de[par(r)] for r in fakebr]


def validacao_cruzada(rep, fakebr, dobras):
    """C pela média da acurácia balanceada nas dobras; devolve C e as predições fora da dobra."""
    y = rotulos(fakebr)
    melhor = None
    for C in VALORES_C:
        pred, por_dobra = np.zeros(len(fakebr), dtype=int), []
        for k in range(N_DOBRAS):
            tr = [r for r, d in zip(fakebr, dobras) if d != k]
            idx = [i for i, d in enumerate(dobras) if d == k]
            t = ajustar_lr(rep, tr, C)
            pred[idx] = t["modelo"].predict(t["transformar"]([fakebr[i] for i in idx]))
            por_dobra.append(bal(y[idx], pred[idx]))
        media = float(np.mean(por_dobra))
        if melhor is None or media > melhor[0]:
            melhor = (media, C, pred.copy(), por_dobra)
    media, C, pred, por_dobra = melhor
    acerto = pred == y
    return C, {"y": y, "pred": pred, "acerto": acerto, "bal": bal(y, pred),
               "rev_falso": (pred[y == 1] == 1).mean(), "rev_verdadeiro": (pred[y == 0] == 0).mean(),
               "por_dobra": [pct(b) for b in por_dobra]}


def resumo(av, idx):
    return {"n": int(len(av["y"])), "bal": pct(av["bal"]), "ic95": ic(bal_bootstrap(av, idx)),
            "rev_falso": pct(av["rev_falso"]), "rev_verdadeiro": pct(av["rev_verdadeiro"])}


def sobreposicao(nosso, fakebr):
    textos = {norm(r["texto_entrada"]) for r in nosso}
    prefixos = {t[:100] for t in textos if len(t) >= 100}
    exatos = sum(norm(r["texto_entrada"]) in textos for r in fakebr)
    pref = sum(len(norm(r["texto_entrada"])) >= 100 and norm(r["texto_entrada"])[:100] in prefixos for r in fakebr)
    return {"fakebr_identicos_na_nossa_base": exatos, "fakebr_mesmos_100_primeiros": pref,
            "pct_identicos": round(100 * exatos / len(fakebr), 3)}


def main():
    fumaca = "--fumaca" in sys.argv
    regs = [json.loads(l) for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()]
    por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste", "fakebr")}
    if fumaca:
        rnd = random.Random(1)
        pares = set(rnd.sample(sorted({par(r) for r in por["fakebr"]}), 150))
        por["fakebr"] = [r for r in por["fakebr"] if par(r) in pares]
        por["teste"] = rnd.sample(por["teste"], 300)
        print("FUMAÇA: amostras pequenas, nenhuma métrica é mostrada nem salva")
    fakebr, teste = por["fakebr"], por["teste"]
    pareado = por["treino"] + por["validacao"] + por["teste"]
    print(f"Fake.br {len(fakebr)} | nosso teste {len(teste)} | nosso pareado {len(pareado)}")

    sob = sobreposicao(pareado, fakebr)
    print("sobreposição: calculada (oculta na fumaça)" if fumaca else f"sobreposição (relatada antes do resultado): {sob}")
    dobras = dobras_por_par(fakebr)
    print(f"dobras por par: {[dobras.count(k) for k in range(N_DOBRAS)]} itens por dobra")

    rnd = random.Random(SEED)
    sub = []
    for classe in ("FALSO", "VERDADEIRO"):
        cands = sorted((r for r in por["treino"] if r["veredito"] == classe), key=lambda r: r["id"])
        sub += rnd.sample(cands, min(N_TAMANHO, len(cands)))

    rng = np.random.default_rng(SEED)
    idx_fb, idx_te, idx_par = (indices_bootstrap(len(x), rng) for x in (fakebr, teste, pareado))
    res = {"sobreposicao": sob, "n": {"fakebr": len(fakebr), "teste": len(teste), "pareado": len(pareado),
                                      "controle_tamanho_treino": len(sub)}, "por_braco": {}}
    acertos_fb_nosso = {}

    for braco, campo in BRACOS.items():
        rep = rep_tfidf(campo)
        C_fb, fb_fb = validacao_cruzada(rep, fakebr, dobras)
        t_fb = ajustar_lr(rep, fakebr, C_fb)
        fb_nosso, fb_pareado = avaliar(t_fb, teste), avaliar(t_fb, pareado)
        t_nosso = treinar(rep, {"treino": por["treino"], "validacao": por["validacao"]})
        nosso_fb, nosso_int = avaliar(t_nosso, fakebr), avaliar(t_nosso, teste)
        t_tam = treinar(rep, {"treino": sub, "validacao": por["validacao"]})
        tam_fb = avaliar(t_tam, fakebr)
        acertos_fb_nosso[braco] = fb_nosso["acerto"]
        print(f"\n{braco}: C Fake.br = {C_fb} | C nosso = {t_nosso['C']} | C controle de tamanho = {t_tam['C']}")
        if fumaca:
            print("  (fumaça) todas as células rodaram")
            continue
        b = {"C_fakebr": C_fb, "C_nosso": t_nosso["C"], "C_tamanho": t_tam["C"],
             "fakebr_fakebr": resumo(fb_fb, idx_fb) | {"por_dobra": fb_fb["por_dobra"]},
             "fakebr_nosso_teste": resumo(fb_nosso, idx_te),
             "fakebr_nosso_pareado": resumo(fb_pareado, idx_par),
             "nosso_nosso_conferencia": resumo(nosso_int, idx_te),
             "nosso_fakebr_conferencia": resumo(nosso_fb, idx_fb),
             "nosso7200_fakebr": resumo(tam_fb, idx_fb)}
        dif = bal_bootstrap(fb_fb, idx_fb) - bal_bootstrap(nosso_fb, idx_fb)
        b["queda_mesmos_itens_pp"] = {"valor": pct(fb_fb["bal"] - nosso_fb["bal"]), "ic95": ic(dif)}
        assim = b["nosso7200_fakebr"]["bal"] - b["fakebr_nosso_teste"]["bal"]
        b["assimetria_pp"] = round(assim, 2)
        b["leitura_assimetria"] = ("a nossa base transfere melhor" if assim >= 5 else
                                   "o Fake.br transfere melhor" if assim <= -5 else "simétrica")
        v = b["fakebr_nosso_teste"]["bal"]
        b["leitura_principal"] = ("sinal comum entre as bases" if v >= 80 else "transferência parcial" if v >= 65 else
                                  "transferência fraca" if v >= 55 else "nada transfere" if v >= 45 else "inversão")
        for nome, chave in (("nosso → nosso", "nosso_nosso_conferencia"), ("nosso → Fake.br", "nosso_fakebr_conferencia")):
            ref = REFERENCIA[braco]["interno" if "nosso_nosso" in chave else "fakebr"]
            print(f"  conferência {nome}: {b[chave]['bal']:.2f}% (referência {ref}%)")
        for nome, chave in (("Fake.br → Fake.br", "fakebr_fakebr"), ("Fake.br → nosso teste", "fakebr_nosso_teste"),
                            ("Fake.br → nosso pareado", "fakebr_nosso_pareado"), ("nosso 7.200 → Fake.br", "nosso7200_fakebr")):
            x = b[chave]
            print(f"  {nome:24s} {x['bal']:6.2f}% {x['ic95']} | rev. FALSO {x['rev_falso']:.1f}% | "
                  f"rev. VERDADEIRO {x['rev_verdadeiro']:.1f}%")
        print(f"  queda nos mesmos itens (Fake.br → Fake.br − nosso → Fake.br): {b['queda_mesmos_itens_pp']}")
        print(f"  assimetria (nosso 7.200 → Fake.br − Fake.br → nosso): {assim:+.2f} pp → {b['leitura_assimetria']}")
        print(f"  leitura principal: {b['leitura_principal']}")
        res["por_braco"][braco] = b

    if fumaca:
        return
    bb, c, p = mcnemar(acertos_fb_nosso["refinado"], acertos_fb_nosso["bruto"])
    rb, rr = res["por_braco"]["bruto"], res["por_braco"]["refinado"]
    i4 = round(rr["fakebr_nosso_teste"]["bal"] - rb["fakebr_nosso_teste"]["bal"], 2)
    res["refinado_menos_bruto_fakebr_nosso"] = {"pp": i4, "mcnemar_so_refinado_acerta": bb,
                                                "mcnemar_so_bruto_acerta": c, "p": p}
    i1 = rb["fakebr_nosso_teste"]
    res["confronto_previsoes"] = {
        "I1": [i1["bal"], PREVISOES["I1"][0] <= i1["bal"] <= PREVISOES["I1"][1]],
        "I2": [round(i1["rev_verdadeiro"] - i1["rev_falso"], 2), i1["rev_verdadeiro"] - i1["rev_falso"] >= PREVISOES["I2"]],
        "I3": [rb["fakebr_fakebr"]["bal"], PREVISOES["I3"][0] <= rb["fakebr_fakebr"]["bal"] <= PREVISOES["I3"][1]],
        "I4": [i4, PREVISOES["I4"][0] <= i4 <= PREVISOES["I4"][1]],
        "I5": [round(rb["nosso7200_fakebr"]["bal"] - REFERENCIA["bruto"]["fakebr"], 2),
               abs(rb["nosso7200_fakebr"]["bal"] - REFERENCIA["bruto"]["fakebr"]) <= PREVISOES["I5"]],
        "I6": [res["sobreposicao"]["pct_identicos"], res["sobreposicao"]["pct_identicos"] < PREVISOES["I6"]]}
    print(f"\nrefinado − bruto (Fake.br → nosso): {i4:+.2f} pp | McNemar {bb} × {c}, p = {p:.4f}")
    print(f"confronto com as previsões (§2): {res['confronto_previsoes']}")
    with open(ARQ_RESULTADO, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nsalvo em {ARQ_RESULTADO}")


if __name__ == "__main__":
    main()
