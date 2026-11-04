# -*- coding: utf-8 -*-
"""Avaliação cruzada no Fake.br -- braço BRUTO.

    python scripts/avaliacao_cruzada.py

Implementa LITERALMENTE o desenho de relatorios/PREREGISTRO_AVALIACAO_CRUZADA.md
(commit 954a2d9). Onde o pré-registro é ambíguo, a escolha feita aqui está
marcada com "AMBIGUIDADE" e é impressa na saída, para entrar no adendo.

Nada é ajustado usando o Fake.br: o C é fixo (10), o limiar é o padrão (0,5) e o
vocabulário é o do treino interno.
"""
import os
import re
import sys
import json
import time
import statistics
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fundir_corpus import normalizar_fonte                           # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
PASTA_FAKEBR = os.path.join(_RAIZ, "datasets", "externos", "fakebr", "full_texts")
SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_avaliacao_cruzada.json")

C_FIXO = 10.0
CORTE_MIN, CORTE_SEM_FIM = 250, 300
TITULO_MIN, TITULO_MAX = 20, 200
REFERENCIA_INTERNA = 0.9205
T0 = time.time()

AMBIGUIDADES = [
    "Entrada B: o pré-registro fixa a normalização só para a entrada A; aplicada "
    "também aos títulos, para as duas entradas passarem pelo mesmo pré-processamento.",
    "Entrada A: 'primeiro fim de frase depois de 250 caracteres' foi lido "
    "literalmente -- o corte acontece no primeiro . ! ? a partir do caractere 250, "
    "onde quer que esteja; só sem nenhum fim de frase depois de 250 é que o corte é em 300.",
]


def log(msg=""):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def ler(caminho):
    dados = open(caminho, "rb").read()
    try:
        return dados.decode("utf-8")
    except UnicodeDecodeError:
        return dados.decode("latin-1")


def meta(classe, n):
    linhas = ler(os.path.join(PASTA_FAKEBR, f"{classe}-meta-information", f"{n}-meta.txt")).splitlines()
    return {"link": linhas[1].strip(), "categoria": linhas[2].strip(), "data": linhas[3].strip()}


def entrada_texto(bruto):
    """Entrada A: quebras de linha e tabulações -> espaço; corte no primeiro fim
    de frase a partir do caractere 250; sem fim de frase, corte em 300."""
    t = re.sub(r"[\r\n\t]+", " ", bruto.replace("﻿", "")).strip()
    m = re.search(r"[.!?]", t[CORTE_MIN:])
    t = t[:CORTE_MIN + m.end()] if m else t[:CORTE_SEM_FIM]
    return normalizar_fonte(t)[0]


def titulo(bruto):
    """Entrada B: 1ª linha terminada por quebra de linha, ou texto antes do
    primeiro '.' seguido de dois espaços; vale só com 20 a 200 caracteres."""
    t = bruto.replace("﻿", "")
    candidatos = []
    if "\n" in t:
        candidatos.append(t.split("\n", 1)[0])
    m = re.search(r"\.\s{2}", t)
    if m:
        candidatos.append(t[:m.start()])
    for c in candidatos:
        c = c.strip()
        if TITULO_MIN <= len(c) <= TITULO_MAX:
            return normalizar_fonte(c)[0]
    return None


def carregar_fakebr():
    nums = sorted(int(f[:-4]) for f in os.listdir(os.path.join(PASTA_FAKEBR, "fake"))
                  if re.fullmatch(r"\d+\.txt", f))
    pares, ilegiveis = [], []
    for n in nums:
        par = {"n": n}
        try:
            for classe in ("fake", "true"):
                bruto = ler(os.path.join(PASTA_FAKEBR, classe, f"{n}.txt"))
                par[classe] = {"bruto": bruto, **meta(classe, n)}
        except Exception as e:                           # relatado, não escondido
            ilegiveis.append((n, type(e).__name__))
            continue
        pares.append(par)
    return pares, ilegiveis


def treinar():
    log(f"lendo {os.path.basename(ARQ_CORPUS)} e treinando o modelo interno (C={C_FIXO})...")
    with open(ARQ_CORPUS, encoding="utf-8") as f:
        corpus = [json.loads(l) for l in f if l.strip()]
    treino = [r for r in corpus if r["subconjunto_pareado"] and r["split"] == "treino"]
    teste = [r for r in corpus if r["subconjunto_pareado"] and r["split"] == "teste"]
    vet = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Xtr = vet.fit_transform([r["texto_entrada"] for r in treino])
    ytr = [1 if r["veredito"] == "FALSO" else 0 for r in treino]
    modelo = LogisticRegression(C=C_FIXO, max_iter=2000).fit(Xtr, ytr)
    interno = balanced_accuracy_score([1 if r["veredito"] == "FALSO" else 0 for r in teste],
                                      modelo.predict(vet.transform([r["texto_entrada"] for r in teste])))
    log(f"  treino: {len(treino):,} registros | teste interno (conferência): {interno:.2%}"
        .replace(",", "."))
    if abs(interno - REFERENCIA_INTERNA) > 0.002:
        log(f"  [!] difere da referência {REFERENCIA_INTERNA:.2%} -- o corpus mudou desde o modelo_bruto")
    return vet, modelo, interno


def avaliar(nome, vet, modelo, itens):
    """itens: lista de (texto, rotulo 1=FALSO, categoria)."""
    X = vet.transform([t for t, _, _ in itens])
    y = [r for _, r, _ in itens]
    pred = modelo.predict(X)
    bal = balanced_accuracy_score(y, pred)
    rec_f = recall_score(y, pred, pos_label=1)
    rec_v = recall_score(y, pred, pos_label=0)
    log(f"  {nome}: acurácia balanceada {bal:.2%} | revocação FALSO {rec_f:.2%} | "
        f"revocação VERDADEIRO {rec_v:.2%}")
    por_cat = {}
    grupos = defaultdict(list)
    for i, (_, _, cat) in enumerate(itens):
        grupos[cat].append(i)
    for cat, idx in sorted(grupos.items(), key=lambda x: -len(x[1])):
        yc = [y[i] for i in idx]
        if len(set(yc)) < 2:
            continue
        b = balanced_accuracy_score(yc, [pred[i] for i in idx])
        por_cat[cat] = {"acuracia_balanceada": round(b, 4), "n": len(idx)}
        log(f"      {cat:22s} {b:6.1%}  (n={len(idx)})")
    return {"acuracia_balanceada": round(bal, 4), "revocacao_falso": round(rec_f, 4),
            "revocacao_verdadeiro": round(rec_v, 4), "n": len(itens), "por_categoria": por_cat}


def main():
    log("AVALIAÇÃO CRUZADA NO FAKE.BR -- braço bruto (pré-registro, commit 954a2d9)")
    for a in AMBIGUIDADES:
        log(f"  AMBIGUIDADE: {a}")
    log("")

    vet, modelo, interno = treinar()

    log("")
    log("carregando o Fake.br...")
    pares, ilegiveis = carregar_fakebr()
    log(f"  {len(pares)} pares lidos | ilegíveis: {len(ilegiveis)} {ilegiveis[:5]}")

    # ---- entrada A ----
    log("")
    log("ENTRADA A -- texto (todos os pares)")
    itens_a = []
    for p in pares:
        for classe, rot in (("fake", 1), ("true", 0)):
            itens_a.append((entrada_texto(p[classe]["bruto"]), rot, p[classe]["categoria"]))
    comp = {rot: statistics.median(len(t) for t, r, _ in itens_a if r == rot) for rot in (0, 1)}
    log(f"  mediana de caracteres da entrada: falsas {int(comp[1])} | verdadeiras {int(comp[0])}")
    res_a = avaliar("texto", vet, modelo, itens_a)

    # ---- entrada B ----
    log("")
    log("ENTRADA B -- título (só pares com título identificável dos dois lados)")
    itens_b, com_titulo = [], Counter()
    for p in pares:
        tf, tv = titulo(p["fake"]["bruto"]), titulo(p["true"]["bruto"])
        com_titulo["fake"] += tf is not None
        com_titulo["true"] += tv is not None
        if tf and tv:
            itens_b.append((tf, 1, p["fake"]["categoria"]))
            itens_b.append((tv, 0, p["true"]["categoria"]))
    log(f"  com título: falsas {com_titulo['fake']} | verdadeiras {com_titulo['true']} | "
        f"pares com os dois: {len(itens_b) // 2} de {len(pares)}")
    res_b = avaliar("título", vet, modelo, itens_b) if itens_b else None

    # ---- confronto com o pré-registro ----
    a, b = res_a["acuracia_balanceada"], (res_b or {}).get("acuracia_balanceada")
    queda = (interno - a) * 100
    dif_rec = (res_a["revocacao_verdadeiro"] - res_a["revocacao_falso"]) * 100
    log("")
    log("=" * 72)
    log("CONFRONTO COM O PRÉ-REGISTRO")
    log("=" * 72)
    linhas = [
        ("P1 acurácia bal. texto", f"{a:.2%}", "55–70%", 0.55 <= a <= 0.70, "~55%", abs(a - 0.55) <= 0.03),
        ("P2 queda vs. interno", f"{queda:.1f} pp", "≥ 20 pp", queda >= 20, "~35 pp", abs(queda - 35) <= 5),
        ("P3 rev.V − rev.F (texto)", f"{dif_rec:+.1f} pp", "≥ +20 pp", dif_rec >= 20, "parecidas", abs(dif_rec) < 10),
    ]
    if b is not None:
        d = (b - a) * 100
        linhas.append(("P4 título − texto", f"{d:+.1f} pp", "≥ +5 pp", d >= 5, "título pior", d < 0))
    log(f"  {'previsão':26s} {'resultado':>11s}   {'seção 2':>9s}      {'pesquisador':>12s}")
    for nome, res, p2, ok2, pp, okp in linhas:
        log(f"  {nome:26s} {res:>11s}   {p2:>9s} {'✓' if ok2 else '✗'}    {pp:>12s} {'✓' if okp else '✗'}")
    log("  (P5 -- categorias -- ver a tabela da entrada A acima)")

    faixas = [(0.85, "≥ 85%: refuta a leitura principal -- há sinal que transfere além do gênero."),
              (0.70, "70–85%: transferência parcial -- parte conteúdo/sensacionalismo, parte gênero."),
              (0.55, "55–70%: confirma -- o modelo aprendeu majoritariamente gênero e fonte."),
              (0.45, "45–55%: nada transfere -- não separa gênero de mudança de definição."),
              (0.00, "< 45%: o modelo inverte as classes -- dependência forte de gênero.")]
    leitura = next(t for lim, t in faixas if a >= lim)
    log("")
    log(f"  LEITURA PRÉ-REGISTRADA (entrada A = {a:.2%}): {leitura}")
    log("=" * 72)

    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump({"preregistro_commit": "954a2d9", "interno": round(interno, 4),
                   "ambiguidades": AMBIGUIDADES, "ilegiveis": ilegiveis,
                   "entrada_A_texto": res_a, "entrada_B_titulo": res_b,
                   "titulos_identificados": dict(com_titulo), "leitura": leitura},
                  f, indent=2, ensure_ascii=False)
    log(f"resultados salvos em relatorios/{os.path.basename(SAIDA)}")


if __name__ == "__main__":
    main()
