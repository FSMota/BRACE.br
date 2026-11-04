# -*- coding: utf-8 -*-
"""Controles feitos para FALHAR se o pipeline de avaliação estiver quebrado.

    python scripts/controles_pipeline.py

Não medem se o modelo é bom; medem se o número do `modelo_bruto.py` pode ser
levado a sério. Cada controle tem um resultado ESPERADO escrito antes de rodar.

  1. SOBREPOSIÇÃO ENTRE PARTES -- nenhum id, URL ou texto de entrada pode
     aparecer em duas partes (treino/validação/teste), e nenhum grupo de
     duplicatas pode atravessar a fronteira. Esperado: zero em tudo.
     Se falhar: o modelo foi avaliado em parte do que viu no treino.

  2. RÓTULOS EMBARALHADOS -- mesmo vetorizador, mesmo modelo, mesmo C do
     modelo_bruto, mas com os rótulos de TREINO sorteados. O teste continua
     com os rótulos reais. Esperado: ~50% (entre 48% e 52%).
     Se der mais: existe caminho da resposta até o modelo que não passa pelo
     texto -- vazamento no pipeline, e o 92% não vale nada.

Roda sobre o mesmo recorte do experimento: `subconjunto_pareado`.
"""
import os
import re
import json
import time
import random
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_controles.json")
SPLITS = ("treino", "validacao", "teste")
C_MODELO = 10.0                 # o escolhido pela validação no modelo_bruto
SEMENTES_EMBARALHAR = [0, 1, 2]
FAIXA_ACASO = (0.48, 0.52)
T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def norm(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


def controle_sobreposicao(regs):
    log("=" * 64)
    log("CONTROLE 1 -- sobreposição entre treino, validação e teste")
    log("=" * 64)
    falhas = {}
    for rotulo, chave in (("id", lambda r: r["id"]),
                          ("url + texto", lambda r: (r["url"], norm(r["texto_entrada"]))),
                          ("texto de entrada", lambda r: norm(r["texto_entrada"]))):
        onde = defaultdict(set)
        for r in regs:
            onde[chave(r)].add(r["split"])
        cruzam = sum(1 for s in onde.values() if len(s) > 1)
        falhas[rotulo] = cruzam
        log(f"  {rotulo:18s} em mais de uma parte: {cruzam:5d}   {'OK' if cruzam == 0 else 'FALHOU'}")

    # A mesma URL em partes diferentes com textos diferentes é esperada no lado
    # falso (artigo-lista: um artigo, vários boatos), mas precisa ser contada.
    url_split = defaultdict(set)
    for r in regs:
        url_split[r["url"]].add(r["split"])
    so_url = sum(1 for s in url_split.values() if len(s) > 1)
    falhas["url (texto diferente)"] = so_url
    log(f"  {'url, texto diferente':18s} em mais de uma parte: {so_url:5d}   "
        f"{'OK' if so_url == 0 else 'ATENÇÃO -- ver nota'}")

    # Grupos de quase-duplicatas: marcados no corpus inteiro, então checa-se
    # no corpus inteiro -- o pareado usa só canônicos, mas o grupo vale para todos.
    grupos = defaultdict(set)
    for r in regs:
        if r.get("grupo_duplicata"):
            grupos[(r["veredito"], r["grupo_duplicata"])].add(r["split"])
    atravessam = sum(1 for s in grupos.values() if len(s) > 1)
    falhas["grupo de duplicata"] = atravessam
    log(f"  {'grupo de duplicata':18s} atravessando partes: {atravessam:5d}   "
        f"{'OK' if atravessam == 0 else 'FALHOU'}")

    # Quase-duplicata que o MinHash não pegou: mesmos primeiros 120 caracteres.
    ini = defaultdict(set)
    for r in regs:
        t = norm(r["texto_entrada"])
        if len(t) >= 120:
            ini[t[:120]].add(r["split"])
    prefixo = sum(1 for s in ini.values() if len(s) > 1)
    falhas["mesmo início (120c)"] = prefixo
    log(f"  {'mesmo início (120c)':18s} em mais de uma parte: {prefixo:5d}   "
        f"{'OK' if prefixo == 0 else 'ATENÇÃO'}")
    return falhas


def controle_embaralhado(regs):
    log("")
    log("=" * 64)
    log("CONTROLE 2 -- rótulos de treino embaralhados (esperado: ~50%)")
    log("=" * 64)
    por = {s: [r for r in regs if r["split"] == s] for s in SPLITS}
    X = {s: [r["texto_entrada"] for r in rs] for s, rs in por.items()}
    y = {s: [1 if r["veredito"] == "FALSO" else 0 for r in rs] for s, rs in por.items()}

    log("vetorizando (mesma configuração do modelo_bruto)...")
    vet = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    Mtr, Mte = vet.fit_transform(X["treino"]), vet.transform(X["teste"])

    log("referência: rótulos REAIS...")
    real = balanced_accuracy_score(y["teste"], LogisticRegression(C=C_MODELO, max_iter=2000)
                                   .fit(Mtr, y["treino"]).predict(Mte))
    log(f"  rótulos reais ........... {real:.2%}")

    resultados = []
    for semente in SEMENTES_EMBARALHAR:
        ytr = y["treino"][:]
        random.Random(semente).shuffle(ytr)
        acc = balanced_accuracy_score(y["teste"], LogisticRegression(C=C_MODELO, max_iter=2000)
                                      .fit(Mtr, ytr).predict(Mte))
        resultados.append(acc)
        ok = FAIXA_ACASO[0] <= acc <= FAIXA_ACASO[1]
        log(f"  embaralhado (semente {semente}) .. {acc:.2%}   {'OK' if ok else 'FALHOU'}")
    return {"reais": round(real, 4), "embaralhados": [round(a, 4) for a in resultados]}


def main():
    log(f"lendo {os.path.basename(ARQ_CORPUS)}...")
    with open(ARQ_CORPUS, encoding="utf-8") as f:
        corpus = [json.loads(l) for l in f if l.strip()]
    pareado = [r for r in corpus if r["subconjunto_pareado"]]
    log(f"  {len(pareado):,} registros no pareado".replace(",", "."))
    log("")

    s1 = controle_sobreposicao(pareado)
    log("")
    log("  mesmo controle no CORPUS INTEIRO (inclui duplicatas marcadas e fora do pareado):")
    s1_total = controle_sobreposicao(corpus)
    s2 = controle_embaralhado(pareado)

    log("")
    log("=" * 64)
    graves = s1["id"] + s1["url + texto"] + s1["texto de entrada"] + s1["grupo de duplicata"]
    emb_ok = all(FAIXA_ACASO[0] <= a <= FAIXA_ACASO[1] for a in s2["embaralhados"])
    log(f"VEREDITO   sobreposição grave: {'nenhuma' if graves == 0 else graves}   |   "
        f"embaralhado: {'OK (~50%)' if emb_ok else 'FALHOU'}")
    log("=" * 64)
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump({"sobreposicao_pareado": s1, "sobreposicao_corpus": s1_total,
                   "embaralhado": s2}, f, indent=2, ensure_ascii=False)
    log(f"resultados salvos em relatorios/{os.path.basename(SAIDA)}")


if __name__ == "__main__":
    main()
