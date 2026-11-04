# -*- coding: utf-8 -*-
"""Primeiro modelo: braço BRUTO da RQ1 -- TF-IDF + regressão logística.

    python scripts/modelo_bruto.py

Entrada: `texto_entrada` do `corpus_fundido.jsonl`, só `subconjunto_pareado`.
Treina no treino, escolhe o C na validação, reporta no teste (divisão temporal).

Duas rodadas, e a diferença entre elas é o resultado que interessa:
  A. pareado completo
  B. pareado SEM `entrada_com_veredito` -- quanto da acurácia vinha do
     vocabulário do checador que sobrou na entrada (RQ4)

O número alto aqui NÃO é detecção de desinformação: o baseline de exclamação
sozinho já dava 68%. As palavras de maior peso mostram o que o modelo está
usando -- é isso que responde se ele aprendeu veracidade ou gênero textual.
"""
import os
import sys
import json
import time
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_modelo_bruto.json")
VALORES_C = [0.1, 1.0, 10.0]
TOP_PALAVRAS = 25
T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def rodar(nome, regs):
    log("=" * 64)
    log(f"RODADA {nome}")
    log("=" * 64)
    por_split = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste")}
    for s, rs in por_split.items():
        c = Counter(r["veredito"] for r in rs)
        log(f"  {s:10s} {len(rs):6d} registros  (falsas {c['FALSO']}, verdadeiras {c['VERDADEIRO']})")

    X = {s: [r["texto_entrada"] for r in rs] for s, rs in por_split.items()}
    y = {s: [1 if r["veredito"] == "FALSO" else 0 for r in rs] for s, rs in por_split.items()}

    log("vetorizando (TF-IDF, palavras e pares de palavras, min_df=3)...")
    vet = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, strip_accents=None)
    M = {"treino": vet.fit_transform(X["treino"])}
    M["validacao"], M["teste"] = vet.transform(X["validacao"]), vet.transform(X["teste"])
    log(f"  vocabulário: {len(vet.vocabulary_):,} termos".replace(",", "."))

    melhor = None
    for C in VALORES_C:
        log(f"treinando regressão logística com C={C}...")
        m = LogisticRegression(C=C, max_iter=2000)
        m.fit(M["treino"], y["treino"])
        bal = balanced_accuracy_score(y["validacao"], m.predict(M["validacao"]))
        log(f"  validação: acurácia balanceada {bal:.2%}")
        if not melhor or bal > melhor[0]:
            melhor = (bal, C, m)
    _, C, modelo = melhor
    log(f"escolhido C={C} pela validação")

    pred = modelo.predict(M["teste"])
    bal = balanced_accuracy_score(y["teste"], pred)
    log("")
    log(f">>> TESTE: acurácia balanceada {bal:.2%}")
    log("")
    print(classification_report(y["teste"], pred, target_names=["VERDADEIRO", "FALSO"], digits=3), flush=True)
    cm = confusion_matrix(y["teste"], pred)
    log(f"matriz de confusão (linhas = real, colunas = previsto) [VERDADEIRO, FALSO]:")
    log(f"  VERDADEIRO {cm[0][0]:6d} {cm[0][1]:6d}")
    log(f"  FALSO      {cm[1][0]:6d} {cm[1][1]:6d}")

    # Acerto por tema: se um tema vai muito melhor, o modelo pode estar
    # separando por assunto, não por veracidade.
    log("")
    log("acurácia no teste por tema:")
    por_tema = {}
    for t in sorted({r["tema_pareamento"] for r in por_split["teste"]}):
        idx = [i for i, r in enumerate(por_split["teste"]) if r["tema_pareamento"] == t]
        if len(idx) >= 20:
            acc = sum(pred[i] == y["teste"][i] for i in idx) / len(idx)
            por_tema[t] = round(acc, 4)
            log(f"  {t:22s} {acc:6.1%}  (n={len(idx)})")

    nomes = vet.get_feature_names_out()
    coef = modelo.coef_[0]
    ordem = coef.argsort()
    falso = [(nomes[i], round(float(coef[i]), 2)) for i in ordem[::-1][:TOP_PALAVRAS]]
    verdadeiro = [(nomes[i], round(float(coef[i]), 2)) for i in ordem[:TOP_PALAVRAS]]
    log("")
    log(f"{TOP_PALAVRAS} termos que mais empurram para FALSO:")
    log("  " + ", ".join(f"{t} ({c})" for t, c in falso))
    log(f"{TOP_PALAVRAS} termos que mais empurram para VERDADEIRO:")
    log("  " + ", ".join(f"{t} ({c})" for t, c in verdadeiro))
    return {"acuracia_balanceada_teste": round(bal, 4), "C": C,
            "n": {s: len(rs) for s, rs in por_split.items()},
            "por_tema": por_tema, "top_falso": falso, "top_verdadeiro": verdadeiro}


def main():
    log(f"lendo {os.path.basename(ARQ_CORPUS)}...")
    with open(ARQ_CORPUS, encoding="utf-8") as f:
        corpus = [json.loads(l) for l in f if l.strip()]
    pareado = [r for r in corpus if r["subconjunto_pareado"]]
    log(f"  {len(corpus):,} registros; {len(pareado):,} no subconjunto pareado".replace(",", "."))

    resultados = {"A_pareado_completo": rodar("A -- pareado completo", pareado)}
    sem = [r for r in pareado if not r["entrada_com_veredito"]]
    resultados["B_sem_veredito"] = rodar("B -- pareado SEM entrada_com_veredito", sem)

    a = resultados["A_pareado_completo"]["acuracia_balanceada_teste"]
    b = resultados["B_sem_veredito"]["acuracia_balanceada_teste"]
    log("")
    log("=" * 64)
    log(f"RESUMO   A (completo) {a:.2%}   |   B (sem veredito) {b:.2%}   |   diferença {100 * (a - b):+.2f} pp")
    log("=" * 64)
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    log(f"resultados salvos em relatorios/{os.path.basename(SAIDA)}")


if __name__ == "__main__":
    main()
