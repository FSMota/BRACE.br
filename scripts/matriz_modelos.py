# -*- coding: utf-8 -*-
"""Matriz de modelos: as conclusões valem para outras representações e classificadores?
Pré-registro: `relatorios/PREREGISTRO_MATRIZ_MODELOS.md`.

    python scripts/matriz_modelos.py --fumaca [--parte P]       # amostra pequena; não mostra métricas
    python scripts/matriz_modelos.py --parte tfidf               # P e Ch × LR, SVM, NB (rodada em 13/10/2026)
    python scripts/matriz_modelos.py --parte densa               # Bc e Ol × LR, SVM, NB (vetores do PC com GPU)
    python scripts/matriz_modelos.py --controle --parte tfidf    # controle refeito (desvio declarado, ver abaixo)
    python scripts/matriz_modelos.py --controle --parte densa

O BERTimbau ajustado (Ba) roda no PC com GPU (`scripts/ajustar_bertimbau.py`) e
grava as predições no mesmo formato. A apuração das afirmações A1–A8 junta
tudo (`scripts/apurar_matriz.py`).

Escolhas fixadas no pré-registro (§1): grade escolhida na `validacao` (nossa
direção) ou na média de 5 dobras por par do Fake.br (direção Fake.br); nada é
ajustado na base de teste.

CONTROLE REFEITO (desvio declarado, 13/10/2026, depois da rodada tfidf): o
controle do pré-registro (rótulos embaralhados com semente 0, grade escolhida
na validação) saiu fora de 45–55% em 5 de 6 modelos. Um diagnóstico com 20
sementes mostrou que não há vazamento (média 48,4%), mas que, com a grade
escolhida na validação, o modelo embaralhado escolhe a regularização mais forte
e oscila muito (desvio 4,8 pp). `--controle` refaz o controle como no
comparativo dos braços: 3 sementes (0, 1, 2), com o hiperparâmetro do modelo
real, sem escolha na validação. O resultado original continua relatado.
"""
import os
import sys
import json
import random

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparativo_bracos import rotulos, bal, pct, SEED                      # noqa: E402
from avaliacao_inversa import dobras_por_par, N_DOBRAS                      # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
PASTA = os.path.join(_RAIZ, "relatorios", "matriz")
PASTA_VETORES = os.path.join(_RAIZ, "datasets", "vetores")
BRACOS = {"bruto": "texto_entrada", "refinado": "ideia_central"}
SEMENTES_CONTROLE = [0, 1, 2]
PARTES = {"tfidf": ("P", "Ch"), "densa": ("Bc", "Ol")}
ARQ_VETORES = {"Bc": "vetores_bertimbau.npz", "Ol": "vetores_bge-m3.npz"}
TFIDF = {
    "P": lambda: TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True),
    "Ch": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True),
}


def grade(rep, clf):
    if clf == "LR":
        return [0.1, 1.0, 10.0]
    if clf == "SVM":
        return [0.01, 0.1, 1.0]
    return [0.1, 0.5, 1.0] if rep in TFIDF else [None]      # NB gaussiano: sem grade (§1.3)


def classificador(rep, clf, h):
    if clf == "LR":
        return LogisticRegression(C=h, max_iter=2000)
    if clf == "SVM":
        return LinearSVC(C=h)
    return MultinomialNB(alpha=h) if rep in TFIDF else GaussianNB()


_VETORES = {}


def vetores(rep, campo):
    """Vetores densos por id, extraídos no PC com GPU (`scripts/extrair_vetores.py`)."""
    if rep not in _VETORES:
        z = np.load(os.path.join(PASTA_VETORES, ARQ_VETORES[rep]), allow_pickle=False)
        _VETORES[rep] = {"ids": {i: k for k, i in enumerate(z["ids"])},
                         "texto_entrada": z["texto_entrada"], "ideia_central": z["ideia_central"]}
    v = _VETORES[rep]
    return lambda regs: v[campo][[v["ids"][r["id"]] for r in regs]]


def ajustar(rep, clf, h, treino, campo, y=None):
    if rep in TFIDF:
        vet = TFIDF[rep]().fit([r[campo] for r in treino])
        transformar = lambda regs: vet.transform([r[campo] for r in regs])    # noqa: E731
    else:
        transformar = vetores(rep, campo)
    m = classificador(rep, clf, h).fit(transformar(treino), rotulos(treino) if y is None else y)
    return lambda regs: m.predict(transformar(regs))


def embaralhados(treino, semente):
    y = list(rotulos(treino))
    random.Random(semente).shuffle(y)
    return np.array(y)


def nossa_direcao(rep, clf, por, campo, embaralhar=False):
    y = embaralhados(por["treino"], SEED) if embaralhar else None
    melhor = None
    for h in grade(rep, clf):
        prever = ajustar(rep, clf, h, por["treino"], campo, y)
        b = bal(rotulos(por["validacao"]), prever(por["validacao"]))
        if melhor is None or b > melhor[0]:
            melhor = (b, h, prever)
    return melhor[1], melhor[2]


def direcao_fakebr(rep, clf, fakebr, campo, dobras):
    y = rotulos(fakebr)
    melhor = None
    for h in grade(rep, clf):
        pred, bals = np.zeros(len(fakebr), dtype=int), []
        for k in range(N_DOBRAS):
            tr = [r for r, d in zip(fakebr, dobras) if d != k]
            idx = [i for i, d in enumerate(dobras) if d == k]
            pred[idx] = ajustar(rep, clf, h, tr, campo)([fakebr[i] for i in idx])
            bals.append(bal(y[idx], pred[idx]))
        if melhor is None or np.mean(bals) > melhor[0]:
            melhor = (float(np.mean(bals)), h, pred.copy())
    return melhor[1], melhor[2], ajustar(rep, clf, melhor[1], fakebr, campo)


def carregar(fumaca):
    regs = [json.loads(l) for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()]
    por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste", "fakebr")}
    if fumaca:
        rnd = random.Random(1)
        por = {s: rnd.sample(v, min(len(v), 400)) for s, v in por.items()}
        pares = {int(r["id"].rsplit("-", 1)[1]) for r in por["fakebr"]}
        todos = [r for r in regs if r["split"] == "fakebr"]
        por["fakebr"] = [r for r in todos if int(r["id"].rsplit("-", 1)[1]) in pares][:400]
        print("FUMAÇA: amostras de até 400 por parte; nenhuma métrica é mostrada nem salva")
    return por


def rodar_parte(parte, fumaca):
    por = carregar(fumaca)
    dobras = dobras_por_par(por["fakebr"])
    y_te, y_fb = rotulos(por["teste"]), rotulos(por["fakebr"])
    os.makedirs(PASTA, exist_ok=True)
    resumo = {}
    for rep in PARTES[parte]:
        for clf in ("LR", "SVM", "NB"):
            for braco, campo in BRACOS.items():
                nome = f"{rep}+{clf}|{braco}"
                h_n, prever_n = nossa_direcao(rep, clf, por, campo)
                p_te, p_fb = prever_n(por["teste"]), prever_n(por["fakebr"])
                h_f, p_fbfb, prever_f = direcao_fakebr(rep, clf, por["fakebr"], campo, dobras)
                p_fbte = prever_f(por["teste"])
                linha = {"h_nosso": h_n, "h_fakebr": h_f}
                if braco == "bruto":
                    h_e, prever_e = nossa_direcao(rep, clf, por, campo, embaralhar=True)
                    p_emb = prever_e(por["teste"])
                    linha["h_embaralhado"] = h_e
                if fumaca:
                    print(f"  {nome}: rodou (hiperparâmetros {linha})")
                    continue
                linha.update({"nosso_teste": pct(bal(y_te, p_te)), "nosso_fakebr": pct(bal(y_fb, p_fb)),
                              "fakebr_fakebr": pct(bal(y_fb, p_fbfb)), "fakebr_nosso": pct(bal(y_te, p_fbte))})
                salvar = {"teste_ids": np.array([r["id"] for r in por["teste"]]),
                          "fakebr_ids": np.array([r["id"] for r in por["fakebr"]]),
                          "nosso_teste": p_te, "nosso_fakebr": p_fb, "fakebr_fakebr": p_fbfb, "fakebr_nosso": p_fbte}
                if braco == "bruto":
                    linha["embaralhado_teste"] = pct(bal(y_te, p_emb))
                    salvar["embaralhado_teste"] = p_emb
                np.savez_compressed(os.path.join(PASTA, f"pred_{rep}_{clf}_{braco}.npz"), **salvar)
                resumo[nome] = linha
                print(f"  {nome:16s} nosso→teste {linha['nosso_teste']:6.2f} | nosso→Fake.br {linha['nosso_fakebr']:6.2f} | "
                      f"Fake.br→Fake.br {linha['fakebr_fakebr']:6.2f} | Fake.br→nosso {linha['fakebr_nosso']:6.2f}"
                      + (f" | embaralhado {linha['embaralhado_teste']:.2f}" if braco == "bruto" else ""))
    if not fumaca:
        with open(os.path.join(PASTA, f"resumo_{parte}.json"), "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
        print(f"salvo em {PASTA}")


def rodar_controle(parte, fumaca):
    """3 sementes, hiperparâmetro do modelo real (o `h_nosso` já gravado), braço bruto, nossa direção."""
    por = carregar(fumaca)
    y_te = rotulos(por["teste"])
    resumo_real = {} if fumaca else json.load(open(os.path.join(PASTA, f"resumo_{parte}.json"), encoding="utf-8"))
    saida = {}
    for rep in PARTES[parte]:
        for clf in ("LR", "SVM", "NB"):
            nome = f"{rep}+{clf}|bruto"
            h = grade(rep, clf)[-1] if fumaca else resumo_real[nome]["h_nosso"]
            accs = []
            for s in SEMENTES_CONTROLE:
                prever = ajustar(rep, clf, h, por["treino"], "texto_entrada", embaralhados(por["treino"], s))
                accs.append(pct(bal(y_te, prever(por["teste"]))))
            if fumaca:
                print(f"  {nome}: controle rodou")
                continue
            saida[nome] = {"h": h, "acuracias": accs, "dentro_45_55": all(45 <= a <= 55 for a in accs)}
            print(f"  {nome:16s} h={h} | embaralhado {accs} | {'ok' if saida[nome]['dentro_45_55'] else 'FORA'}")
    if not fumaca:
        with open(os.path.join(PASTA, f"controle_{parte}.json"), "w", encoding="utf-8") as f:
            json.dump(saida, f, indent=2, ensure_ascii=False)
        print(f"salvo em {PASTA}")


def main():
    fumaca = "--fumaca" in sys.argv
    parte = sys.argv[sys.argv.index("--parte") + 1] if "--parte" in sys.argv else "tfidf"
    if parte not in PARTES:
        sys.exit(f"parte desconhecida: {parte}")
    if "--controle" in sys.argv:
        rodar_controle(parte, fumaca)
    else:
        rodar_parte(parte, fumaca)


if __name__ == "__main__":
    main()
