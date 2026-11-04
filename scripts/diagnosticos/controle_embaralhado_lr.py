"""Diagnóstico do controle com rótulos embaralhados (exploratório, não altera a leitura pré-registrada)."""
import sys, json, random
import numpy as np
sys.path.insert(0, "scripts")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from comparativo_bracos import rotulos, bal

regs = [json.loads(l) for l in open("datasets/braco_refinado.jsonl", encoding="utf-8") if l.strip()]
por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "teste")}
vet = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True).fit([r["texto_entrada"] for r in por["treino"]])
Xtr, Xte = vet.transform([r["texto_entrada"] for r in por["treino"]]), vet.transform([r["texto_entrada"] for r in por["teste"]])
y, yte = rotulos(por["treino"]), rotulos(por["teste"])


def embaralhar(s):
    z = list(y); random.Random(s).shuffle(z); return np.array(z)


def concordancia_exata(s):
    """Embaralhado com concordância EXATA de 50% com o rótulo real, dentro de cada classe."""
    rng, z = random.Random(s), y.copy()
    for c in (0, 1):
        idx = [i for i in range(len(y)) if y[i] == c]
        rng.shuffle(idx)
        for i in idx[: len(idx) // 2]:
            z[i] = 1 - c
    return z


for s in (0, 1, 2):
    z = embaralhar(s)
    linha = [f"semente {s}: concordância com o real {100 * (z == y).mean():.2f}%"]
    for C in (0.1, 10.0):
        p = LogisticRegression(C=C, max_iter=2000).fit(Xtr, z).predict(Xte)
        linha.append(f"C={C}: {100 * bal(yte, p):.2f}%")
    z2 = concordancia_exata(s)
    p = LogisticRegression(C=0.1, max_iter=2000).fit(Xtr, z2).predict(Xte)
    linha.append(f"| concordância exata 50% ({100 * (z2 == y).mean():.2f}%), C=0.1: {100 * bal(yte, p):.2f}%")
    print("  ".join(linha))

print("\n20 sementes (3 a 22), embaralhamento simples:")
for C in (0.1, 10.0):
    vals = [100 * bal(yte, LogisticRegression(C=C, max_iter=2000).fit(Xtr, embaralhar(s)).predict(Xte)) for s in range(3, 23)]
    v = np.array(vals)
    print(f"  C={C}: média {v.mean():.2f}% | desvio {v.std():.2f} | mín {v.min():.2f} | máx {v.max():.2f} | acima de 50: {(v > 50).sum()}/20 | fora de 45–55: {((v < 45) | (v > 55)).sum()}/20")
