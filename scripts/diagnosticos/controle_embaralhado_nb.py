"""Diagnóstico do controle no Naive Bayes (exploratório): 20 sementes, alfa do modelo real."""
import sys, json, random
import numpy as np
sys.path.insert(0, "scripts")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from comparativo_bracos import rotulos, bal

regs = [json.loads(l) for l in open("datasets/braco_refinado.jsonl", encoding="utf-8") if l.strip()]
por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "teste")}
y, yte = rotulos(por["treino"]), rotulos(por["teste"])
for nome, vet, alfa in (("P+NB", TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True), 0.5),
                        ("Ch+NB", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True), 0.1)):
    vet.fit([r["texto_entrada"] for r in por["treino"]])
    Xtr, Xte = vet.transform([r["texto_entrada"] for r in por["treino"]]), vet.transform([r["texto_entrada"] for r in por["teste"]])
    vals, prop = [], []
    for s in range(3, 23):
        z = list(y); random.Random(s).shuffle(z)
        p = MultinomialNB(alpha=alfa).fit(Xtr, np.array(z)).predict(Xte)
        vals.append(100 * bal(yte, p)); prop.append(100 * p.mean())
    v = np.array(vals)
    print(f"{nome}: média {v.mean():.2f}% | desvio {v.std():.2f} | mín {v.min():.2f} | máx {v.max():.2f} | "
          f"acima de 50: {(v > 50).sum()}/20 | fora de 45–55: {((v < 45) | (v > 55)).sum()}/20 | "
          f"% previsto FALSO: {np.mean(prop):.1f}")
