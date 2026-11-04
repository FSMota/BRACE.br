"""IC 95% do modelo só de forma no teste interno (cálculo posterior, 04/11/2026).

    python scripts/diagnosticos/ic_estilo_interno.py

modelo_estilo.py calculou o IC só no Fake.br, a pergunta daquele teste. Este
diagnóstico, pedido na revisão do TCC, completa o IC no teste interno.
Mesmo treino, mesmos modelos e mesma semente de modelo_estilo.py; bootstrap por
item com 2.000 reamostragens, como no modelo de referência."""
import os, sys, json
import numpy as np
RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))
os.chdir(RAIZ)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
import modelo_estilo as me

corpus = [json.loads(l) for l in open(me.ARQ_CORPUS, encoding="utf-8") if l.strip()]
par = [r for r in corpus if r["subconjunto_pareado"]]
tr = [r for r in par if r["split"] == "treino"]
te = [r for r in par if r["split"] == "teste"]
Xtr = np.array([me.atributos(r["texto_entrada"]) for r in tr]); ytr = np.array([r["veredito"] == "FALSO" for r in tr], int)
Xte = np.array([me.atributos(r["texto_entrada"]) for r in te]); yte = np.array([r["veredito"] == "FALSO" for r in te], int)
esc = StandardScaler().fit(Xtr)
mods = {"regressao_logistica": (LogisticRegression(max_iter=2000).fit(esc.transform(Xtr), ytr), True),
        "gradient_boosting": (HistGradientBoostingClassifier(random_state=me.SEED).fit(Xtr, ytr), False)}
rng = np.random.default_rng(0)
idx = rng.integers(0, len(yte), size=(2000, len(yte)))
for nome, (m, e) in mods.items():
    p = m.predict(esc.transform(Xte) if e else Xte)
    a = (p == yte)
    y = yte[idx]; ac = a[idx]; pos = y == 1
    bal = 0.5 * ((ac & pos).sum(1) / pos.sum(1) + (ac & ~pos).sum(1) / (~pos).sum(1))
    print(nome, f"{100*balanced_accuracy_score(yte, p):.2f}", [round(100*np.percentile(bal, q), 2) for q in (2.5, 97.5)])
