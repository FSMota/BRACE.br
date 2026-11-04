"""Diagnóstico do controle no NB gaussiano (exploratório, 14/10/2026, depois da rodada densa).

Hipótese: com rótulos embaralhados, as duas "classes" têm médias quase iguais,
mas variâncias estimadas um pouco diferentes. Em 768–1024 dimensões, essas
pequenas diferenças se somam, e o NB gaussiano passa a decidir pela DISTÂNCIA
ao centro dos dados (fronteira quadrática), não pela direção. Se a distância
ao centro já separar corrente de notícia, o controle sai acima do acaso sem
nenhum vazamento de rótulo. Teste: (1) a distância ao centro, sozinha, separa
as classes? (2) as predições do NB embaralhado acompanham essa distância?
Rodar da raiz do repositório.
"""
import sys, json, random
import numpy as np
sys.path.insert(0, "scripts")
from sklearn.naive_bayes import GaussianNB
from comparativo_bracos import rotulos, bal

regs = [json.loads(l) for l in open("datasets/braco_refinado.jsonl", encoding="utf-8") if l.strip()]
por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "teste")}
y, yte = rotulos(por["treino"]), rotulos(por["teste"])
for nome, arq in (("Bc", "vetores_bertimbau.npz"), ("Ol", "vetores_bge-m3.npz")):
    z = np.load(f"datasets/vetores/{arq}")
    pos = {i: k for k, i in enumerate(z["ids"])}
    Xtr = z["texto_entrada"][[pos[r["id"]] for r in por["treino"]]]
    Xte = z["texto_entrada"][[pos[r["id"]] for r in por["teste"]]]
    centro, escala = Xtr.mean(0), Xtr.std(0) + 1e-9
    dist_tr = (((Xtr - centro) / escala) ** 2).sum(1)
    dist = (((Xte - centro) / escala) ** 2).sum(1)
    limiar = np.median(dist_tr)
    b = 100 * bal(yte, (dist > limiar).astype(int))
    print(f"{nome}: só a distância ao centro (limiar = mediana do treino): {max(b, 100 - b):.2f}% "
          f"({'longe = FALSO' if b >= 50 else 'perto = FALSO'})")
    for s in range(0, 10):
        zl = list(y); random.Random(s).shuffle(zl)
        p = GaussianNB().fit(Xtr, np.array(zl)).predict(Xte)
        corr = np.corrcoef(p, dist)[0, 1]
        print(f"   semente {s}: NB embaralhado {100 * bal(yte, p):6.2f}% | correlação predição × distância {corr:+.2f}")
