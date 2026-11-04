"""Sensibilidade ao C do modelo de referência (exploratório, 25/10/2026, depois dos experimentos).

O C escolhido na validação (10) é o maior da grade {0,1; 1; 10}. Pergunta: com a
grade ampliada, a validação escolheria outro C, e isso mudaria os números do
modelo de referência (TF-IDF de palavras + LR, braço bruto) no teste interno e
no Fake.br? Mesmo vetorizador e mesma divisão do `modelo_bruto.py`. Nada aqui
substitui os números do TCC; é a resposta à pergunta "e se a grade fosse maior?".
Rodar da raiz do repositório. Saída: relatorios/diagnostico_sensibilidade_c.json
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

VALORES_C = (0.1, 1, 10, 30, 100, 1000)

regs = [json.loads(l) for l in open("datasets/braco_refinado.jsonl", encoding="utf-8") if l.strip()]
fakebr = [r for r in regs if r["origem"] == "fakebr"]
corpus = [r for r in regs if r["origem"] == "corpus"]
X = {s: [r["texto_entrada"] for r in corpus if r["split"] == s] for s in ("treino", "validacao", "teste")}
y = {s: [int(r["veredito"] == "FALSO") for r in corpus if r["split"] == s] for s in X}
X["fakebr"] = [r["texto_entrada"] for r in fakebr]
y["fakebr"] = [int(r["veredito"] == "FALSO") for r in fakebr]

vet = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
M = {"treino": vet.fit_transform(X["treino"])}
for s in ("validacao", "teste", "fakebr"):
    M[s] = vet.transform(X[s])

linhas = []
for C in VALORES_C:
    m = LogisticRegression(C=C, max_iter=4000).fit(M["treino"], y["treino"])
    linha = {"C": C, **{s: round(100 * balanced_accuracy_score(y[s], m.predict(M[s])), 2)
                        for s in ("validacao", "teste", "fakebr")}}
    linhas.append(linha)
    print(f"C={C:<6} validação {linha['validacao']:.2f}%  teste {linha['teste']:.2f}%  Fake.br {linha['fakebr']:.2f}%")

melhor = max(linhas, key=lambda l: l["validacao"])
saida = {"n": {s: len(v) for s, v in y.items()}, "grade_original": [0.1, 1, 10], "linhas": linhas,
         "escolhido_grade_ampliada": melhor["C"]}
with open("relatorios/diagnostico_sensibilidade_c.json", "w", encoding="utf-8") as f:
    json.dump(saida, f, indent=2, ensure_ascii=False)
print(f"escolhido com a grade ampliada: C={melhor['C']}")
