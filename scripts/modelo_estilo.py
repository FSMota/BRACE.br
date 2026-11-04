# -*- coding: utf-8 -*-
"""Teste 2 -- modelo SÓ DE ESTILO, interno e cruzado (Fake.br).

    python scripts/modelo_estilo.py

Pergunta: a transferência de ~23 pp acima do acaso no Fake.br (73,03%) é
estilo? O modelo não vê NENHUMA palavra de conteúdo -- só atributos de forma
do `texto_entrada`, os mesmos nas duas bases:

  tamanho ............ caracteres (log), palavras, tamanho médio de palavra e de frase
  caixa alta ......... % de letras maiúsculas, % de palavras TODAS em maiúsculas
  pontuação .......... ! e ? por 100 caracteres, "!!"/"??" repetidos, reticências,
                       aspas (declaração citada é marca de jornalismo)
  marcas de rede ..... emojis, hashtags, @menções, links
  números ............ % de dígitos
  interlocução ....... "você/vocês" por 100 palavras (a corrente fala COM o leitor)

Dois classificadores: regressão logística (efeito linear de cada atributo,
interpretável) e gradient boosting (captura combinações). Mesmo treino do
pré-registro: `subconjunto_pareado`, split `treino`; nada ajustado no Fake.br.

PREVISÕES -- escritas antes de rodar (commit deste arquivo):
  E1  interno, melhor dos dois ......................... 78% – 88%
  E2  Fake.br, melhor dos dois ......................... 60% – 68%
  E3  atributo de maior peso para FALSO ................ caixa alta ou "!"

LEITURAS FIXADAS:
  Fake.br >= 70%  -> o estilo explica praticamente toda a transferência (73%).
  Fake.br 58-70%  -> o estilo explica boa parte; o resto é léxico/conteúdo.
  Fake.br <= 58%  -> o estilo NÃO é o que transfere; o sinal está nas palavras.
Exploratório (não estava no pré-registro original); relatar como tal.
"""
import os
import re
import sys
import json
import math
import time

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from avaliacao_cruzada import carregar_fakebr, entrada_texto              # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
SAIDA = os.path.join(_RAIZ, "relatorios", "resultado_modelo_estilo.json")
N_BOOT, SEED = 2000, 0
T0 = time.time()

EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐⭕]")
NOMES = ["log_caracteres", "palavras", "tam_medio_palavra", "tam_medio_frase",
         "pct_maiusculas", "pct_palavras_caixa_alta", "exclamacao_100c", "interrogacao_100c",
         "pontuacao_repetida", "reticencias", "aspas", "emojis", "hashtags", "mencoes_arroba",
         "links", "pct_digitos", "voce_100p"]


def log(msg=""):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def atributos(t):
    n = max(1, len(t))
    palavras = re.findall(r"\w+", t)
    np_ = max(1, len(palavras))
    letras = [c for c in t if c.isalpha()]
    frases = [f for f in re.split(r"[.!?]+", t) if f.strip()]
    return [
        math.log(1 + len(t)),
        len(palavras),
        sum(len(p) for p in palavras) / np_,
        len(palavras) / max(1, len(frases)),
        sum(c.isupper() for c in letras) / max(1, len(letras)),
        sum(1 for p in palavras if len(p) >= 2 and p.isupper() and not p.isdigit()) / np_,
        100 * t.count("!") / n,
        100 * t.count("?") / n,
        len(re.findall(r"[!?]{2,}", t)),
        len(re.findall(r"\.\.\.|…", t)),
        len(re.findall(r"[\"“”]", t)),
        len(EMOJI.findall(t)),
        len(re.findall(r"#\w+", t)),
        len(re.findall(r"@\w+", t)),
        len(re.findall(r"https?://|www\.", t, re.I)),
        sum(c.isdigit() for c in t) / n,
        100 * len(re.findall(r"\bvoc[êe]s?\b", t, re.I)) / np_,
    ]


def ic_pares(acertos_f, acertos_v, rng):
    pares = np.stack([acertos_f, acertos_v], axis=1).astype(float)
    n = len(pares)
    medias = [pares[rng.integers(0, n, n)].mean() for _ in range(N_BOOT)]
    return float(np.percentile(medias, 2.5)), float(np.percentile(medias, 97.5))


def main():
    rng = np.random.default_rng(SEED)
    log("TESTE 2 -- modelo só de estilo (exploratório; previsões no cabeçalho do script)")
    with open(ARQ_CORPUS, encoding="utf-8") as f:
        corpus = [json.loads(l) for l in f if l.strip()]
    par = [r for r in corpus if r["subconjunto_pareado"]]
    tr = [r for r in par if r["split"] == "treino"]
    te = [r for r in par if r["split"] == "teste"]
    Xtr = np.array([atributos(r["texto_entrada"]) for r in tr])
    ytr = np.array([1 if r["veredito"] == "FALSO" else 0 for r in tr])
    Xte = np.array([atributos(r["texto_entrada"]) for r in te])
    yte = np.array([1 if r["veredito"] == "FALSO" else 0 for r in te])
    log(f"  treino {len(tr):,} | teste interno {len(te):,} | {len(NOMES)} atributos".replace(",", "."))

    log("carregando o Fake.br...")
    pares, _ = carregar_fakebr()
    tf = [entrada_texto(p["fake"]["bruto"]) for p in pares]
    tv = [entrada_texto(p["true"]["bruto"]) for p in pares]
    Xfb = np.array([atributos(t) for t in tf + tv])
    yfb = np.array([1] * len(tf) + [0] * len(tv))
    n = len(pares)

    log("")
    log("médias por classe (treino interno | Fake.br):")
    log(f"  {'atributo':26s} {'int FALSO':>10s} {'int VERD':>10s}   {'FB FALSO':>10s} {'FB VERD':>10s}")
    for j, nome in enumerate(NOMES):
        log(f"  {nome:26s} {Xtr[ytr == 1, j].mean():10.3f} {Xtr[ytr == 0, j].mean():10.3f}   "
            f"{Xfb[yfb == 1, j].mean():10.3f} {Xfb[yfb == 0, j].mean():10.3f}")

    resultados = {}
    esc = StandardScaler().fit(Xtr)
    modelos = {
        "regressao_logistica": (LogisticRegression(max_iter=2000).fit(esc.transform(Xtr), ytr), True),
        "gradient_boosting": (HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, ytr), False),
    }
    for nome, (m, escala) in modelos.items():
        prep = esc.transform if escala else (lambda X: X)
        pi, pf = m.predict(prep(Xte)), m.predict(prep(Xfb))
        bi, bf = balanced_accuracy_score(yte, pi), balanced_accuracy_score(yfb, pf)
        ic = ic_pares(pf[:n] == 1, pf[n:] == 0, rng)
        rf, rv = recall_score(yfb, pf, pos_label=1), recall_score(yfb, pf, pos_label=0)
        log("")
        log(f"{nome}:")
        log(f"  interno {bi:.2%} | Fake.br {bf:.2%}  IC95% [{ic[0]:.2%}, {ic[1]:.2%}]")
        log(f"  Fake.br revocação FALSO {rf:.2%} | VERDADEIRO {rv:.2%}")
        resultados[nome] = {"interno": round(bi, 4), "fakebr": round(bf, 4),
                            "fakebr_ic95": [round(ic[0], 4), round(ic[1], 4)],
                            "rev_falso": round(rf, 4), "rev_verdadeiro": round(rv, 4)}

    coef = modelos["regressao_logistica"][0].coef_[0]
    ordem = np.argsort(coef)[::-1]
    log("")
    log("pesos da regressão logística (atributos padronizados; + empurra para FALSO):")
    for j in ordem:
        log(f"  {NOMES[j]:26s} {coef[j]:+.3f}")
    resultados["pesos_rl"] = {NOMES[j]: round(float(coef[j]), 3) for j in ordem}

    melhor_int = max(resultados[k]["interno"] for k in modelos)
    melhor_fb = max(resultados[k]["fakebr"] for k in modelos)
    topo = NOMES[ordem[0]]
    log("")
    log("=" * 72)
    log("CONFRONTO COM AS PREVISÕES DO CABEÇALHO")
    log(f"  E1 interno       {melhor_int:.2%}   prev. 78–88%   {'✓' if 0.78 <= melhor_int <= 0.88 else '✗'}")
    log(f"  E2 Fake.br       {melhor_fb:.2%}   prev. 60–68%   {'✓' if 0.60 <= melhor_fb <= 0.68 else '✗'}")
    log(f"  E3 maior peso    {topo:26s} prev. caixa alta ou '!'   "
        f"{'✓' if topo in ('pct_maiusculas', 'pct_palavras_caixa_alta', 'exclamacao_100c') else '✗'}")
    if melhor_fb >= 0.70:
        leitura = "o estilo explica praticamente toda a transferência (73%)."
    elif melhor_fb > 0.58:
        leitura = "o estilo explica boa parte da transferência; o resto é léxico/conteúdo."
    else:
        leitura = "o estilo NÃO é o que transfere; o sinal está nas palavras."
    log(f"  LEITURA: {leitura}")
    log(f"  referência: TF-IDF de palavras -- interno 92,05% | Fake.br 73,03%")
    log("=" * 72)
    resultados["leitura"] = leitura
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    log(f"resultados salvos em relatorios/{os.path.basename(SAIDA)}")


if __name__ == "__main__":
    main()
