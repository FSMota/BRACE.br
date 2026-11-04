# -*- coding: utf-8 -*-
"""RQ3, uso A -- alinhamento de racionais: as palavras que decidiram a
classificação coincidem com as que o checador usou para justificar o veredito?
Pré-registro: `relatorios/PREREGISTRO_RQ3_ALINHAMENTO.md`.

    python scripts/juntar_refinamento.py                    # já rodado (braco_refinado.jsonl)
    python scripts/alinhamento_racionais.py --fumaca        # testa o código na VALIDAÇÃO; não mostra métricas
    python scripts/alinhamento_racionais.py                 # a rodada pré-registrada, no TESTE

Ideia: com regressão logística, a contribuição de cada palavra para a decisão
de um texto é exata (peso × TF-IDF) -- não precisa de LIME nem de SHAP. Para
cada falsa do teste que o modelo acertou, pegam-se as k palavras de maior
contribuição para FALSO e mede-se quantas aparecem na justificativa do checador.

O CONTROLE é o que dá sentido ao número: a justificativa fala do mesmo assunto
que o texto, então palavras sorteadas do próprio texto também aparecem nela.
O controle é a fração das palavras do texto que aparece na justificativa -- a
esperança exata de sortear k palavras do texto ao acaso. A medida principal é
a DIFERENÇA modelo − controle, pareada por item.

Palavras: minúsculas, só letras (\\w+ sem dígitos), 3+ caracteres, fora da
lista de palavras vazias abaixo. Bigramas do modelo contribuem com suas duas
palavras. Comparação exata; sensibilidade por radical (5 primeiras letras).
"""
import os
import re
import sys
import json

import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comparativo_bracos import rep_tfidf, treinar          # noqa: E402

ARQ_BRACO = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_RESULTADO = os.path.join(_RAIZ, "relatorios", "resultado_rq3_alinhamento.json")
KS = (5, 10)
N_BOOT, SEED = 2000, 0
BRACOS = {"bruto": "texto_entrada", "refinado": "ideia_central"}
REFERENCIAS = ("justificativa", "justificativa_bruta")    # principal, sensibilidade

VAZIAS = set("""
a à ao aos as às até com como contra da das de dela dele deles depois do dos e é ela elas ele eles em entre era eram
essa essas esse esses esta está estão estas este estes eu foi foram há isso isto já la lá lhe lhes mais mas me mesmo
meu minha muito na nas não nem no nos nós nossa nosso num numa o os ou para pela pelas pelo pelos por porque qual
quando que quem se sem ser será seu seus sua suas só também te tem têm ter teu tua tu um uma umas uns você vocês vai
vão são sobre seja sendo sido tinha tudo todo todos toda todas ainda assim aqui ali onde cada outro outra outros outras
pode podem poderia deve devem diz disse ser estar fazer feito fez faz tão bem bom boa vez vezes dia dias ano anos
""".split())


def palavras(texto):
    return [p for p in re.findall(r"[^\W\d_]+", (texto or "").lower()) if len(p) >= 3 and p not in VAZIAS]


def casa(conj_texto, conj_ref, radical=False):
    if not radical:
        return conj_texto & conj_ref
    refs = {r[:5] for r in conj_ref}
    return {p for p in conj_texto if p[:5] in refs}


def racional_modelo(treinado, reg, campo, k):
    """As k primeiras palavras distintas, por contribuição para FALSO (peso × TF-IDF > 0).

    Desempenho (09/10/2026, depois do pré-registro, sem mudança de lógica): o
    vocabulário e as listas por texto ficam em cache. A lista de um texto é
    calculada uma vez com o maior k; as k primeiras palavras de qualquer k menor
    são, por construção, o começo dela (mesma ordem, mesmo critério de parada)."""
    cache = treinado.setdefault("_cache_racional", {})
    chave = reg["id"]
    if chave not in cache:
        cache[chave] = _lista_racional(treinado, reg, max(max(KS), k))
    return cache[chave][1][:k]


def _lista_racional(treinado, reg, kmax):
    """Devolve (esgotou, palavras): esgotou = acabaram as contribuições positivas antes de kmax."""
    if "_nomes" not in treinado:
        treinado["_nomes"] = treinado["transformar"].vets[0].get_feature_names_out()
    nomes = treinado["_nomes"]
    coef = treinado["modelo"].coef_[0]
    linha = treinado["transformar"]([reg])
    contrib = linha.data * coef[linha.indices]
    ordem = np.argsort(-contrib)
    vistas, esgotou = [], True
    for j in ordem:
        if contrib[j] <= 0:
            break
        for p in palavras(nomes[linha.indices[j]]):
            if p not in vistas:
                vistas.append(p)
        if len(vistas) >= kmax:
            esgotou = False
            break
    return esgotou, vistas[:kmax]


def ic(v, rng):
    v = np.asarray(v)
    meds = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(N_BOOT)]
    return [round(100 * float(np.percentile(meds, 2.5)), 2), round(100 * float(np.percentile(meds, 97.5)), 2)]


def main():
    fumaca = "--fumaca" in sys.argv
    alvo = "validacao" if fumaca else "teste"
    regs = [json.loads(l) for l in open(ARQ_BRACO, encoding="utf-8") if l.strip()]
    just = {}
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        r = json.loads(l)
        if r["veredito"] == "FALSO" and r["subconjunto_pareado"]:
            just[r["id"]] = {k: r.get(k) or "" for k in REFERENCIAS}
    por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste")}
    rng = np.random.default_rng(SEED)
    res = {"conjunto": alvo, "ks": KS, "por_braco": {}}
    print(f"conjunto avaliado: {alvo}{'  (FUMAÇA: só confere o código, sem métricas)' if fumaca else ''}")

    for braco, campo in BRACOS.items():
        t = treinar(rep_tfidf(campo), por)
        falsas = [r for r in por[alvo] if r["veredito"] == "FALSO" and just.get(r["id"], {}).get("justificativa")]
        pred = t["modelo"].predict(t["transformar"](falsas))
        acertos = [r for r, p in zip(falsas, pred) if p == 1]
        print(f"\n{braco}: C={t['C']} | falsas com justificativa {len(falsas)} | acertadas {len(acertos)}")
        if fumaca:
            r = acertos[0]
            print(f"  exemplo {r['id']}: top-5 do modelo = {racional_modelo(t, r, campo, 5)}")
            print(f"  palavras da justificativa (10 primeiras) = {palavras(just[r['id']]['justificativa'])[:10]}")
            continue
        bloco = {"C": t["C"], "n_falsas": len(falsas), "n_acertadas": len(acertos)}
        for conj_nome, conj in (("acertadas", acertos), ("todas_as_falsas", falsas)):
            for ref in REFERENCIAS:
                for radical in (False, True):
                    for k in KS:
                        mod, ctl, n_usados = [], [], 0
                        for r in conj:
                            pt = set(palavras(r[campo]))
                            if len(pt) < k:
                                continue          # texto curto demais para k palavras
                            pr = set(palavras(just[r["id"]][ref]))
                            top = set(racional_modelo(t, r, campo, k))
                            if not top:
                                continue
                            mod.append(len(casa(top, pr, radical)) / len(top))
                            ctl.append(len(casa(pt, pr, radical)) / len(pt))
                            n_usados += 1
                        if not mod:
                            continue
                        dif = np.array(mod) - np.array(ctl)
                        chave = f"{conj_nome}|{ref}|{'radical' if radical else 'exata'}|k{k}"
                        bloco[chave] = {"n": n_usados,
                                        "modelo_pct": round(100 * float(np.mean(mod)), 2), "modelo_ic95": ic(mod, rng),
                                        "controle_pct": round(100 * float(np.mean(ctl)), 2), "controle_ic95": ic(ctl, rng),
                                        "dif_pp": round(100 * float(dif.mean()), 2), "dif_ic95": ic(dif, rng)}
                        if conj_nome == "acertadas" and not radical:
                            b = bloco[chave]
                            print(f"  {ref:20s} k={k:2d} n={b['n']:5d} | modelo {b['modelo_pct']:5.1f}% | controle "
                                  f"{b['controle_pct']:5.1f}% | modelo − controle {b['dif_pp']:+5.1f} pp {b['dif_ic95']}")
        # Descritivo: palavras que mais aparecem no top-5 do modelo.
        from collections import Counter
        freq = Counter(p for r in acertos for p in racional_modelo(t, r, campo, 5))
        bloco["palavras_mais_frequentes_top5"] = freq.most_common(30)
        print("  mais frequentes no top-5:", ", ".join(p for p, _ in freq.most_common(20)))
        res["por_braco"][braco] = bloco

    if not fumaca:
        with open(ARQ_RESULTADO, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        print(f"\nsalvo em {ARQ_RESULTADO}")


if __name__ == "__main__":
    main()
