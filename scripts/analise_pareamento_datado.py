# -*- coding: utf-8 -*-
"""Análise do pareamento datado × não datado (medidas T1–T5).
Pré-registro: `relatorios/PREREGISTRO_PAREAMENTO_DATADO.md`. Sorteio e coleta:
`scripts/pareamento_datado.py`.

    python scripts/analise_pareamento_datado.py --fumaca   # confere o código com o que já foi coletado; sem métricas
    python scripts/analise_pareamento_datado.py            # a rodada pré-registrada, depois da coleta

Montagem da variante N (regra fixada no `pareamento_datado.py`): por tema, as
primeiras URLs do ranking, na ordem, que estejam no corpus ou na coleta nova,
até o pedido do tema. As verdadeiras novas passam pela MESMA preparação das
outras (lide corrigido, legenda de foto removida, título + lide,
`normalizar_fonte`) e pelos MESMOS filtros de entrada do pareado (degenerada,
narração do checador, cabeçalho do checador). A que cai num filtro é trocada
pela próxima do ranking, e o número de trocas é relatado. Uma URL reaproveitada
do corpus só vale se for elegível para o pareado (canônica, não repetida, sem
os mesmos filtros).
"""
import os
import sys
import json
from collections import Counter

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coletar_verdadeiras as cv                                                  # noqa: E402
from pareamento_datado import (ARQ_AMOSTRA, ARQ_COLETA, ARQ_CORPUS, ANOS, SEED, dividir,   # noqa: E402
                               url_verdadeira)
from preparar_verdadeiras import corrigir_lides, tirar_legenda_da_foto           # noqa: E402
from fundir_corpus import normalizar_fonte, elegivel                              # noqa: E402
from limpeza_viral import degenerada, narrativa_checador, cabecalho_checador     # noqa: E402
from comparativo_bracos import (rep_tfidf, treinar, avaliar, bal_bootstrap, indices_bootstrap,   # noqa: E402
                                ic, mcnemar, pct)

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_BRACO = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_RESULTADO = os.path.join(_RAIZ, "relatorios", "pareamento_datado", "resultado.json")
MARCAS_EPOCA = set("""dilma temer bolsonaro lula haddad aécio cunha impeachment lava jato copa olimpíada
olimpíadas zika covid coronavírus pandemia quarentena cloroquina vacina vacinas eleição eleições
2014 2015 2016 2017 2018 2019 2020""".split())
PREVISOES = {"T1_D": (48, 55), "T1_N": (60, 72), "T2": (2, 8), "T3": (None, -5), "T4": (-3, 3), "T5": (None, -2)}


def ano(r):
    d = (r.get("data_publicacao") or "")[:4]
    return int(d) if d.isdigit() else None


def preparar_nova(r):
    r = dict(r)
    corrigir_lides([r])
    tirar_legenda_da_foto([r])
    texto, _ = normalizar_fonte(f"{r['titulo'].strip()} {r['lide'].strip()}")
    return {"id": "N-" + cv.chave(r["url_origem_real"]), "veredito": "VERDADEIRO", "texto_entrada": texto,
            "data_publicacao": r["data_publicacao"], "tema_pareamento": r["tema_pareamento"],
            "url": r["url_origem_real"], "origem_n": "coleta_nova"}


def passa_filtros(texto):
    return not (degenerada(texto) or narrativa_checador(texto) or cabecalho_checador(texto))


def montar_n(amostra, por_url, coleta, fumaca):
    escolhidas, trocas, faltas = [], Counter(), {}
    for tema, ranking in amostra["N_rankings"].items():
        pedido, obtidas = amostra["N_pedido_por_tema"][tema], 0
        for url in ranking:
            if obtidas >= pedido:
                break
            k = cv.chave(url)
            if k in por_url:
                r = por_url[k]
                if not elegivel(r) or not r.get("duplicata_canonica", True):
                    trocas["corpus_inelegivel"] += 1
                    continue
                reg = {"id": r["id"], "veredito": "VERDADEIRO", "texto_entrada": r["texto_entrada"],
                       "data_publicacao": r["data_publicacao"], "tema_pareamento": tema, "url": url,
                       "origem_n": "corpus"}
            elif k in coleta:
                reg = preparar_nova(coleta[k])
                if not passa_filtros(reg["texto_entrada"]):
                    trocas["coleta_filtrada"] += 1
                    continue
            else:
                if fumaca:
                    continue
                trocas["nao_coletada"] += 1
                continue
            escolhidas.append(reg)
            obtidas += 1
        if obtidas < pedido:
            faltas[tema] = pedido - obtidas
    return escolhidas, trocas, faltas


def rep_ano():
    def ajustar(treino):
        def transformar(regs):
            linhas = [ANOS.index(ano(r)) if ano(r) in ANOS else None for r in regs]
            dados = [(i, j) for i, j in enumerate(linhas) if j is not None]
            return csr_matrix(([1.0] * len(dados), ([i for i, _ in dados], [j for _, j in dados])),
                              shape=(len(regs), len(ANOS)))
        return transformar
    return ajustar


def partes(falsas, parte_f, verdadeiras, parte_v):
    return {p: [r for r in falsas if parte_f[r["id"]] == p] + [r for r in verdadeiras if parte_v[r["id"]] == p]
            for p in ("treino", "validacao", "teste")}


def resumo(av, rng):
    return {"n": int(len(av["y"])), "bal": pct(av["bal"]), "ic95": ic(bal_bootstrap(av, indices_bootstrap(len(av["y"]), rng))),
            "rev_falso": pct(av["rev_falso"]), "rev_verdadeiro": pct(av["rev_verdadeiro"])}


def dif_pareada(a, b, rng):
    idx = indices_bootstrap(len(a["y"]), rng)
    d = bal_bootstrap(a, idx) - bal_bootstrap(b, idx)
    return {"pp": pct(a["bal"] - b["bal"]), "ic95": ic(d)}


def dif_independente(a, b, rng):
    d = bal_bootstrap(a, indices_bootstrap(len(a["y"]), rng)) - bal_bootstrap(b, indices_bootstrap(len(b["y"]), rng))
    return {"pp": pct(a["bal"] - b["bal"]), "ic95": ic(d)}


def pesos_epoca(t):
    nomes = t["transformar"].vets[0].get_feature_names_out()
    coef = t["modelo"].coef_[0]
    saida = {}
    for classe, ordem in (("FALSO", np.argsort(-coef)[:30]), ("VERDADEIRO", np.argsort(coef)[:30])):
        termos = [str(nomes[i]) for i in ordem]
        saida[classe] = {"termos": termos, "marcas_de_epoca": sum(any(p in MARCAS_EPOCA for p in t_.split())
                                                                 for t_ in termos)}
    return saida


def dentro(v, faixa):
    lo, hi = faixa
    return (lo is None or v >= lo) and (hi is None or v <= hi)


def main():
    fumaca = "--fumaca" in sys.argv
    amostra = json.load(open(ARQ_AMOSTRA, encoding="utf-8"))
    corpus = {}
    por_url = {}
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        r = json.loads(l)
        corpus[r["id"]] = r
        if r["veredito"] == "VERDADEIRO":
            por_url.setdefault(cv.chave(url_verdadeira(r)), r)
    coleta = {cv.chave(r["url_origem_real"]): r for r in cv.ler_jsonl(ARQ_COLETA)}

    falsas = [corpus[i] for i in amostra["falsas"]]
    verd_d = [corpus[i] for i in amostra["D_verdadeiras"]]
    verd_n, trocas, faltas = montar_n(amostra, por_url, coleta, fumaca)
    parte_n = dividir([r["id"] for r in verd_n], f"{SEED}|N")
    print(f"falsas {len(falsas)} | D {len(verd_d)} | N {len(verd_n)} de {sum(amostra['N_pedido_por_tema'].values())}")
    print(f"N: origem {dict(Counter(r['origem_n'] for r in verd_n))} | trocas {dict(trocas)} | faltas {faltas}")
    if faltas and not fumaca:
        print("[!] N incompleta: a coleta não terminou ou o ranking esgotou; relatar no adendo")

    D = partes(falsas, amostra["falsas"], verd_d, amostra["D_verdadeiras"])
    N = partes(falsas, amostra["falsas"], verd_n, parte_n)
    url_de = lambda r: cv.chave(url_verdadeira(r))                     # noqa: E731
    treino_url = {v: {url_de(r) for r in X["treino"] + X["validacao"] if r["veredito"] == "VERDADEIRO"}
                  for v, X in (("D", D), ("N", N))}
    D_teste_p_N = [r for r in D["teste"] if r["veredito"] == "FALSO" or url_de(r) not in treino_url["N"]]
    N_teste_p_D = [r for r in N["teste"] if r["veredito"] == "FALSO" or url_de(r) not in treino_url["D"]]
    fakebr = [json.loads(l) for l in open(ARQ_BRACO, encoding="utf-8") if '"split": "fakebr"' in l]
    print(f"teste D {len(D['teste'])} (sem treino de N: {len(D_teste_p_N)}) | teste N {len(N['teste'])} "
          f"(sem treino de D: {len(N_teste_p_D)}) | Fake.br {len(fakebr)}")
    print(f"anos das verdadeiras: D {dict(sorted(Counter(ano(r) for r in verd_d).items()))} | "
          f"N {dict(sorted(Counter(ano(r) for r in verd_n).items()))}")

    ano_d, ano_n = treinar(rep_ano(), D), treinar(rep_ano(), N)
    tf_d, tf_n = treinar(rep_tfidf("texto_entrada"), D), treinar(rep_tfidf("texto_entrada"), N)
    av = {"ano_D": avaliar(ano_d, D["teste"]), "ano_N": avaliar(ano_n, N["teste"]),
          "D_D": avaliar(tf_d, D["teste"]), "N_N": avaliar(tf_n, N["teste"]),
          "N_Dteste": avaliar(tf_n, D_teste_p_N), "D_Dteste_mesmos": avaliar(tf_d, D_teste_p_N),
          "D_Nteste": avaliar(tf_d, N_teste_p_D),
          "D_fakebr": avaliar(tf_d, fakebr), "N_fakebr": avaliar(tf_n, fakebr)}
    if fumaca:
        print("FUMAÇA: todos os modelos treinaram e todas as avaliações rodaram; nenhuma métrica mostrada")
        return

    rng = np.random.default_rng(SEED)
    res = {"n": {"falsas": len(falsas), "D": len(verd_d), "N": len(verd_n), "N_trocas": dict(trocas),
                 "N_faltas": faltas, "N_origem": dict(Counter(r["origem_n"] for r in verd_n)),
                 "teste_D_sem_treino_N": len(D_teste_p_N), "teste_N_sem_treino_D": len(N_teste_p_D)},
           "C": {"ano_D": ano_d["C"], "ano_N": ano_n["C"], "tfidf_D": tf_d["C"], "tfidf_N": tf_n["C"]},
           "celulas": {k: resumo(v, rng) for k, v in av.items()}}
    res["T1"] = {"D": res["celulas"]["ano_D"]["bal"], "N": res["celulas"]["ano_N"]["bal"]}
    res["T2"] = dif_independente(av["N_N"], av["D_D"], rng)
    res["T3"] = dif_pareada(av["N_Dteste"], av["D_Dteste_mesmos"], rng)
    res["T4"] = dif_independente(av["D_Nteste"], av["D_D"], rng)
    res["T5"] = dif_pareada(av["N_fakebr"], av["D_fakebr"], rng)
    b, c, p = mcnemar(av["N_fakebr"]["acerto"], av["D_fakebr"]["acerto"])
    res["T5"]["mcnemar"] = {"so_N_acerta": b, "so_D_acerta": c, "p": p}
    b, c, p = mcnemar(av["N_Dteste"]["acerto"], av["D_Dteste_mesmos"]["acerto"])
    res["T3"]["mcnemar"] = {"so_N_acerta": b, "so_D_acerta": c, "p": p}
    res["pesos"] = {"D": pesos_epoca(tf_d), "N": pesos_epoca(tf_n)}

    t1n, t3 = res["T1"]["N"], res["T3"]["pp"]
    if t1n >= 60:
        res["leitura"] = ("o pareamento datado é necessário" if t3 <= -5 else "atalho disponível, pouco usado")
    elif t1n < 55:
        res["leitura"] = "a datação não fez diferença aqui"
    else:
        res["leitura"] = "inconclusivo"
    res["confronto_previsoes"] = {
        "T1_D": [res["T1"]["D"], dentro(res["T1"]["D"], PREVISOES["T1_D"])],
        "T1_N": [t1n, dentro(t1n, PREVISOES["T1_N"])],
        "T2": [res["T2"]["pp"], dentro(res["T2"]["pp"], PREVISOES["T2"])],
        "T3": [t3, dentro(t3, PREVISOES["T3"])],
        "T4": [res["T4"]["pp"], dentro(res["T4"]["pp"], PREVISOES["T4"])],
        "T5": [res["T5"]["pp"], dentro(res["T5"]["pp"], PREVISOES["T5"])]}

    for k, v in res["celulas"].items():
        print(f"  {k:18s} {v['bal']:6.2f}% {v['ic95']} | rev. FALSO {v['rev_falso']:.1f}% | "
              f"rev. VERDADEIRO {v['rev_verdadeiro']:.1f}% | n {v['n']}")
    for t in ("T2", "T3", "T4", "T5"):
        print(f"  {t}: {res[t]}")
    for v in ("D", "N"):
        print(f"  pesos {v}: marcas de época no top-30 FALSO {res['pesos'][v]['FALSO']['marcas_de_epoca']}, "
              f"VERDADEIRO {res['pesos'][v]['VERDADEIRO']['marcas_de_epoca']}")
    print(f"leitura (§3): {res['leitura']}")
    print(f"confronto (§2): {res['confronto_previsoes']}")
    with open(ARQ_RESULTADO, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"salvo em {ARQ_RESULTADO}")


if __name__ == "__main__":
    main()
