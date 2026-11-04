# -*- coding: utf-8 -*-
"""Protótipo de recuperação de checagens (RQ3, uso B).
Pré-registro: `relatorios/PREREGISTRO_RECUPERACAO.md`.

    python scripts/recuperacao.py --sortear     # sorteia as 50 consultas e fixa o acervo (feito em 13/10/2026)
    python scripts/recuperacao.py --buscar      # roda S0–S4 e monta a planilha às cegas
    python scripts/recuperacao.py --apurar      # depois da anotação: sucesso@3, precisão@1, cobertura, utilidade

Sistemas (§1): S1 TF-IDF de `texto_entrada`; S2 TF-IDF de `ideia_central`;
S3 bge-m3 de `texto_entrada`; S4 bge-m3 de `ideia_central`; S0 3 falsas do
acervo sorteadas do mesmo tema da consulta. As 3 primeiras de cada um; a
planilha mostra a união, sem repetição, em ordem embaralhada, SEM o sistema e
SEM a posição. Quem é quem fica em `chave_planilha.json`, que o anotador não
abre.
"""
import os
import sys
import json
import math
import random
import hashlib
from collections import Counter

import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_VETORES = os.path.join(_RAIZ, "datasets", "vetores", "vetores_bge-m3.npz")
PASTA = os.path.join(_RAIZ, "relatorios", "recuperacao")
ARQ_CONSULTAS = os.path.join(PASTA, "consultas.json")
ARQ_CANDIDATOS = os.path.join(PASTA, "candidatos.json")
ARQ_CHAVE = os.path.join(PASTA, "chave_planilha.json")
ARQ_PLANILHA = os.path.join(PASTA, "planilha_recuperacao.xlsx")
ARQ_RESULTADO = os.path.join(PASTA, "resultado_recuperacao.json")
N_CONSULTAS = 50
K = 3
SEED = 0
SISTEMAS = ("S0", "S1", "S2", "S3", "S4")
J1 = ["mesma_alegacao", "relacionada", "nao_relacionada"]
J2 = ["sim", "em_parte", "nao"]


def falsas_do_braco():
    regs = [json.loads(l) for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()]
    return [r for r in regs if r["veredito"] == "FALSO" and r["origem"] == "corpus"]


def sortear():
    if os.path.exists(ARQ_CONSULTAS):
        sys.exit(f"{ARQ_CONSULTAS} já existe: o sorteio é feito uma vez só.")
    falsas = falsas_do_braco()
    teste = sorted((r["id"] for r in falsas if r["split"] == "teste"))
    acervo = sorted((r["id"] for r in falsas if r["split"] in ("treino", "validacao")))
    consultas = random.Random(SEED).sample(teste, N_CONSULTAS)
    os.makedirs(PASTA, exist_ok=True)
    with open(ARQ_CONSULTAS, "w", encoding="utf-8") as f:
        json.dump({"semente": SEED, "n_teste_disponivel": len(teste), "consultas": consultas,
                   "n_acervo": len(acervo),
                   "acervo_sha": hashlib.sha256("\n".join(acervo).encode()).hexdigest()[:16]},
                  f, indent=1, ensure_ascii=False)
    print(f"consultas: {len(consultas)} de {len(teste)} falsas do teste | acervo: {len(acervo)} falsas")


# ==========================================================================
def top_k(matriz_consultas, matriz_acervo, k=K):
    sim = matriz_consultas @ matriz_acervo.T
    sim = sim.toarray() if hasattr(sim, "toarray") else np.asarray(sim)
    return [list(np.argsort(-linha, kind="stable")[:k]) for linha in sim]


def buscar():
    if os.path.exists(ARQ_PLANILHA) and "--sobrescrever" not in sys.argv:
        sys.exit(f"[!] {ARQ_PLANILHA} já existe -- pode ter anotações.")
    from sklearn.feature_extraction.text import TfidfVectorizer
    cfg = json.load(open(ARQ_CONSULTAS, encoding="utf-8"))
    falsas = {r["id"]: r for r in falsas_do_braco()}
    acervo = sorted(i for i, r in falsas.items() if r["split"] in ("treino", "validacao"))
    sha = hashlib.sha256("\n".join(acervo).encode()).hexdigest()[:16]
    if sha != cfg["acervo_sha"]:
        sys.exit(f"acervo mudou desde o sorteio ({sha} != {cfg['acervo_sha']})")
    consultas = cfg["consultas"]
    meta = {}
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in falsas:
            meta[r["id"]] = {"tema": r["tema_pareamento"], "url": r.get("url") or "",
                             "justificativa": r.get("justificativa") or ""}

    resultados = {q: {} for q in consultas}
    # S1, S2: TF-IDF ajustado no acervo
    for nome, campo in (("S1", "texto_entrada"), ("S2", "ideia_central")):
        vet = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit([falsas[i][campo] for i in acervo])
        tops = top_k(vet.transform([falsas[q][campo] for q in consultas]), vet.transform([falsas[i][campo] for i in acervo]))
        for q, t in zip(consultas, tops):
            resultados[q][nome] = [acervo[j] for j in t]
    # S3, S4: bge-m3, cosseno
    z = np.load(ARQ_VETORES)
    pos = {i: k for k, i in enumerate(z["ids"])}
    for nome, campo in (("S3", "texto_entrada"), ("S4", "ideia_central")):
        M = z[campo] / np.linalg.norm(z[campo], axis=1, keepdims=True)
        tops = top_k(M[[pos[q] for q in consultas]], M[[pos[i] for i in acervo]])
        for q, t in zip(consultas, tops):
            resultados[q][nome] = [acervo[j] for j in t]
    # S0: sorteio do mesmo tema
    por_tema = {}
    for i in acervo:
        por_tema.setdefault(meta[i]["tema"], []).append(i)
    for q in consultas:
        resultados[q]["S0"] = random.Random(f"{SEED}|S0|{q}").sample(por_tema[meta[q]["tema"]], K)

    with open(ARQ_CANDIDATOS, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=1, ensure_ascii=False)

    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = Workbook()
    ins = wb.active
    ins.title = "instrucoes"
    textos_ins = [
        "Recuperação de checagens: o candidato é a MESMA alegação falsa da consulta?",
        "Cada linha traz um boato novo (consulta, de agosto de 2024 em diante) e uma checagem antiga "
        "(candidato). Os candidatos vêm de sistemas diferentes, misturados; não tente adivinhar de qual.",
        "Coluna H (julgamento 1): mesma_alegacao = o candidato afirma o MESMO fato falso, ainda que com "
        "outras palavras, outra data ou outro tom; relacionada = mesmo assunto, alegação diferente; "
        "nao_relacionada = outro assunto.",
        "Coluna I (julgamento 2), SÓ se H = mesma_alegacao: a justificativa do candidato (coluna F) desmente "
        "a consulta? sim / em_parte / nao.",
        "Preencha H antes de ler a justificativa. Os links (C e E) abrem as checagens, se precisar.",
        "Segundo anotador: preencha K e L SEM olhar H, I e J (oculte-as antes de começar).",
    ]
    for n, t in enumerate(textos_ins, start=1):
        ins[f"A{n}"] = t
        ins[f"A{n}"].alignment = Alignment(wrap_text=True, vertical="top")
        ins.row_dimensions[n].height = 45
    ins["A1"].font = Font(bold=True, size=13)
    ins.column_dimensions["A"].width = 120

    ws = wb.create_sheet("anotacao")
    ws.append(["n", "consulta (boato novo)", "link da consulta", "candidato (checagem antiga)", "link do candidato",
               "justificativa do candidato", "consulta_n", "julgamento_1", "julgamento_2", "observacao_1",
               "julgamento_1_anotador_2", "julgamento_2_anotador_2", "observacao_2"])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    chave, n = {}, 0
    for qn, q in enumerate(consultas, start=1):
        cands = []
        for s in SISTEMAS:
            for posicao, c in enumerate(resultados[q][s], start=1):
                if c not in cands:
                    cands.append(c)
        random.Random(f"{SEED}|ordem|{q}").shuffle(cands)
        for c in cands:
            n += 1
            ws.append([n, falsas[q]["texto_entrada"], meta[q]["url"], falsas[c]["texto_entrada"], meta[c]["url"],
                       meta[c]["justificativa"], qn, None, None, None, None, None, None])
            chave[n] = {"consulta": q, "candidato": c,
                        "sistemas": {s: resultados[q][s].index(c) + 1 for s in SISTEMAS if c in resultados[q][s]}}
    for linha in ws.iter_rows(min_row=2):
        for c in linha:
            c.alignment = Alignment(wrap_text=c.column in (2, 4, 6), vertical="top")
    for col, larg in zip("ABCDEFGHIJKLM", (5, 55, 20, 55, 20, 55, 8, 18, 12, 25, 18, 12, 25)):
        ws.column_dimensions[col].width = larg
    ws.freeze_panes = "B2"
    for col, opcoes in (("H", J1), ("I", J2), ("K", J1), ("L", J2)):
        dv = DataValidation(type="list", formula1='"' + ",".join(opcoes) + '"', allow_blank=True,
                            showErrorMessage=True, error="Use uma das opções da lista: " + ", ".join(opcoes))
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n + 1}")
    wb.save(ARQ_PLANILHA)
    with open(ARQ_CHAVE, "w", encoding="utf-8") as f:
        json.dump(chave, f, indent=1, ensure_ascii=False)
    por_consulta = Counter(v["consulta"] for v in chave.values())
    print(f"planilha: {n} pares para julgar ({min(por_consulta.values())} a {max(por_consulta.values())} por consulta) "
          f"-> {ARQ_PLANILHA}")
    print(f"chave (não abrir antes de anotar): {ARQ_CHAVE}")


# ==========================================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (c - m), 1), round(100 * (c + m), 1)]


def apurar():
    from openpyxl import load_workbook
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from comparativo_bracos import mcnemar, holm
    chave = {int(k): v for k, v in json.load(open(ARQ_CHAVE, encoding="utf-8")).items()}
    cfg = json.load(open(ARQ_CONSULTAS, encoding="utf-8"))
    ws = load_workbook(ARQ_PLANILHA)["anotacao"]
    cab = [c.value for c in ws[1]]
    linhas = [dict(zip(cab, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
    linhas = [l for l in linhas if l.get("n") not in (None, "")]
    usar = "consenso_j1" if "consenso_j1" in cab and any(l.get("consenso_j1") for l in linhas) else "julgamento_1"
    usar2 = "consenso_j2" if usar == "consenso_j1" else "julgamento_2"
    vazias = [l["n"] for l in linhas if l.get(usar) not in J1]
    if vazias:
        sys.exit(f"{len(vazias)} linhas sem julgamento 1 válido (ex.: {vazias[:10]})")
    julg = {int(l["n"]): (l[usar], l.get(usar2)) for l in linhas}
    res = {"coluna_usada": usar, "n_pares": len(julg), "por_sistema": {}}
    acerto = {s: [] for s in SISTEMAS}
    for q in cfg["consultas"]:
        pares = [(n, v) for n, v in chave.items() if v["consulta"] == q]
        for s in SISTEMAS:
            ranks = sorted((v["sistemas"][s], julg[n][0]) for n, v in pares if s in v["sistemas"])
            acerto[s].append(any(j == "mesma_alegacao" for _, j in ranks))
            res.setdefault("_p1", {}).setdefault(s, []).append(bool(ranks) and ranks[0][1] == "mesma_alegacao")
    cobertura = [any(julg[n][0] == "mesma_alegacao" for n, v in chave.items() if v["consulta"] == q)
                 for q in cfg["consultas"]]
    for s in SISTEMAS:
        k = sum(acerto[s])
        p1 = sum(res["_p1"][s])
        res["por_sistema"][s] = {"sucesso_3": round(100 * k / N_CONSULTAS, 1), "sucesso_3_ic95": wilson(k, N_CONSULTAS),
                                 "precisao_1": round(100 * p1 / N_CONSULTAS, 1), "precisao_1_ic95": wilson(p1, N_CONSULTAS)}
    del res["_p1"]
    res["cobertura"] = {"pct": round(100 * sum(cobertura) / N_CONSULTAS, 1), "ic95": wilson(sum(cobertura), N_CONSULTAS)}
    mesmas = [j2 for j1, j2 in julg.values() if j1 == "mesma_alegacao"]
    res["utilidade_justificativa"] = {"n": len(mesmas), "contagem": dict(Counter(mesmas)),
                                      "pct_sim": round(100 * mesmas.count("sim") / len(mesmas), 1) if mesmas else None}
    comps = (("S2", "S1"), ("S4", "S3"), ("S3", "S1"))
    brutos = []
    for a, b in comps:
        so_a, so_b, p = mcnemar(np.array(acerto[a]), np.array(acerto[b]))
        brutos.append(p)
        res.setdefault("comparacoes", {})[f"{a}-{b}"] = {
            "dif_pp": round(res["por_sistema"][a]["sucesso_3"] - res["por_sistema"][b]["sucesso_3"], 1),
            "so_" + a: so_a, "so_" + b: so_b, "p": p}
    for (a, b), pc in zip(comps, holm(brutos)):
        res["comparacoes"][f"{a}-{b}"]["p_holm"] = pc
    melhor = max(res["por_sistema"][s]["sucesso_3"] for s in ("S1", "S2", "S3", "S4"))
    res["leitura"] = ("memória de checagens" if melhor >= 30 else "útil em parte" if melhor >= 10 else
                      "os boatos pouco se repetem ou a busca falha (ver cobertura)")
    if "julgamento_1_anotador_2" in cab and all(l.get("julgamento_1_anotador_2") in J1 for l in linhas):
        from sklearn.metrics import cohen_kappa_score
        a1 = [l["julgamento_1"] for l in linhas]
        a2 = [l["julgamento_1_anotador_2"] for l in linhas]
        res["concordancia"] = {"kappa_3_categorias": round(cohen_kappa_score(a1, a2), 3),
                               "kappa_mesma_sim_nao": round(cohen_kappa_score([x == "mesma_alegacao" for x in a1],
                                                                              [x == "mesma_alegacao" for x in a2]), 3),
                               "divergencias": sum(x != y for x, y in zip(a1, a2))}
    with open(ARQ_RESULTADO, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps(res, indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    if "--sortear" in sys.argv:
        sortear()
    elif "--buscar" in sys.argv:
        buscar()
    elif "--apurar" in sys.argv:
        apurar()
    else:
        print(__doc__)
