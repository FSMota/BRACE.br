# -*- coding: utf-8 -*-
"""Comparativo bruto × refinado × combinado -- RQ1, conforme
`relatorios/PREREGISTRO_BRACO_REFINADO.md` (commit cf8a101).

    python scripts/juntar_refinamento.py      # antes: confere e junta a saída
    python scripts/comparativo_bracos.py

    # teste com dados sintéticos (não grava em relatorios/)
    python scripts/comparativo_bracos.py --dados X.jsonl --resultado Y.json

Este script foi commitado ANTES de qualquer análise ler as frases refinadas
(§1.6). Tudo o que ele decide -- limiares, faixas, leituras -- vem do
pré-registro; onde o pré-registro deixou algo em aberto, a escolha está em
AMBIGUIDADES e vai para o adendo.

O que roda, na ordem:
  0. conjunto de avaliação: ids com saída válida, menos os "não é boato" da
     anotação manual (se apurada), os mesmos nos três braços;
  1. checagens de validade C1 (tamanho), C2 (marcas por classe), C4 (rótulos
     embaralhados e sobreposição). C3 (fidelidade) é manual:
     `auditoria_fidelidade.py`;
  2. os três braços, rodada A (completo) e B (sem `entrada_com_veredito`):
     interno e Fake.br, IC por bootstrap, revocações, categorias do Fake.br;
  3. família confirmatória com Holm: H1 (queda refinado < queda bruto), bruto ×
     refinado no Fake.br, bruto × combinado no Fake.br;
  4. leituras fixadas e confronto com as previsões da §2 e do pesquisador (§5).
"""
import os
import re
import sys
import json
import math
import time
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refinar_cego import MARCAS, PROMPT_SHA, PROMPT_VERSAO, medir      # noqa: E402

ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_ANOTACAO = os.path.join(_RAIZ, "relatorios", "anotacao", "resultado_anotacao.json")
ARQ_RESULTADO = os.path.join(_RAIZ, "relatorios", "resultado_comparativo_bracos.json")

# --- fixado no pré-registro ------------------------------------------------
VALORES_C = [0.1, 1.0, 10.0]                  # a grade do modelo_bruto.py (§1.3)
N_BOOT = 2000                                  # §1.5
SEED = 0
ALFA = 0.05
SEMENTES_EMBARALHAR = [0, 1, 2]                # as do controles_pipeline.py
FAIXA_EMBARALHADO = (0.45, 0.55)               # C4
LIMIAR_TAMANHO = 0.60                          # C1
LIMIAR_MARCA_PP = 10.0                         # C2
LIMIAR_VEREDITO_FALSAS = 2.0                   # C2, % das frases falsas do corpus
REFERENCIA_BRUTO = {"interno": 0.9119, "fakebr": 0.7351, "queda": 0.1768}   # adendo 5

# Previsões: (mínimo, máximo) em %, ou pp para quedas e diferenças. None = sem limite.
PREVISOES_SECAO2 = {"R1": (75, 85), "R2": (55, 63), "R3": (15, 25), "R4": (91, 93),
                    "R5": (72, 76), "R6": (None, 10), "C1": (58, 65)}
PREVISOES_PESQUISADOR = {"R1": (79, 90), "R2": (60, 70), "R3": (10, 15), "R4": (91, 93),
                         "R5": (70, 80), "R6": (None, 10), "C1": (50, 60)}
PESQUISADOR_C2 = "nenhuma marca vai diferir >= 10 pp"
PESQUISADOR_H1 = "confirmada"

AMBIGUIDADES = [
    "C1: 'só o número de palavras (e o de caracteres)' -- implementado como os dois "
    "atributos juntos (palavras e log de caracteres), padronizados no treino, regressão "
    "logística com C pela validação na mesma grade.",
    "C1 > 60%: 'a acurácia do refinado acima desse baseline' é relatada como diferença em pp "
    "(refinado − tamanho), interno e Fake.br; 'refinado + tamanho' = TF-IDF da frase ao lado "
    "dos dois atributos de tamanho padronizados.",
    "C2: frases 'com a marca' são tiradas de TODAS as partes e das duas classes (treino, "
    "validação, teste e Fake.br), e o braço refinado é retreinado sem elas.",
    "C4 sobreposição: frase refinada idêntica após minúsculas e espaços normalizados, "
    "presente em mais de uma parte do pareado. 'Acurácia sem elas' = o mesmo modelo, "
    "avaliado só nas frases do teste que não aparecem no treino nem na validação.",
    "C4 embaralhado: usa o C que o braço escolheu com os rótulos reais.",
    "H1: p-valor por bootstrap bicaudal, 2·min(P(dif ≤ 0), P(dif ≥ 0)), com piso 1/N_BOOT, "
    "para entrar na correção de Holm junto dos dois McNemar.",
    "H1 (a)/(b): (a) se o refinado for melhor que o bruto no Fake.br com McNemar significativo "
    "após Holm; (b) se for pior com significância; se não houver diferença significativa no "
    "Fake.br, a queda menor vem só do interno -- lido como (b) (não é 'generaliza melhor').",
    "Bootstrap: os mesmos índices reamostrados em todos os braços (os itens são os mesmos), o "
    "que torna pareadas as comparações entre braços.",
]

T0 = time.time()


def log(msg=""):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def arg(nome, padrao):
    return sys.argv[sys.argv.index(nome) + 1] if nome in sys.argv else padrao


def pct(x):
    return round(float(100 * x), 2)


# ==========================================================================
# representações
# ==========================================================================
def tfidf():
    return TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)


def rep_tfidf(*campos):
    """Um TF-IDF por campo, cada um ajustado só no treino, lado a lado (§1.3)."""
    def ajustar(treino):
        vets = [tfidf().fit([r[c] for r in treino]) for c in campos]

        def transformar(regs):
            return hstack([v.transform([r[c] for r in regs]) for v, c in zip(vets, campos)]).tocsr()
        transformar.vets, transformar.campos = vets, campos
        return transformar
    return ajustar


def atributos_tamanho(regs):
    return np.array([[len(re.findall(r"\w+", r["ideia_central"])), math.log1p(len(r["ideia_central"]))]
                     for r in regs])


def rep_tamanho():
    def ajustar(treino):
        sc = StandardScaler().fit(atributos_tamanho(treino))

        def transformar(regs):
            return csr_matrix(sc.transform(atributos_tamanho(regs)))
        return transformar
    return ajustar


def rep_juntas(*reps):
    def ajustar(treino):
        ts = [r(treino) for r in reps]

        def transformar(regs):
            return hstack([t(regs) for t in ts]).tocsr()
        return transformar
    return ajustar


# ==========================================================================
# treino, avaliação, bootstrap
# ==========================================================================
def rotulos(regs):
    return np.array([1 if r["veredito"] == "FALSO" else 0 for r in regs])


def bal(y, pred):
    return 0.5 * ((pred[y == 1] == 1).mean() + (pred[y == 0] == 0).mean())


def treinar(rep, dados, embaralhar=None, C=None):
    """Ajusta a representação no treino; C pela validação (ou fixo). Devolve o modelo pronto."""
    transformar = rep(dados["treino"])
    Xtr, ytr = transformar(dados["treino"]), rotulos(dados["treino"])
    if embaralhar is not None:
        ytr = list(ytr)
        random.Random(embaralhar).shuffle(ytr)
        ytr = np.array(ytr)
    Xva, yva = transformar(dados["validacao"]), rotulos(dados["validacao"])
    melhor = None
    for c in ([C] if C is not None else VALORES_C):
        m = LogisticRegression(C=c, max_iter=2000).fit(Xtr, ytr)
        b = bal(yva, m.predict(Xva))
        if melhor is None or b > melhor[0]:
            melhor = (b, c, m)
    return {"transformar": transformar, "modelo": melhor[2], "C": melhor[1], "validacao": melhor[0]}


def avaliar(treinado, regs):
    y = rotulos(regs)
    pred = treinado["modelo"].predict(treinado["transformar"](regs))
    return {"y": y, "pred": pred, "acerto": pred == y, "bal": bal(y, pred),
            "rev_falso": (pred[y == 1] == 1).mean(), "rev_verdadeiro": (pred[y == 0] == 0).mean()}


def indices_bootstrap(n, rng):
    return rng.integers(0, n, size=(N_BOOT, n), dtype=np.int32)


def bal_bootstrap(av, idx):
    y, a = av["y"][idx], av["acerto"][idx]
    pos = (y == 1)
    rf = (a & pos).sum(axis=1) / np.maximum(1, pos.sum(axis=1))
    rv = (a & ~pos).sum(axis=1) / np.maximum(1, (~pos).sum(axis=1))
    return 0.5 * (rf + rv)


def ic(v):
    return [pct(np.percentile(v, 2.5)), pct(np.percentile(v, 97.5))]


def mcnemar(acerto_a, acerto_b):
    b = int(np.sum(acerto_a & ~acerto_b))
    c = int(np.sum(~acerto_a & acerto_b))
    return b, c, (binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0)


def holm(pvalores):
    ordem = sorted(range(len(pvalores)), key=lambda i: pvalores[i])
    m, corr, maior = len(pvalores), [0.0] * len(pvalores), 0.0
    for pos, i in enumerate(ordem):
        maior = max(maior, min(1.0, (m - pos) * pvalores[i]))
        corr[i] = maior
    return corr


def por_categoria(av, regs):
    grupos = defaultdict(list)
    for i, r in enumerate(regs):
        grupos[r.get("categoria_fakebr") or "?"].append(i)
    saida = {}
    for cat, idx in sorted(grupos.items(), key=lambda x: -len(x[1])):
        idx = np.array(idx)
        if len(set(av["y"][idx])) == 2:
            saida[cat] = {"acuracia_balanceada": pct(bal(av["y"][idx], av["pred"][idx])), "n": len(idx)}
    return saida


# ==========================================================================
# leituras fixadas (§4)
# ==========================================================================
def leitura_interno(v):
    if v >= 88:
        return "o conteúdo sozinho separa quase como o texto bruto: forma e conteúdo são sinais redundantes"
    if v >= 75:
        return "o conteúdo separa bem, mas abaixo da forma: há diferença de assunto entre as classes dentro do tema"
    if v > 60:
        return "a separação interna era majoritariamente forma; o conteúdo contribui pouco"
    return "sem a forma, quase nada separa -- descartar antes (C3) que o refinamento apagou o conteúdo"


def leitura_fakebr(v):
    if v >= 70:
        return "o conteúdo transfere tanto quanto o texto bruto"
    if v >= 60:
        return "o conteúdo transfere de forma comparável à forma (65,18%)"
    if v > 55:
        return "o conteúdo transfere pouco, compatível com os ~8 pp de léxico além da forma"
    return "o conteúdo aprendido é específico do corpus; o que o bruto transferia era quase só forma"


def leitura_combinado(dif_pp, ic_dif):
    if dif_pp >= 2 and ic_dif[0] > 0:
        return "a frase refinada acrescenta sinal que transfere (complemento)"
    if dif_pp <= -2:
        return "a frase refinada atrapalha fora da base"
    return "a frase refinada é redundante com o texto bruto"


def dentro(v, faixa):
    lo, hi = faixa
    return (lo is None or v >= lo) and (hi is None or v <= hi)


# ==========================================================================
def main():
    arq_dados = arg("--dados", ARQ_DADOS)
    arq_res = arg("--resultado", ARQ_RESULTADO)
    rng = np.random.default_rng(SEED)
    res = {"prompt_versao": PROMPT_VERSAO, "prompt_sha": PROMPT_SHA, "ambiguidades": AMBIGUIDADES}

    # ---- 0. conjunto de avaliação -------------------------------------------
    log(f"lendo {arq_dados}")
    regs = [json.loads(l) for l in open(arq_dados, encoding="utf-8") if l.strip()]
    nao_boato = set()
    if os.path.exists(ARQ_ANOTACAO):
        nao_boato = set(json.load(open(ARQ_ANOTACAO, encoding="utf-8")).get("ids_nao_boato", []))
        log(f"anotação apurada: {len(nao_boato)} ids 'não é boato' saem dos três braços")
    else:
        log("[!] anotação não apurada: roda SEM a exclusão dos 'não é boato' (vira adendo, §1.3)")
    n_antes = len(regs)
    regs = [r for r in regs if r["id"] not in nao_boato]
    por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste", "fakebr")}
    pareado_total = 2 * 11975
    n_corpus = sum(len(por[s]) for s in ("treino", "validacao", "teste"))
    perda = 1 - n_corpus / pareado_total
    log(f"  corpus {n_corpus:,} de {pareado_total:,} (perda {100 * perda:.2f}%) | Fake.br {len(por['fakebr']):,} de 7.200"
        .replace(",", "."))
    for s, rs in por.items():
        c = Counter(r["veredito"] for r in rs)
        log(f"    {s:10s} FALSO {c['FALSO']:6d} | VERDADEIRO {c['VERDADEIRO']:6d}")
    res["conjunto"] = {"registros_lidos": n_antes, "excluidos_nao_boato": n_antes - len(regs),
                       "anotacao_apurada": bool(nao_boato) or os.path.exists(ARQ_ANOTACAO),
                       "perda_pareado_pct": pct(perda),
                       "por_split": {s: dict(Counter(r["veredito"] for r in rs)) for s, rs in por.items()}}
    teste, fakebr = por["teste"], por["fakebr"]
    idx_int, idx_fb = indices_bootstrap(len(teste), rng), indices_bootstrap(len(fakebr), rng)

    # ---- 1. checagens de validade --------------------------------------------
    log("")
    log("=" * 72)
    log("C2 -- marcas por classe na frase refinada (fila inteira avaliada)")
    log("=" * 72)
    medidas = defaultdict(list)
    for r in regs:
        m = medir(r["ideia_central"])
        m["ajustada"] = bool(r.get("saida_ajustes"))
        medidas[r["grupo"]].append(m)
    colunas = list(MARCAS) + ["mais_de_25", "mais_de_1_frase", "ajustada"]
    resumo_c2 = {}
    for g, ms in sorted(medidas.items()):
        k = len(ms)
        linha = {"n": k, "palavras_media": round(sum(m["palavras"] for m in ms) / k, 2),
                 "pct_mais_de_25": round(100 * sum(m["palavras"] > 25 for m in ms) / k, 2),
                 "pct_mais_de_1_frase": round(100 * sum(m["frases"] > 1 for m in ms) / k, 2),
                 "pct_ajustada": round(100 * sum(m["ajustada"] for m in ms) / k, 2),
                 **{f"pct_{c}": round(100 * sum(m[c] for m in ms) / k, 2) for c in MARCAS}}
        resumo_c2[g] = linha
        log(f"  {g:18s} n={k:6d} palavras {linha['palavras_media']:5.1f} | "
            + " | ".join(f"{c} {linha['pct_' + c]:5.2f}%" for c in colunas))
    marcas_alerta = set()
    for base in ("corpus", "fakebr"):
        f, v = resumo_c2.get(f"{base}_falso"), resumo_c2.get(f"{base}_verdadeiro")
        if not (f and v):
            continue
        difs = {c: round(f[f"pct_{c}"] - v[f"pct_{c}"], 2) for c in colunas}
        difs["palavras"] = round(f["palavras_media"] - v["palavras_media"], 2)
        alerta = [c for c in MARCAS if abs(difs[c]) >= LIMIAR_MARCA_PP]
        marcas_alerta.update(alerta)
        resumo_c2[f"diferenca_{base}"] = difs
        log(f"  falso − verdadeiro, {base:6s}: " + "  ".join(f"{c} {d:+.2f}" for c, d in difs.items())
            + (f"   <-- ≥ {LIMIAR_MARCA_PP:.0f} pp: {', '.join(alerta)}" if alerta else "   ok"))
    veredito_falsas = resumo_c2.get("corpus_falso", {}).get("pct_veredito", 0.0)
    vazou_veredito = veredito_falsas > LIMIAR_VEREDITO_FALSAS
    log(f"  veredito nas frases falsas do corpus: {veredito_falsas:.2f}% "
        f"(limiar {LIMIAR_VEREDITO_FALSAS}%) {'<-- VAZAMENTO CRIADO PELO REFINAMENTO' if vazou_veredito else 'ok'}")
    res["C2"] = {"por_grupo": resumo_c2, "marcas_acima_do_limiar": sorted(marcas_alerta),
                 "veredito_falsas_corpus_pct": veredito_falsas, "vazamento_veredito": vazou_veredito,
                 "passou": not marcas_alerta and not vazou_veredito}

    log("")
    log("=" * 72)
    log("C1 -- baseline de TAMANHO no braço refinado")
    log("=" * 72)
    t_tam = treinar(rep_tamanho(), por)
    av_tam_int, av_tam_fb = avaliar(t_tam, teste), avaliar(t_tam, fakebr)
    tam_int, tam_fb = pct(av_tam_int["bal"]), pct(av_tam_fb["bal"])
    log(f"  tamanho sozinho: interno {tam_int:.2f}% | Fake.br {tam_fb:.2f}% (C={t_tam['C']})")
    atalho_tamanho = av_tam_int["bal"] > LIMIAR_TAMANHO
    log(f"  {'> 60%: TAMANHO É ATALHO DECLARADO do braço refinado' if atalho_tamanho else '≤ 60%: ok'}")
    res["C1"] = {"interno": tam_int, "interno_ic95": ic(bal_bootstrap(av_tam_int, idx_int)),
                 "fakebr": tam_fb, "C": t_tam["C"], "atalho": atalho_tamanho, "passou": not atalho_tamanho}

    # ---- 2. os três braços -----------------------------------------------------
    BRACOS = {"bruto": rep_tfidf("texto_entrada"),
              "refinado": rep_tfidf("ideia_central"),
              "combinado": rep_tfidf("texto_entrada", "ideia_central")}
    log("")
    log("=" * 72)
    log("BRAÇOS -- rodada A (completo)")
    log("=" * 72)
    treinados, av, boot = {}, {}, {}
    res["bracos"] = {}
    for nome, rep in BRACOS.items():
        t = treinar(rep, por)
        ai, af = avaliar(t, teste), avaliar(t, fakebr)
        bi, bf = bal_bootstrap(ai, idx_int), bal_bootstrap(af, idx_fb)
        treinados[nome], av[nome], boot[nome] = t, (ai, af), (bi, bf)
        queda = ai["bal"] - af["bal"]
        res["bracos"][nome] = {
            "C": t["C"], "validacao": pct(t["validacao"]),
            "interno": pct(ai["bal"]), "interno_ic95": ic(bi),
            "interno_rev_falso": pct(ai["rev_falso"]), "interno_rev_verdadeiro": pct(ai["rev_verdadeiro"]),
            "fakebr": pct(af["bal"]), "fakebr_ic95": ic(bf),
            "fakebr_rev_falso": pct(af["rev_falso"]), "fakebr_rev_verdadeiro": pct(af["rev_verdadeiro"]),
            "queda_pp": pct(queda), "queda_ic95": ic(bi - bf),
            "fakebr_por_categoria": por_categoria(af, fakebr)}
        b = res["bracos"][nome]
        log(f"  {nome:9s} C={t['C']:<4} interno {b['interno']:.2f}% {b['interno_ic95']} | "
            f"Fake.br {b['fakebr']:.2f}% {b['fakebr_ic95']} | queda {b['queda_pp']:.2f} pp {b['queda_ic95']}")
        log(f"            revocação Fake.br FALSO {b['fakebr_rev_falso']:.2f}% | VERDADEIRO {b['fakebr_rev_verdadeiro']:.2f}%")
    rb = res["bracos"]["bruto"]
    log(f"  bruto × adendo 5: interno {rb['interno'] - 100 * REFERENCIA_BRUTO['interno']:+.2f} pp | "
        f"Fake.br {rb['fakebr'] - 100 * REFERENCIA_BRUTO['fakebr']:+.2f} pp "
        f"(perda de ids {100 * perda:.2f}%; se < 0,5%, espera-se < 0,5 pp)")
    res["bruto_vs_adendo5_pp"] = {"interno": round(rb["interno"] - 100 * REFERENCIA_BRUTO["interno"], 2),
                                  "fakebr": round(rb["fakebr"] - 100 * REFERENCIA_BRUTO["fakebr"], 2)}

    # Termos do refinado (R7) e quanto o combinado usa de cada bloco.
    t_ref = treinados["refinado"]
    nomes = t_ref["transformar"].vets[0].get_feature_names_out()
    coef = t_ref["modelo"].coef_[0]
    ordem = np.argsort(coef)
    res["termos_refinado"] = {"FALSO": [[nomes[i], round(float(coef[i]), 3)] for i in ordem[::-1][:25]],
                              "VERDADEIRO": [[nomes[i], round(float(coef[i]), 3)] for i in ordem[:25]]}
    log("  refinado, maior peso para FALSO: " + ", ".join(n for n, _ in res["termos_refinado"]["FALSO"][:15]))
    log("  refinado, maior peso para VERDADEIRO: " + ", ".join(n for n, _ in res["termos_refinado"]["VERDADEIRO"][:15]))
    t_comb = treinados["combinado"]
    n_bruto = len(t_comb["transformar"].vets[0].vocabulary_)
    cc = np.abs(t_comb["modelo"].coef_[0])
    massa_bruto = float(cc[:n_bruto].sum() / cc.sum())
    log(f"  combinado: {100 * massa_bruto:.1f}% da massa |coef| no bloco bruto, "
        f"{100 * (1 - massa_bruto):.1f}% no refinado")
    res["combinado_massa_coef_bloco_bruto_pct"] = pct(massa_bruto)

    # Consequências de C1 e C2.
    if atalho_tamanho:
        t = treinar(rep_juntas(rep_tfidf("ideia_central"), rep_tamanho()), por)
        ai, af = avaliar(t, teste), avaliar(t, fakebr)
        res["C1"]["refinado_mais_tamanho"] = {"interno": pct(ai["bal"]), "fakebr": pct(af["bal"]), "C": t["C"]}
        res["C1"]["refinado_acima_do_tamanho_pp"] = {
            "interno": round(res["bracos"]["refinado"]["interno"] - tam_int, 2),
            "fakebr": round(res["bracos"]["refinado"]["fakebr"] - tam_fb, 2)}
        log(f"  C1: refinado + tamanho {pct(ai['bal']):.2f}% / {pct(af['bal']):.2f}% | refinado acima do "
            f"tamanho {res['C1']['refinado_acima_do_tamanho_pp']}")
    sem_marca = sorted(marcas_alerta | ({"veredito"} if vazou_veredito else set()))
    if sem_marca:
        res["C2"]["refinado_sem_frases_marcadas"] = {}
        for marca in sem_marca + (["todas"] if len(sem_marca) > 1 else []):
            alvo = sem_marca if marca == "todas" else [marca]
            filtro = {s: [r for r in rs if not any(MARCAS[m].search(r["ideia_central"]) for m in alvo)]
                      for s, rs in por.items()}
            t = treinar(rep_tfidf("ideia_central"), filtro)
            ai, af = avaliar(t, filtro["teste"]), avaliar(t, filtro["fakebr"])
            res["C2"]["refinado_sem_frases_marcadas"][marca] = {
                "interno": pct(ai["bal"]), "fakebr": pct(af["bal"]),
                "n_teste": len(filtro["teste"]), "n_fakebr": len(filtro["fakebr"])}
            log(f"  C2: refinado sem frases com '{marca}': interno {pct(ai['bal']):.2f}% | "
                f"Fake.br {pct(af['bal']):.2f}%")

    # ---- C4 --------------------------------------------------------------------
    log("")
    log("=" * 72)
    log("C4 -- controles de pipeline no refinado e no combinado")
    log("=" * 72)
    res["C4"] = {"embaralhado": {}}
    c4_ok = True
    for nome in ("refinado", "combinado"):
        accs = []
        for s in SEMENTES_EMBARALHAR:
            t = treinar(BRACOS[nome], por, embaralhar=s, C=treinados[nome]["C"])
            accs.append(avaliar(t, teste)["bal"])
        ok = all(FAIXA_EMBARALHADO[0] <= a <= FAIXA_EMBARALHADO[1] for a in accs)
        c4_ok &= ok
        res["C4"]["embaralhado"][nome] = {"acuracias": [pct(a) for a in accs], "ok": ok}
        log(f"  {nome:9s} rótulos embaralhados: {', '.join(f'{pct(a):.2f}%' for a in accs)}  "
            f"{'ok' if ok else '<-- FORA DE 45–55%: VAZAMENTO NO PIPELINE, não interpretar'}")

    def norm(t):
        return re.sub(r"\s+", " ", t.lower()).strip()
    onde = defaultdict(set)
    for s in ("treino", "validacao", "teste"):
        for r in por[s]:
            onde[norm(r["ideia_central"])].add(s)
    cruzam = {f for f, ss in onde.items() if len(ss) > 1}
    n_reg_cruzam = sum(1 for s in ("treino", "validacao", "teste") for r in por[s] if norm(r["ideia_central"]) in cruzam)
    pct_cruzam = 100 * n_reg_cruzam / max(1, n_corpus)
    vistos_treino = {norm(r["ideia_central"]) for s in ("treino", "validacao") for r in por[s]}
    mascara = np.array([norm(r["ideia_central"]) not in vistos_treino for r in teste])
    ai = av["refinado"][0]
    sem = bal(ai["y"][mascara], ai["pred"][mascara]) if mascara.sum() else float("nan")
    res["C4"]["sobreposicao"] = {"frases_em_mais_de_uma_parte": len(cruzam), "registros": n_reg_cruzam,
                                 "pct_registros": round(pct_cruzam, 3),
                                 "teste_com_frase_vista": int((~mascara).sum()),
                                 "refinado_interno_sem_elas": pct(sem)}
    log(f"  sobreposição: {len(cruzam)} frases idênticas em mais de uma parte ({n_reg_cruzam} registros, "
        f"{pct_cruzam:.3f}%; previsão < 0,5%) | teste com frase já vista: {int((~mascara).sum())} | "
        f"refinado interno sem elas {pct(sem):.2f}%")
    res["C4"]["passou"] = c4_ok

    # ---- rodada B ---------------------------------------------------------------
    log("")
    log("BRAÇOS -- rodada B (sem entrada_com_veredito)")
    por_b = {s: [r for r in rs if not r.get("entrada_com_veredito")] for s, rs in por.items()}
    res["rodada_B"] = {}
    for nome, rep in BRACOS.items():
        t = treinar(rep, por_b)
        ai, af = avaliar(t, por_b["teste"]), avaliar(t, por_b["fakebr"])
        res["rodada_B"][nome] = {"interno": pct(ai["bal"]), "fakebr": pct(af["bal"]), "C": t["C"]}
        log(f"  {nome:9s} interno {pct(ai['bal']):.2f}% | Fake.br {pct(af['bal']):.2f}%")

    # ---- 3. família confirmatória ------------------------------------------------
    log("")
    log("=" * 72)
    log("FAMÍLIA CONFIRMATÓRIA (Holm, alfa 0,05)")
    log("=" * 72)
    (bi_b, bf_b), (bi_r, bf_r) = boot["bruto"], boot["refinado"]
    dif = (bi_b - bf_b) - (bi_r - bf_r)                    # queda bruto − queda refinado
    dif_obs = res["bracos"]["bruto"]["queda_pp"] - res["bracos"]["refinado"]["queda_pp"]
    p_h1 = max(1 / N_BOOT, min(1.0, 2 * min((dif <= 0).mean(), (dif >= 0).mean())))
    _, _, p_ref = mcnemar(av["bruto"][1]["acerto"], av["refinado"][1]["acerto"])
    _, _, p_comb = mcnemar(av["bruto"][1]["acerto"], av["combinado"][1]["acerto"])
    p_holm = holm([p_h1, p_ref, p_comb])
    ic_dif = ic(dif)
    h1 = ("confirmada" if ic_dif[0] > 0 else "refutada" if ic_dif[1] < 0 else "inconclusiva")
    fb_dif_ref = res["bracos"]["refinado"]["fakebr"] - res["bracos"]["bruto"]["fakebr"]
    if p_holm[1] < ALFA:
        via = "(a)" if fb_dif_ref > 0 else "(b)"
    else:
        via = "(b) -- Fake.br sem diferença significativa; a queda menor viria só do interno"
    fb_dif_comb = res["bracos"]["combinado"]["fakebr"] - res["bracos"]["bruto"]["fakebr"]
    ic_comb = ic(boot["combinado"][1] - boot["bruto"][1])
    ic_ref_fb = ic(boot["refinado"][1] - boot["bruto"][1])
    res["confirmatoria"] = {
        "H1": {"queda_bruto_menos_refinado_pp": round(dif_obs, 2), "ic95": ic_dif, "p": p_h1,
               "p_holm": p_holm[0], "resultado": h1, "via": via if h1 == "confirmada" else None},
        "refinado_vs_bruto_fakebr": {"dif_pp": round(fb_dif_ref, 2), "ic95": ic_ref_fb, "p": p_ref, "p_holm": p_holm[1]},
        "combinado_vs_bruto_fakebr": {"dif_pp": round(fb_dif_comb, 2), "ic95": ic_comb, "p": p_comb, "p_holm": p_holm[2]}}
    log(f"  H1 queda bruto − queda refinado: {dif_obs:+.2f} pp {ic_dif} | p {p_h1:.4g} (Holm {p_holm[0]:.4g}) "
        f"-> {h1.upper()}" + (f" {via}" if h1 == "confirmada" else ""))
    log(f"  refinado − bruto no Fake.br: {fb_dif_ref:+.2f} pp {ic_ref_fb} | McNemar p {p_ref:.4g} (Holm {p_holm[1]:.4g})")
    log(f"  combinado − bruto no Fake.br: {fb_dif_comb:+.2f} pp {ic_comb} | McNemar p {p_comb:.4g} (Holm {p_holm[2]:.4g})")

    # ---- 4. validade, leituras, confronto -------------------------------------
    log("")
    log("=" * 72)
    log("VALIDADE E LEITURAS")
    log("=" * 72)
    valido = res["C1"]["passou"] and res["C2"]["passou"] and res["C4"]["passou"]
    log(f"  C1 {'ok' if res['C1']['passou'] else 'ATALHO'} | C2 {'ok' if res['C2']['passou'] else 'ATALHO'} | "
        f"C4 {'ok' if res['C4']['passou'] else 'FALHOU'} | C3 pendente (auditoria_fidelidade.py)")
    log("  " + ("o refinado PODE ser lido como 'conteúdo sem forma', se C3 também passar" if valido else
                "o refinado NÃO pode ser lido como 'conteúdo sem forma' sem as ressalvas acima"))
    r = res["bracos"]
    leituras = {"refinado_interno": leitura_interno(r["refinado"]["interno"]),
                "refinado_fakebr": leitura_fakebr(r["refinado"]["fakebr"]),
                "combinado": leitura_combinado(fb_dif_comb, ic_comb)}
    for k, v in leituras.items():
        log(f"  {k:18s} {v}")
    res["validade"] = {"C1_C2_C4_passaram": valido, "C3": "pendente"}
    res["leituras"] = leituras

    valores = {"R1": r["refinado"]["interno"], "R2": r["refinado"]["fakebr"], "R3": r["refinado"]["queda_pp"],
               "R4": r["combinado"]["interno"], "R5": r["combinado"]["fakebr"],
               "R6": abs(r["refinado"]["fakebr_rev_falso"] - r["refinado"]["fakebr_rev_verdadeiro"]),
               "C1": tam_int}
    log("")
    log(f"  {'':4s} {'resultado':>10s}   {'§2':>12s}     {'pesquisador':>12s}")
    confronto = {}
    for k, v in valores.items():
        s2, ps = PREVISOES_SECAO2[k], PREVISOES_PESQUISADOR[k]
        confronto[k] = {"valor": round(v, 2), "secao2": dentro(v, s2), "pesquisador": dentro(v, ps)}
        log(f"  {k:4s} {v:10.2f}   {str(s2):>12s} {'✓' if confronto[k]['secao2'] else '✗'}   "
            f"{str(ps):>12s} {'✓' if confronto[k]['pesquisador'] else '✗'}")
    confronto["C2"] = {"valor": sorted(marcas_alerta) or "nenhuma",
                       "pesquisador": not marcas_alerta, "previsao": PESQUISADOR_C2}
    confronto["H1"] = {"valor": h1, "pesquisador": h1 == PESQUISADOR_H1}
    log(f"  C2   marcas ≥ 10 pp: {confronto['C2']['valor']} | pesquisador "
        f"{'✓' if confronto['C2']['pesquisador'] else '✗'}")
    log(f"  H1   {h1} | pesquisador ({PESQUISADOR_H1}) {'✓' if confronto['H1']['pesquisador'] else '✗'}")
    res["confronto"] = confronto

    os.makedirs(os.path.dirname(arq_res), exist_ok=True)
    with open(arq_res, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=float)
    log(f"\nsalvo em {arq_res}")


if __name__ == "__main__":
    main()
