# -*- coding: utf-8 -*-
"""Levanta, a partir dos PRÓPRIOS DADOS, o vocabulário do checador e mede quanto
dele vaza para a entrada do experimento -- nas duas classes. Somente leitura.

    python scripts/auditar_vazamento.py

Por que dirigido por dados: todas as contagens anteriores vinham de listas
escritas à mão (ATRIBUTIVOS, NARRATIVA_CHECADOR), e uma lista só acha o que quem
a escreveu previu. O piloto do refinamento mostrou entradas que nenhuma regra
pegava ("7 Fake news sobre a morte de...", "se espalhou feito pólvora").

Método:
  A = TEXTO DO CHECADOR: o artigo do Boatos.org MENOS os virais transcritos
      nele -- sobra a voz do jornalista (e o rodapé do site).
  B = BOATO DE REFERÊNCIA: virais de alta confiança -- vieram do <blockquote>,
      literalidade conferida, sem nenhuma marcação de defeito.
  Termo do checador = palavra ou par de palavras com frequência de documento
  muito maior em A do que em B (log-razão com suavização), presente em pelo
  menos MIN_DF_A documentos de A e em no máximo MAX_PCT_B dos de B.

Depois: quantas entradas do subconjunto pareado (falsas E verdadeiras) contêm
esses termos, quantos por entrada, e exemplos em cada faixa.
"""
import os
import re
import sys
import json
import math
import random
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_FINAL = os.path.join(_RAIZ, "datasets", "dataset_tcc_fake_news_final.json")
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
SAIDA = os.path.join(_RAIZ, "relatorios", "auditoria_vazamento.json")

MIN_DF_A = 80          # o termo precisa ser hábito do checador, não acaso
MAX_PCT_B = 0.5        # e raro nos boatos de referência (% dos documentos)
TOP = 80
SEED = 0


def termos(texto):
    """Palavras e pares de palavras, em minúsculas, sem números puros."""
    p = [w for w in re.findall(r"[a-zà-ú]+", (texto or "").lower()) if len(w) > 1]
    return set(p) | {f"{a} {b}" for a, b in zip(p, p[1:])}


def norm(t):
    return re.sub(r"\s+", " ", (t or "")).strip()


def main():
    with open(ARQ_FINAL, encoding="utf-8") as f:
        falsas = json.load(f)
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]

    # ---- A: voz do checador (artigo menos os virais), um documento por artigo
    virais_por_url = defaultdict(list)
    artigo_por_url = {}
    for r in falsas:
        virais_por_url[r["url_origem_fake"]].append(norm(r.get("texto_viral_bruto")))
        artigo_por_url[r["url_origem_fake"]] = norm(r.get("texto_bruto_jornalista"))
    docs_a = []
    for url, artigo in artigo_por_url.items():
        for v in sorted(virais_por_url[url], key=len, reverse=True):
            if len(v) >= 30:
                artigo = artigo.replace(v, " ")
        docs_a.append(termos(artigo))

    # ---- B: boatos de referência
    ref = {r["id"] for r in corpus
           if r["veredito"] == "FALSO" and str(r.get("origem_viral", "")).startswith("html")
           and r.get("verificacao_viral") == "conferido" and r.get("duplicata_canonica")
           and not r["entrada_com_veredito"] and not r["entrada_narrativa_checador"]
           and not r["entrada_degenerada"]}
    docs_b = [termos(r["texto_entrada"]) for r in corpus if r["id"] in ref]
    print(f"A (voz do checador): {len(docs_a):,} artigos | B (boatos de referência): {len(docs_b):,}"
          .replace(",", "."))

    df_a, df_b = Counter(), Counter()
    for d in docs_a:
        df_a.update(d)
    for d in docs_b:
        df_b.update(d)
    na, nb = len(docs_a), len(docs_b)
    candidatos = []
    for t, fa in df_a.items():
        if fa < MIN_DF_A:
            continue
        pb = 100 * df_b[t] / nb
        if pb > MAX_PCT_B:
            continue
        lr = math.log((fa + 1) / (na + 2)) - math.log((df_b[t] + 1) / (nb + 2))
        candidatos.append((lr, t, 100 * fa / na, pb))
    candidatos.sort(reverse=True)
    # Pares cujas duas palavras já estão na lista como unigramas não somam
    # informação; ficam só os pares que dizem algo além das palavras soltas.
    escolhidos = []
    for lr, t, pa, pb in candidatos:
        if " " in t and all(w in {x[1] for x in escolhidos} for w in t.split()):
            continue
        escolhidos.append((lr, t, pa, pb))
        if len(escolhidos) == TOP:
            break
    vocab = {t for _, t, _, _ in escolhidos}

    print(f"\n{TOP} termos do checador (em % dos documentos: checador × boato de referência):")
    for i in range(0, len(escolhidos), 4):
        print("  " + " | ".join(f"{t[:22]:22s} {pa:4.1f}×{pb:3.1f}" for _, t, pa, pb in escolhidos[i:i + 4]))

    # ---- presença nas entradas do pareado
    par = [r for r in corpus if r["subconjunto_pareado"]]
    contagem = {}
    print("\nentradas do PAREADO com termos do checador:")
    print(f"  {'classe':12s} {'≥1 termo':>10s} {'≥2':>8s} {'≥3':>8s}")
    exemplos = defaultdict(list)
    for classe in ("FALSO", "VERDADEIRO"):
        regs = [r for r in par if r["veredito"] == classe]
        ns = []
        for r in regs:
            achados = sorted(vocab & termos(r["texto_entrada"]))
            ns.append(len(achados))
            if achados:
                exemplos[(classe, min(len(achados), 3))].append((achados, r["texto_entrada"]))
        k = len(regs)
        linha = {f">={x}": round(100 * sum(n >= x for n in ns) / k, 1) for x in (1, 2, 3)}
        contagem[classe] = linha
        print(f"  {classe:12s} {linha['>=1']:9.1f}% {linha['>=2']:7.1f}% {linha['>=3']:7.1f}%")

    termo_freq = {c: Counter() for c in ("FALSO", "VERDADEIRO")}
    for r in par:
        termo_freq[r["veredito"]].update(vocab & termos(r["texto_entrada"]))
    print("\ntermos do checador mais frequentes NAS ENTRADAS (falsas | verdadeiras):")
    for t, n in termo_freq["FALSO"].most_common(15):
        print(f"  {t:28s} {n:5d} | {termo_freq['VERDADEIRO'][t]:5d}")

    rng = random.Random(SEED)
    for chave in (("FALSO", 1), ("FALSO", 2), ("FALSO", 3), ("VERDADEIRO", 1), ("VERDADEIRO", 2)):
        lst = exemplos.get(chave, [])
        rng.shuffle(lst)
        print(f"\nexemplos {chave[0]} com {'≥' if chave[1] == 3 else ''}{chave[1]} termo(s) ({len(lst)}):")
        for achados, t in lst[:6]:
            print(f"  {achados[:4]!s:45.45s} {t[:110]!r}")

    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump({"termos": [{"termo": t, "pct_checador": round(pa, 2), "pct_boato_ref": round(pb, 2),
                               "log_razao": round(lr, 2)} for lr, t, pa, pb in escolhidos],
                   "presenca_pareado": contagem,
                   "freq_entradas": {c: dict(v.most_common(40)) for c, v in termo_freq.items()}},
                  f, indent=2, ensure_ascii=False)
    print(f"\nsalvo em relatorios/{os.path.basename(SAIDA)}")


if __name__ == "__main__":
    main()
