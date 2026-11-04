# -*- coding: utf-8 -*-
"""Alucinação na extração por LLM: quantas falsas da via `llm` não existem no artigo?
Pré-registro: `relatorios/PREREGISTRO_ALUCINACAO.md`.

    python scripts/alucinacao.py --medir      # cobertura lexical, faixas, controle dos casos conhecidos
    python scripts/alucinacao.py --planilha   # 30 + 20 itens para validação humana, sem cobertura
    python scripts/alucinacao.py --apurar     # depois da validação

Cobertura lexical: fração das palavras de conteúdo do texto que aparecem em
qualquer lugar do artigo. Paráfrase mantém as palavras; invenção não.
"""
import os
import re
import sys
import json
import random
import unicodedata
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_BRUTO = os.path.join(_RAIZ, "datasets", "dataset_bruto.json")
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
PASTA = os.path.join(_RAIZ, "relatorios", "alucinacao")
ARQ_MEDIDA = os.path.join(PASTA, "resultado_cobertura.json")
ARQ_AMOSTRA = os.path.join(PASTA, "amostra_validacao.json")
ARQ_PLANILHA = os.path.join(PASTA, "planilha_validacao.xlsx")
ARQ_RESULTADO = os.path.join(PASTA, "resultado_validacao.json")
LIMIAR_BAIXO, LIMIAR_ALTO = 0.5, 0.8
TETO_FONTE = 5000
MIN_PALAVRAS = 3
N_BAIXA, N_MEDIA = 30, 20
SEED = 0
CATEGORIAS = ["existe_no_artigo", "inventado", "nao_da_para_saber"]
# Casos achados na anotação manual das 200 falsas (controle de sanidade).
CONHECIDOS = ["luiz-gonzaga-criou-musica-xote-dos-cabeludos", "torre-eiffel-fica-verde-e-amarela"]
VAZIAS = set("""
para pela pelo pelas pelos como mais mas porque quando onde sobre entre depois antes
ainda tambem muito muita muitos muitas esse essa esses essas este esta estes estas isso
isto aquele aquela aquilo seus suas dele dela deles delas nosso nossa voce voces eles
elas todo toda todos todas cada outro outra outros outras mesmo mesma qual quais quem
sera seria foram eram esta estao estava tinha tenho temos fazer feito pode podem vai
vamos aqui agora sendo sido serem desde ate apos sem sob numa num uma umas uns pois
""".split())


def normalizar(t):
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def palavras(t):
    return [p for p in re.findall(r"[a-z0-9]+", normalizar(t)) if len(p) >= 4 and p not in VAZIAS]


def faixa(c):
    return "baixa" if c < LIMIAR_BAIXO else "media" if c < LIMIAR_ALTO else "alta"


def itens():
    d = json.load(open(ARQ_BRUTO, encoding="utf-8"))
    d = d if isinstance(d, list) else list(d.values())
    fontes = {(a.get("url_origem") or "").rstrip("/"): a.get("texto_bruto_jornalista") or "" for a in d}
    saida, curtos = [], 0
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        r = json.loads(l)
        if r["veredito"] != "FALSO" or r.get("origem_viral") != "llm":
            continue
        fonte = fontes.get((r.get("url") or "").rstrip("/"), "")
        ps = palavras(r["texto_entrada"])
        if len(ps) < MIN_PALAVRAS:
            curtos += 1
            continue
        vocab = set(palavras(fonte))
        cob = sum(p in vocab for p in ps) / len(ps)
        saida.append({"id": r["id"], "url": r["url"], "pareado": bool(r.get("subconjunto_pareado")),
                      "cobertura": round(cob, 3), "faixa": faixa(cob), "fonte_cortada": len(fonte) >= TETO_FONTE,
                      "verificacao_viral": r.get("verificacao_viral"), "texto": r["texto_entrada"]})
    return saida, curtos


def resumo(xs):
    n = len(xs)
    c = Counter(x["faixa"] for x in xs)
    return {"n": n, **{f: [c[f], round(100 * c[f] / n, 2) if n else None] for f in ("baixa", "media", "alta")}}


def medir():
    xs, curtos = itens()
    completos = [x for x in xs if not x["fonte_cortada"]]
    res = {"n_avaliados": len(xs), "fora_por_poucas_palavras": curtos,
           "todos": resumo(xs), "fonte_completa": resumo(completos), "fonte_cortada": resumo([x for x in xs if x["fonte_cortada"]]),
           "pareado_fonte_completa": resumo([x for x in completos if x["pareado"]]),
           "faixa_por_verificacao_viral": {v: resumo([x for x in xs if str(x["verificacao_viral"]) == v])
                                           for v in sorted({str(x["verificacao_viral"]) for x in xs})}}
    # Concentração (A2): artigos com alguma instância baixa, e quantos têm todas baixas.
    por_art = defaultdict(list)
    for x in completos:
        por_art[x["url"]].append(x["faixa"])
    com_baixa = [fs for fs in por_art.values() if "baixa" in fs]
    todas = sum(all(f == "baixa" for f in fs) for fs in com_baixa)
    res["concentracao"] = {"artigos_com_baixa": len(com_baixa), "artigos_todas_baixas": todas,
                           "pct": round(100 * todas / len(com_baixa), 1) if com_baixa else None}
    # Controle de sanidade: os casos conhecidos devem cair na faixa baixa.
    conhecidos = [{"url": x["url"], "cobertura": x["cobertura"], "faixa": x["faixa"]}
                  for x in xs if any(k in x["url"] for k in CONHECIDOS)]
    res["controle_conhecidos"] = {"itens": conhecidos, "todos_baixa": all(c["faixa"] == "baixa" for c in conhecidos),
                                  "n": len(conhecidos)}
    res["confronto_parcial"] = {"A1_pct_baixa_fonte_completa": res["fonte_completa"]["baixa"][1],
                                "A1_ok": res["fonte_completa"]["baixa"][1] <= 3,
                                "A2_pct": res["concentracao"]["pct"],
                                "A2_ok": (res["concentracao"]["pct"] or 0) >= 50}
    os.makedirs(PASTA, exist_ok=True)
    with open(ARQ_MEDIDA, "w", encoding="utf-8") as f:
        json.dump(res | {"itens": [{k: v for k, v in x.items() if k != "texto"} for x in xs]}, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def planilha():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    if os.path.exists(ARQ_PLANILHA) and "--sobrescrever" not in sys.argv:
        sys.exit(f"[!] {ARQ_PLANILHA} já existe -- pode ter anotações.")
    xs, _ = itens()
    completos = sorted([x for x in xs if not x["fonte_cortada"]], key=lambda x: x["id"])
    rng = random.Random(SEED)
    # Desvio declarado (adendo 1): a faixa média tem menos de N_MEDIA itens; entram todos.
    medias = [x for x in completos if x["faixa"] == "media"]
    amostra = (rng.sample([x for x in completos if x["faixa"] == "baixa"], N_BAIXA)
               + rng.sample(medias, min(N_MEDIA, len(medias))))
    rng.shuffle(amostra)                                  # cegamento: a ordem não revela a faixa
    with open(ARQ_AMOSTRA, "w", encoding="utf-8") as f:
        json.dump([{"n": i, "id": x["id"], "faixa": x["faixa"]} for i, x in enumerate(amostra, start=1)],
                  f, indent=1, ensure_ascii=False)
    wb = Workbook()
    ws = wb.active
    ws.title = "validacao"
    ws.append(["n", "id", "texto", "link da checagem", "o texto…", "observacao"])
    for c in ws[1]:
        c.font = Font(bold=True)
    for i, x in enumerate(amostra, start=1):
        ws.append([i, x["id"], x["texto"], x["url"], None, None])
    for linha in ws.iter_rows(min_row=2):
        for c in linha:
            c.alignment = Alignment(wrap_text=c.column == 3, vertical="top")
    for col, larg in zip("ABCDEF", (5, 16, 80, 40, 22, 30)):
        ws.column_dimensions[col].width = larg
    dv = DataValidation(type="list", formula1='"' + ",".join(CATEGORIAS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"E2:E{len(amostra) + 1}")
    wb.save(ARQ_PLANILHA)
    print(f"planilha com {len(amostra)} itens ({N_BAIXA} baixa + {len(amostra) - N_BAIXA} média, embaralhados) -> {ARQ_PLANILHA}")


def apurar():
    from openpyxl import load_workbook
    faixas = {x["n"]: x["faixa"] for x in json.load(open(ARQ_AMOSTRA, encoding="utf-8"))}
    ws = load_workbook(ARQ_PLANILHA)["validacao"]
    linhas = [[c.value for c in r] for r in ws.iter_rows(min_row=2) if r[0].value not in (None, "")]
    if any(l[4] not in CATEGORIAS for l in linhas):
        sys.exit("há linhas sem categoria válida")
    med = json.load(open(ARQ_MEDIDA, encoding="utf-8"))
    res = {}
    for fx in ("baixa", "media"):
        cont = Counter(l[4] for l in linhas if faixas[l[0]] == fx)
        n = sum(cont.values())
        res[fx] = {"contagem": dict(cont), "pct_inventado": round(100 * cont["inventado"] / n, 1)}
    pc = med["fonte_completa"]
    estimada = (pc["baixa"][1] * res["baixa"]["pct_inventado"] + pc["media"][1] * res["media"]["pct_inventado"]) / 100
    pp = med["pareado_fonte_completa"]
    estimada_par = (pp["baixa"][1] * res["baixa"]["pct_inventado"] + pp["media"][1] * res["media"]["pct_inventado"]) / 100
    res["taxa_estimada_invencao_pct"] = round(estimada, 2)
    res["taxa_estimada_invencao_pareado_pct"] = round(estimada_par, 2)
    res["confronto"] = {"A3": [res["baixa"]["pct_inventado"], res["baixa"]["pct_inventado"] >= 70],
                        "A4": [res["media"]["pct_inventado"], res["media"]["pct_inventado"] <= 20]}
    a3 = res["baixa"]["pct_inventado"] >= 70
    res["leitura"] = ("defeito material: marcar, retirar e refazer o modelo de referência" if a3 and estimada_par >= 1 else
                      "defeito pequeno: marcar, retirar e declarar" if a3 else
                      "medida fraca: relatar só como limite superior")
    with open(ARQ_RESULTADO, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"--medir": medir, "--planilha": planilha, "--apurar": apurar}.get(
        next((a for a in sys.argv[1:] if a.startswith("--")), ""), lambda: print(__doc__))()
