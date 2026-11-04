# -*- coding: utf-8 -*-
"""Auditoria de FIDELIDADE do refinamento -- checagem C3 do
`relatorios/PREREGISTRO_BRACO_REFINADO.md`.

    python scripts/auditoria_fidelidade.py --sortear     # AGORA, antes de existir qualquer frase
    python scripts/auditoria_fidelidade.py --planilha    # depois do juntar_refinamento.py
    python scripts/auditoria_fidelidade.py --apurar      # depois de preencher a planilha

Por que existe: o piloto mostrou inversões de sentido ("Congresso vota" ->
"aprovou"; notícia verdadeira virando falsa). Se o refinamento distorce uma
classe mais que a outra, o braço refinado não mede "o mesmo conteúdo sem a
forma" -- mede também a distorção.

DESENHO (fixado no pré-registro)
  - Amostra: 100 frases por classe do `teste` do corpus (200), sorteadas por id
    sobre a FILA, que existe antes do refinamento. O sorteio é commitado antes
    de qualquer frase ser lida.
  - A planilha não mostra id nem classe (o id começa por F ou V). A ordem é
    embaralhada. Ressalva declarada: o texto bruto denuncia a classe pela forma,
    então a auditoria não é cega de fato.
  - Quatro categorias: fiel, mudou_sentido, inventou, perdeu_central.
  - Limiar: diferença entre classes em (mudou_sentido + inventou) >= 10 pp ->
    o braço refinado não pode ser lido como "o mesmo conteúdo sem a forma".

Arquivos (relatorios/fidelidade/):
    amostra_fidelidade.json   ids sorteados, na ordem da planilha, + sha da fila
    planilha_fidelidade.xlsx  o que o anotador vê e preenche
    resultado_fidelidade.json saída do --apurar
"""
import os
import sys
import json
import random
import hashlib
from collections import Counter

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anotacao_manual import kappa, wilson          # noqa: E402
from refinar_cego import ARQ_FILA, ler_jsonl        # noqa: E402

ARQ_BRACO = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
PASTA = os.path.join(_RAIZ, "relatorios", "fidelidade")
ARQ_AMOSTRA = os.path.join(PASTA, "amostra_fidelidade.json")
ARQ_PLANILHA = os.path.join(PASTA, "planilha_fidelidade.xlsx")
ARQ_RESULTADO = os.path.join(PASTA, "resultado_fidelidade.json")
SEED = 20261006
N_POR_CLASSE = 100
LIMIAR_PP = 10.0
PREVISAO_PESQUISADOR = "falsa"          # §5, C3: a classe que o refinamento distorce mais

CATEGORIAS = ["fiel", "mudou_sentido", "inventou", "perdeu_central"]
DISTORCAO = ("mudou_sentido", "inventou")
DEFINICOES = [
    ("fiel", "A frase diz o que o texto diz.",
     "Texto: 'Governo anuncia reajuste de 5% no salário mínimo' -> 'O governo reajustou o salário mínimo em 5%.'"),
    ("mudou_sentido", "Alterou modalidade ou fato: 'vota' virou 'aprovou', negação perdida ou "
                      "acrescentada, tempo trocado, sujeito trocado.",
     "Texto: 'Congresso vota hoje a reforma' -> 'O Congresso aprovou a reforma.'"),
    ("inventou", "Acrescentou fato, número, nome ou data que não está no texto.",
     "Texto sem data -> frase com 'em 2022'; texto sem nome -> frase com o nome de um ministro."),
    ("perdeu_central", "Fiel no que diz, mas descreve um detalhe e não a alegação principal.",
     "Corrente sobre vacina que 'altera o DNA' -> 'Uma enfermeira trabalha em Curitiba.'"),
]


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def classe(i):
    return "FALSO" if i.startswith("F") else "VERDADEIRO"


# ==========================================================================
def sortear():
    if os.path.exists(ARQ_AMOSTRA) and "--sobrescrever" not in sys.argv:
        sys.exit(f"[!] {ARQ_AMOSTRA} já existe. O sorteio é feito uma vez só.")
    fila = ler_jsonl(ARQ_FILA)
    rng = random.Random(SEED)
    amostra = []
    for grupo in ("corpus_falso", "corpus_verdadeiro"):
        ids = sorted(r["id"] for r in fila if r["grupo"] == grupo and r["split"] == "teste")
        amostra += rng.sample(ids, N_POR_CLASSE)
    rng.shuffle(amostra)             # a ordem da planilha não revela a classe
    os.makedirs(PASTA, exist_ok=True)
    with open(ARQ_AMOSTRA, "w", encoding="utf-8") as f:
        json.dump({"semente": SEED, "fila_sha256": sha256(ARQ_FILA), "n_por_classe": N_POR_CLASSE,
                   "populacao": "split teste do corpus pareado, na fila do refinamento",
                   "ids_em_ordem": amostra}, f, indent=2, ensure_ascii=False)
    print(f"{len(amostra)} ids sorteados ({dict(Counter(classe(i) for i in amostra))}) -> {ARQ_AMOSTRA}")
    print(f"  semente {SEED} | fila sha256 {sha256(ARQ_FILA)[:16]}")


# ==========================================================================
def planilha():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    if os.path.exists(ARQ_PLANILHA) and "--sobrescrever" not in sys.argv:
        sys.exit(f"[!] {ARQ_PLANILHA} já existe -- pode ter anotações.")
    amostra = json.load(open(ARQ_AMOSTRA, encoding="utf-8"))
    braco = {r["id"]: r for r in ler_jsonl(ARQ_BRACO)}
    faltam = [i for i in amostra["ids_em_ordem"] if i not in braco]
    if faltam:
        print(f"[!] {len(faltam)} ids da amostra sem frase refinada (falha no refinamento): ficam fora, "
              f"relatar -- {dict(Counter(classe(i) for i in faltam))}")

    wb = Workbook()
    ins = wb.active
    ins.title = "instrucoes"
    ins["A1"] = "Fidelidade: a frase refinada (coluna C) diz o que o texto original (coluna B) diz?"
    ins["A1"].font = Font(bold=True, size=13)
    ins["A3"] = ("Compare as duas colunas e escolha UMA categoria na coluna D (lista suspensa). Não "
                 "julgue se o conteúdo é verdadeiro ou falso -- só se a frase é fiel ao texto. Se houver "
                 "mais de um problema, escolha o mais grave na ordem: mudou_sentido, inventou, "
                 "perdeu_central.")
    ins["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ins.row_dimensions[3].height = 60
    ins["A5"], ins["B5"], ins["C5"] = "categoria", "quando usar", "exemplo"
    for c in ("A5", "B5", "C5"):
        ins[c].font = Font(bold=True)
    for i, (cat, quando, ex) in enumerate(DEFINICOES, start=6):
        ins[f"A{i}"], ins[f"B{i}"], ins[f"C{i}"] = cat, quando, ex
        for c in "ABC":
            ins[f"{c}{i}"].alignment = Alignment(wrap_text=True, vertical="top")
        ins.row_dimensions[i].height = 60
    ins[f"A{6 + len(DEFINICOES) + 1}"] = ("Segundo anotador: preencha F e G SEM olhar D e E "
                                          "(oculte-as antes de começar).")
    ins.column_dimensions["A"].width = 16
    ins.column_dimensions["B"].width = 70
    ins.column_dimensions["C"].width = 70

    ws = wb.create_sheet("anotacao")
    ws.append(["n", "texto original", "frase refinada", "anotador_1", "observacao_1",
               "anotador_2", "observacao_2"])
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    linhas = 0
    for n, i in enumerate(amostra["ids_em_ordem"], start=1):
        if i in braco:
            ws.append([n, braco[i]["texto_entrada"], braco[i]["ideia_central"], None, None, None, None])
            linhas += 1
    for linha in ws.iter_rows(min_row=2):
        for c in linha:
            c.alignment = Alignment(wrap_text=c.column in (2, 3), vertical="top")
    for col, larg in zip("ABCDEFG", (5, 70, 50, 15, 30, 15, 30)):
        ws.column_dimensions[col].width = larg
    ws.freeze_panes = "B2"
    lista = DataValidation(type="list", formula1='"' + ",".join(CATEGORIAS) + '"', allow_blank=True,
                           showErrorMessage=True, errorTitle="Categoria inválida",
                           error="Use uma das opções da lista: " + ", ".join(CATEGORIAS))
    ws.add_data_validation(lista)
    lista.add(f"D2:D{linhas + 1}")
    lista.add(f"F2:F{linhas + 1}")
    wb.save(ARQ_PLANILHA)
    print(f"planilha com {linhas} pares -> {ARQ_PLANILHA}")


# ==========================================================================
def apurar():
    from openpyxl import load_workbook
    amostra = json.load(open(ARQ_AMOSTRA, encoding="utf-8"))
    ordem = amostra["ids_em_ordem"]
    ws = load_workbook(ARQ_PLANILHA)["anotacao"]
    cab = [c.value for c in ws[1]]
    linhas = [dict(zip(cab, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
    # Editores de planilha às vezes salvam linhas vazias depois da última (ONLYOFFICE, 09/10/2026).
    linhas = [l for l in linhas if l.get("n") not in (None, "")]
    an1 = {ordem[int(l["n"]) - 1]: (l["anotador_1"] or "").strip().lower() for l in linhas}
    an2 = {ordem[int(l["n"]) - 1]: (l["anotador_2"] or "").strip().lower() for l in linhas}
    feitos = {i: c for i, c in an1.items() if c in CATEGORIAS}
    print(f"anotados: {len(feitos)} de {len(linhas)}")
    if not feitos:
        return

    resultado = {"n": len(feitos), "por_classe": {}}
    distorcao = {}
    for cl in ("FALSO", "VERDADEIRO"):
        cs = [c for i, c in feitos.items() if classe(i) == cl]
        n = len(cs)
        cont = Counter(cs)
        print(f"\n{cl} (n={n})")
        bloco = {"n": n, "categorias": {}}
        for c in CATEGORIAS:
            lo, hi = wilson(cont[c], n)
            bloco["categorias"][c] = {"n": cont[c], "pct": round(100 * cont[c] / max(1, n), 1),
                                      "ic95": [round(100 * lo, 1), round(100 * hi, 1)]}
            print(f"  {c:15s} {cont[c]:4d}  {100 * cont[c] / max(1, n):5.1f}%  [{100 * lo:4.1f}; {100 * hi:4.1f}]")
        k = sum(cont[c] for c in DISTORCAO)
        lo, hi = wilson(k, n)
        distorcao[cl] = 100 * k / max(1, n)
        bloco["distorcao"] = {"n": k, "pct": round(distorcao[cl], 1), "ic95": [round(100 * lo, 1), round(100 * hi, 1)]}
        print(f"  DISTORÇÃO (mudou + inventou) {k} = {distorcao[cl]:.1f}% [{100 * lo:.1f}; {100 * hi:.1f}]")
        resultado["por_classe"][cl] = bloco

    dif = distorcao["FALSO"] - distorcao["VERDADEIRO"]
    mais = "falsa" if dif > 0 else "verdadeira" if dif < 0 else "nenhuma"
    passou = abs(dif) < LIMIAR_PP
    print(f"\ndistorção FALSO − VERDADEIRO: {dif:+.1f} pp (limiar {LIMIAR_PP:.0f}) -> "
          + ("ok" if passou else "o braço refinado NÃO pode ser lido como 'o mesmo conteúdo sem a forma'"))
    print(f"classe mais distorcida: {mais} | previsão do pesquisador: {PREVISAO_PESQUISADOR} "
          f"{'✓' if mais == PREVISAO_PESQUISADOR else '✗'}")
    resultado.update({"distorcao_falso_menos_verdadeiro_pp": round(dif, 1), "passou": passou,
                      "classe_mais_distorcida": mais,
                      "previsao_pesquisador_acertou": mais == PREVISAO_PESQUISADOR})

    pares = [(c, an2[i]) for i, c in feitos.items() if an2.get(i) in CATEGORIAS]
    if pares:
        k = kappa([a for a, _ in pares], [b for _, b in pares])
        kb = kappa([a in DISTORCAO for a, _ in pares], [b in DISTORCAO for _, b in pares])
        print(f"\nconcordância (n={len(pares)}): kappa 4 categorias {k:.2f} | distorção sim × não {kb:.2f}")
        resultado["kappa"] = {"n": len(pares), "quatro_categorias": round(k, 3), "distorcao": round(kb, 3)}

    # Adjudicação (coluna "consenso"): onde os dois anotadores discordaram, o
    # pesquisador decidiu depois de ver os dois argumentos. Relatada À PARTE: o
    # resultado principal continua sendo o do anotador 1, às cegas.
    if "consenso" in cab:
        cons = {ordem[int(l["n"]) - 1]: (l["consenso"] or "").strip().lower() for l in linhas}
        cons = {i: c for i, c in cons.items() if c in CATEGORIAS}
        if len(cons) == len(feitos):
            print("\nADJUDICADO (consenso):")
            adj, dist_adj = {}, {}
            for cl in ("FALSO", "VERDADEIRO"):
                cs = [c for i, c in cons.items() if classe(i) == cl]
                cont = Counter(cs)
                k = sum(cont[c] for c in DISTORCAO)
                lo, hi = wilson(k, len(cs))
                dist_adj[cl] = 100 * k / max(1, len(cs))
                adj[cl] = {"n": len(cs), "categorias": dict(cont),
                           "distorcao": {"n": k, "pct": round(dist_adj[cl], 1),
                                         "ic95": [round(100 * lo, 1), round(100 * hi, 1)]}}
                print(f"  {cl:10s} {dict(cont)} | distorção {k} = {dist_adj[cl]:.1f}% [{100 * lo:.1f}; {100 * hi:.1f}]")
            d = dist_adj["FALSO"] - dist_adj["VERDADEIRO"]
            print(f"  distorção FALSO − VERDADEIRO: {d:+.1f} pp -> {'ok' if abs(d) < LIMIAR_PP else 'acima do limiar'}")
            mudaram = sum(1 for i in cons if cons[i] != feitos[i])
            resultado["adjudicado"] = {"por_classe": adj, "distorcao_falso_menos_verdadeiro_pp": round(d, 1),
                                       "passou": abs(d) < LIMIAR_PP, "itens_alterados_vs_anotador_1": mudaram}

    with open(ARQ_RESULTADO, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nsalvo em {ARQ_RESULTADO}")


if __name__ == "__main__":
    if "--sortear" in sys.argv:
        sortear()
    elif "--planilha" in sys.argv:
        planilha()
    elif "--apurar" in sys.argv:
        apurar()
    else:
        print(__doc__)
