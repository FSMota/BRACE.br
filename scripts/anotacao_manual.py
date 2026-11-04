# -*- coding: utf-8 -*-
"""Anotação manual por amostra: quanto das entradas FALSAS não é o boato?

    python scripts/anotacao_manual.py --sortear 200      # depois de remontar o corpus
    python scripts/anotacao_manual.py --registros         # acrescenta ao meta quantos registros o artigo gerou
    python scripts/anotacao_manual.py --atualizar-categorias   # só com a planilha em branco
    python scripts/anotacao_manual.py --apurar            # depois de preencher a planilha

Por que existe: toda contagem de defeito até aqui veio de REGRA de texto, e
regra só acha o que quem a escreveu previu. Os pilotos do refinamento mostraram
casos que nenhuma regra pegava. A taxa real -- com intervalo de confiança -- só
sai de leitura humana de uma amostra aleatória. É também a verificação manual
com concordância entre anotadores que um corpus publicado precisa declarar.

DESENHO
  - População: as entradas FALSAS do subconjunto pareado -- exatamente o que
    entra no experimento.
  - Sorteio estratificado por via de extração (html / texto / llm), com
    alocação proporcional: a taxa geral continua sem viés, e as vias podem ser
    comparadas (RQ2). Semente fixa.
  - ANOTAÇÃO ÀS CEGAS: a planilha mostra só o texto e o link do artigo. Via de
    extração, marcações automáticas e versão ficam num arquivo à parte -- quem
    vê que a regra já marcou "checador" tende a concordar com ela.
  - Dois anotadores (opcional): colunas para um segundo anotador, que deve
    anotar SEM ver as respostas do primeiro. Com as duas preenchidas, a
    apuração calcula o kappa de Cohen.

Arquivos (relatorios/anotacao/):
    planilha_anotacao.xlsx   o que o anotador vê e preenche
    amostra_meta.json        o que fica escondido: estrato, flags, versão, hash do corpus
    resultado_anotacao.json  saída do --apurar
"""
import os
import re
import sys
import json
import math
import random
import hashlib
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_FALSAS = os.path.join(_RAIZ, "datasets", "dataset_tcc_fake_news_final.json")
PASTA = os.path.join(_RAIZ, "relatorios", "anotacao")
ARQ_PLANILHA = os.path.join(PASTA, "planilha_anotacao.xlsx")
ARQ_META = os.path.join(PASTA, "amostra_meta.json")
ARQ_RESULTADO = os.path.join(PASTA, "resultado_anotacao.json")
SEED = 20261005

CATEGORIAS = ["boato", "misto", "checador", "desmentido", "lixo", "duvida"]
# "desmentido" entrou em 25/10/2026, antes de qualquer anotação (desvio declarado
# em PREREGISTRO_DESMENTIDO.md): a nota oficial ou a fala do citado que desmente
# o boato não é texto do jornalista, e não cabia em nenhuma categoria.
NAO_BOATO = ("misto", "checador", "desmentido", "lixo")
DEFINICOES = [
    ("boato", "É a mensagem que circulou: a corrente, o post, a legenda do vídeo, com o estilo "
              "dela (caixa alta, erros, emojis). Conta como boato MESMO que use as palavras "
              "\"boato\", \"mentira\" ou \"fake news\" — é comum o próprio boato dizer "
              "\"isso não é boato\" ou \"a Globo diz que é mentira\".",
     "URGENTE!!! Isso não é boato, o governo vai confiscar a poupança amanhã, compartilhem"),
    ("misto", "Tem a mensagem viral E texto do checador junto: uma introdução no meio, um "
              "comentário, o título da checagem antes ou depois do boato.",
     "Leia a mensagem que circula online: Djavan foi casado com Maria. Ela grávida, sofreu..."),
    ("checador", "Só texto do jornalista: o título da checagem (geralmente com \"#boato\"), a "
                 "narração sobre a mensagem (\"diz a mensagem\", \"de acordo com o texto\"), a "
                 "explicação ou o veredito. A mensagem viral não aparece.",
     "De acordo com o texto, ele teria dito que colocaria advogados para defender Lázaro."),
    ("desmentido", "O texto DESMENTE o boato e não é do jornalista: a nota de uma empresa ou órgão, "
                   "o comunicado oficial, a fala ou o post de quem foi citado negando o fato ou a "
                   "autoria. É o lado verdadeiro da história, citado pelo checador como evidência.",
     "A CET informa que não haverá ampliação da área do Rodízio Municipal de Veículos."),
    ("lixo", "Nem boato nem checador: cabeçalho de embed (\"Ver essa foto no Instagram\"), link, "
             "crédito de imagem, menu do site, texto curto demais para ter sentido.",
     "Visualizar esta foto no Instagram. Uma publicação compartilhada por Chad Knight (@chadknight)"),
    ("duvida", "Não dá para decidir só pelo texto, nem abrindo o link. Use pouco e explique na "
               "observação.", ""),
]


def mil(n):
    return "{:,}".format(n).replace(",", ".")


def hash_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()[:16]


# ==========================================================================
# --sortear
# ==========================================================================
def sortear(n):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation

    if os.path.exists(ARQ_PLANILHA) and "--sobrescrever" not in sys.argv:
        sys.exit(f"[!] {ARQ_PLANILHA} já existe -- pode ter anotações. Use --sobrescrever "
                 f"só se tiver certeza.")
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    pop = [r for r in corpus if r["subconjunto_pareado"] and r["veredito"] == "FALSO"]
    estratos = defaultdict(list)
    for r in pop:
        estratos[str(r.get("origem_viral", "?")).replace("_sem_indice", "")].append(r)

    # Alocação proporcional, com arredondamento que soma exatamente n.
    quotas = {e: n * len(v) / len(pop) for e, v in estratos.items()}
    alocado = {e: int(q) for e, q in quotas.items()}
    for e in sorted(quotas, key=lambda e: quotas[e] - alocado[e], reverse=True)[:n - sum(alocado.values())]:
        alocado[e] += 1
    rng = random.Random(SEED)
    amostra = []
    for e in sorted(estratos):
        regs = sorted(estratos[e], key=lambda r: r["id"])
        amostra += [(e, r) for r in rng.sample(regs, alocado[e])]
    rng.shuffle(amostra)           # a ordem não revela o estrato

    os.makedirs(PASTA, exist_ok=True)
    wb = Workbook()
    ins = wb.active
    ins.title = "instrucoes"
    ins["A1"] = "Anotação: cada texto é a mensagem viral (o boato) ou outra coisa?"
    ins["A1"].font = Font(bold=True, size=13)
    ins["A3"] = ("Leia o texto da coluna C e escolha UMA categoria na coluna E (lista suspensa). "
                 "Se precisar, abra o link do artigo de checagem (coluna D) para ver onde o texto "
                 "estava. Não se preocupe se o boato é verdadeiro ou falso — todos estes vieram de "
                 "checagens. A pergunta é só: este texto É a mensagem que circulou?")
    ins["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ins.row_dimensions[3].height = 60
    ins["A5"], ins["B5"], ins["C5"] = "categoria", "quando usar", "exemplo"
    for c in ("A5", "B5", "C5"):
        ins[c].font = Font(bold=True)
    for i, (cat, quando, ex) in enumerate(DEFINICOES, start=6):
        ins[f"A{i}"], ins[f"B{i}"], ins[f"C{i}"] = cat, quando, ex
        for c in "ABC":
            ins[f"{c}{i}"].alignment = Alignment(wrap_text=True, vertical="top")
        ins.row_dimensions[i].height = 75
    ins[f"A{6 + len(DEFINICOES) + 1}"] = ("Segundo anotador: preencha as colunas G e H SEM olhar as "
                                          "colunas E e F (oculte-as antes de começar).")
    ins.column_dimensions["A"].width = 12
    ins.column_dimensions["B"].width = 80
    ins.column_dimensions["C"].width = 60

    ws = wb.create_sheet("anotacao")
    cab = ["n", "id", "texto", "link do artigo", "anotador_1", "observacao_1", "anotador_2", "observacao_2"]
    ws.append(cab)
    for c in ws[1]:
        c.font = Font(bold=True)
        c.fill = PatternFill("solid", fgColor="DDDDDD")
    for i, (_, r) in enumerate(amostra, start=1):
        ws.append([i, r["id"], r["texto_entrada"], r["url"], None, None, None, None])
    for linha in ws.iter_rows(min_row=2):
        linha[2].alignment = Alignment(wrap_text=True, vertical="top")
        for c in linha:
            if c.column != 3:
                c.alignment = Alignment(vertical="top")
    for col, larg in zip("ABCDEFGH", (5, 16, 90, 30, 13, 30, 13, 30)):
        ws.column_dimensions[col].width = larg
    ws.freeze_panes = "C2"
    lista = DataValidation(type="list", formula1='"' + ",".join(CATEGORIAS) + '"', allow_blank=True,
                           showErrorMessage=True, errorTitle="Categoria inválida",
                           error="Use uma das opções da lista: " + ", ".join(CATEGORIAS))
    ws.add_data_validation(lista)
    lista.add(f"E2:E{len(amostra) + 1}")
    lista.add(f"G2:G{len(amostra) + 1}")
    wb.save(ARQ_PLANILHA)

    meta = {"semente": SEED, "corpus_sha": hash_arquivo(ARQ_CORPUS), "n": len(amostra),
            "populacao": len(pop), "estratos_populacao": {e: len(v) for e, v in estratos.items()},
            "estratos_amostra": alocado,
            "itens": {r["id"]: {"estrato": e, "extracao": r.get("extracao_versao", "v1"),
                                "split": r["split"], "tema": r["tema_pareamento"],
                                "flag_veredito": r["entrada_com_veredito"],
                                "flag_narrativa": r.get("entrada_narrativa_checador"),
                                "flag_cabecalho": r.get("entrada_cabecalho_checador")}
                      for e, r in amostra}}
    with open(ARQ_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"população: {mil(len(pop))} falsas do pareado | amostra: {len(amostra)}")
    print("  por via de extração (população -> amostra):",
          {e: f"{len(estratos[e])} -> {alocado[e]}" for e in sorted(estratos)})
    print(f"  corpus sha {meta['corpus_sha']} | semente {SEED}")
    print(f"planilha: {ARQ_PLANILHA}")


# ==========================================================================
# --atualizar-categorias
# ==========================================================================
def atualizar_categorias():
    """Reescreve as instruções e a lista suspensa da planilha com as CATEGORIAS atuais.

    Só roda com a planilha em branco: mudar as categorias depois de começar a
    anotar mudaria o instrumento no meio da medida. Os textos, ids e links da
    aba de anotação não mudam.
    """
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    wb = load_workbook(ARQ_PLANILHA)
    ws = wb["anotacao"]
    preenchidos = sum(1 for r in ws.iter_rows(min_row=2, values_only=True) if r[4] or r[6])
    if preenchidos:
        sys.exit(f"[!] {preenchidos} linhas já anotadas -- categorias não mudam no meio da anotação.")
    ins = wb["instrucoes"]
    for linha in range(6, ins.max_row + 1):
        for c in "ABC":
            ins[f"{c}{linha}"] = None
    for i, (cat, quando, ex) in enumerate(DEFINICOES, start=6):
        ins[f"A{i}"], ins[f"B{i}"], ins[f"C{i}"] = cat, quando, ex
        for c in "ABC":
            ins[f"{c}{i}"].alignment = Alignment(wrap_text=True, vertical="top")
            ins[f"{c}{i}"].font = Font(bold=False)
        ins.row_dimensions[i].height = 75
    ins[f"A{6 + len(DEFINICOES) + 1}"] = ("Segundo anotador: preencha as colunas G e H SEM olhar as "
                                          "colunas E e F (oculte-as antes de começar).")
    ws.data_validations.dataValidation = []
    lista = DataValidation(type="list", formula1='"' + ",".join(CATEGORIAS) + '"', allow_blank=True,
                           showErrorMessage=True, errorTitle="Categoria inválida",
                           error="Use uma das opções da lista: " + ", ".join(CATEGORIAS))
    ws.add_data_validation(lista)
    lista.add(f"E2:E{ws.max_row}")
    lista.add(f"G2:G{ws.max_row}")
    wb.save(ARQ_PLANILHA)
    print(f"planilha atualizada: categorias {CATEGORIAS} ({ws.max_row - 1} linhas, nenhuma anotada)")


# ==========================================================================
# --registros
# ==========================================================================
def registros():
    """Acrescenta ao meta quantos registros o artigo de cada item gerou.

    Um artigo vira um registro por bloco de mensagem, e os artigos com vários
    blocos às vezes trazem o desmentido (comunicado oficial, fala do citado)
    num deles. O número fica só no meta: a planilha não muda, e o anotador
    continua sem vê-lo. O link vem da própria planilha (coluna D), e a contagem,
    do lado falso inteiro, antes do pareamento.
    """
    from openpyxl import load_workbook
    meta = json.load(open(ARQ_META, encoding="utf-8"))
    ws = load_workbook(ARQ_PLANILHA, read_only=True)["anotacao"]
    url_por_id = {r[1]: r[3] for r in ws.iter_rows(min_row=2, values_only=True)}
    por_url = Counter(r["url_origem_fake"] for r in json.load(open(ARQ_FALSAS, encoding="utf-8")))
    sem = [i for i in meta["itens"] if por_url.get(url_por_id.get(i), 0) == 0]
    if sem:
        sys.exit(f"[!] {len(sem)} itens sem artigo no lado falso (ex.: {sem[:3]}) -- nada gravado.")
    for i, item in meta["itens"].items():
        item["registros_no_artigo"] = por_url[url_por_id[i]]
    meta["registros_no_artigo_fonte"] = {"arquivo": os.path.basename(ARQ_FALSAS),
                                         "sha": hash_arquivo(ARQ_FALSAS)}
    with open(ARQ_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    dist = Counter(item["registros_no_artigo"] for item in meta["itens"].values())
    print(f"registros por artigo na amostra: {dict(sorted(dist.items()))}")
    print(f"  artigo com 1 registro: {dist[1]} | com 2 ou mais: {sum(v for k, v in dist.items() if k > 1)}")
    print(f"meta: {ARQ_META} (planilha não alterada)")


# ==========================================================================
# --apurar
# ==========================================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margem = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - margem), min(1.0, centro + margem))


def kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def apurar():
    from openpyxl import load_workbook
    meta = json.load(open(ARQ_META, encoding="utf-8"))
    if hash_arquivo(ARQ_CORPUS) != meta["corpus_sha"]:
        print("[!] o corpus mudou desde o sorteio -- as flags e estratos do arquivo meta "
              "continuam valendo, mas a amostra não representa mais o corpus atual.")
    ws = load_workbook(ARQ_PLANILHA)["anotacao"]
    linhas = [dict(zip([c.value for c in ws[1]], [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
    an1 = [(l["id"], (l["anotador_1"] or "").strip().lower()) for l in linhas]
    feitos = [(i, c) for i, c in an1 if c in CATEGORIAS]
    print(f"anotados pelo anotador 1: {len(feitos)} de {len(an1)}")
    if not feitos:
        return
    n = len(feitos)
    cont = Counter(c for _, c in feitos)
    print("\ncategoria      n      %      IC95%")
    resultado = {"n": n, "categorias": {}}
    for c in CATEGORIAS:
        lo, hi = wilson(cont[c], n)
        resultado["categorias"][c] = {"n": cont[c], "pct": round(100 * cont[c] / n, 1),
                                      "ic95": [round(100 * lo, 1), round(100 * hi, 1)]}
        print(f"  {c:10s} {cont[c]:4d}  {100 * cont[c] / n:5.1f}%  [{100 * lo:4.1f}; {100 * hi:4.1f}]")
    nao_boato = sum(cont[c] for c in NAO_BOATO)
    lo, hi = wilson(nao_boato, n)
    print(f"\n  NÃO É SÓ O BOATO (misto + checador + desmentido + lixo): {nao_boato} = {100 * nao_boato / n:.1f}% "
          f"[{100 * lo:.1f}; {100 * hi:.1f}]  -> projeção na população: "
          f"~{mil(round(meta['populacao'] * nao_boato / n))} de {mil(meta['populacao'])}")
    resultado["nao_boato"] = {"n": nao_boato, "pct": round(100 * nao_boato / n, 1),
                              "ic95": [round(100 * lo, 1), round(100 * hi, 1)]}

    print("\npor via de extração (% que NÃO é só o boato):")
    por = defaultdict(list)
    for i, c in feitos:
        por[meta["itens"][i]["estrato"]].append(c)
    resultado["por_via"] = {}
    for e, cs in sorted(por.items()):
        k = sum(c in NAO_BOATO for c in cs)
        lo, hi = wilson(k, len(cs))
        resultado["por_via"][e] = {"n": len(cs), "pct_nao_boato": round(100 * k / len(cs), 1)}
        print(f"  {e:8s} n={len(cs):3d}  {100 * k / len(cs):5.1f}%  [{100 * lo:4.1f}; {100 * hi:4.1f}]")

    if all("registros_no_artigo" in meta["itens"][i] for i, _ in feitos):
        print("\npor registros do artigo (% que NÃO é só o boato):")
        por = defaultdict(list)
        for i, c in feitos:
            por["1" if meta["itens"][i]["registros_no_artigo"] == 1 else "2+"].append(c)
        resultado["por_registros_no_artigo"] = {}
        for e, cs in sorted(por.items()):
            k = sum(c in NAO_BOATO for c in cs)
            lo, hi = wilson(k, len(cs))
            resultado["por_registros_no_artigo"][e] = {"n": len(cs), "pct_nao_boato": round(100 * k / len(cs), 1),
                                                       "ic95": [round(100 * lo, 1), round(100 * hi, 1)]}
            print(f"  {e:8s} n={len(cs):3d}  {100 * k / len(cs):5.1f}%  [{100 * lo:4.1f}; {100 * hi:4.1f}]")

    # As regras automáticas acertam? (precisão e cobertura contra a leitura humana)
    print("\nregras automáticas contra a leitura humana (verdade = NÃO é só o boato):")
    verdade = {i: c in NAO_BOATO for i, c in feitos}
    for flag in ("flag_veredito", "flag_narrativa", "flag_cabecalho", "flag_desmentido"):
        pred = {i: bool(meta["itens"][i].get(flag)) for i, _ in feitos}
        vp = sum(pred[i] and verdade[i] for i in pred)
        fp = sum(pred[i] and not verdade[i] for i in pred)
        fn = sum(not pred[i] and verdade[i] for i in pred)
        prec = vp / (vp + fp) if vp + fp else float("nan")
        cob = vp / (vp + fn) if vp + fn else float("nan")
        print(f"  {flag:16s} marcados {vp + fp:3d} | precisão {prec:.2f} | cobertura {cob:.2f}")

    a2 = {l["id"]: (l["anotador_2"] or "").strip().lower() for l in linhas}
    pares = [(c, a2[i]) for i, c in feitos if a2.get(i) in CATEGORIAS]
    if pares:
        k = kappa([x for x, _ in pares], [y for _, y in pares])
        kb = kappa([x in ("boato",) for x, _ in pares], [y in ("boato",) for _, y in pares])
        print(f"\nconcordância (n={len(pares)}): kappa 5 categorias {k:.2f} | kappa boato × não-boato {kb:.2f}")
        resultado["kappa"] = {"n": len(pares), "cinco_categorias": round(k, 3), "binario": round(kb, 3)}

    resultado["ids_nao_boato"] = sorted(i for i, c in feitos if c in NAO_BOATO)
    with open(ARQ_RESULTADO, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\nsalvo em {ARQ_RESULTADO}")


if __name__ == "__main__":
    if "--sortear" in sys.argv:
        sortear(int(sys.argv[sys.argv.index("--sortear") + 1]))
    elif "--atualizar-categorias" in sys.argv:
        atualizar_categorias()
    elif "--registros" in sys.argv:
        registros()
    elif "--apurar" in sys.argv:
        apurar()
    else:
        print(__doc__)
