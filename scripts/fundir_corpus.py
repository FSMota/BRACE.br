# -*- coding: utf-8 -*-
"""Funde os lados FALSO e VERDADEIRO num corpus único, com esquema comum.

    python scripts/fundir_corpus.py              # simula, não grava
    python scripts/fundir_corpus.py --aplicar    # grava datasets/corpus_fundido.jsonl

Três decisões:

1. NADA É DESCARTADO, TUDO É MARCADO. Entram todos os registros dos dois
   lados; duplicatas ficam com `duplicata_canonica = False` e o subconjunto
   balanceado com `subconjunto_pareado = True`. O experimento filtra. Qualquer
   decisão (limiar de duplicata, critério de pareamento) pode ser revista sem
   refazer a fusão -- e dá para relatar quanto cada filtro muda o resultado.

2. `texto_entrada` É A ENTRADA CRUA DA RQ1, e só ela. Falso: `texto_viral_bruto`.
   Verdadeiro: título + lide, unidos por um espaço. Nada de separador especial
   ("\\n", " | ", ". "): ele só existiria no lado verdadeiro e viraria atalho.
   O recorte título+lide é o que neutraliza o comprimento (baseline de
   comprimento cai de 83,5% com o corpo inteiro para 47,7%).

3. PAREAMENTO POR (tema_pareamento, ano, split). Por ano porque a época não pode
   separar as classes; por split porque os cortes temporais caem no meio do
   ano, e parear só por ano deixaria o teste desequilibrado mesmo com o total
   em 50/50. Em cada célula entram todos os canônicos do lado menor e um
   sorteio reproduzível do lado maior.

CAMPOS QUE NUNCA SÃO ENTRADA DE TREINO (ver README.md):
`texto_bruto_jornalista`, `titulo_artigo_checagem`, `justificativa*` -- só
existem no lado falso e o denunciam. `conteudo_real` completo -- o comprimento
sozinho separa as classes. `ideia_central` das falsas ainda é da versão com
prompt assimétrico (`refinamento_versao`), e não pode ser comparado com nada
até o refinamento cego dos dois lados.
"""
import os
import re
import sys
import json
import random
import hashlib
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_FALSAS = os.path.join(_RAIZ, "datasets", "dataset_tcc_fake_news_final.json")
ARQ_VERDADEIRAS = os.path.join(_RAIZ, "datasets", "dataset_verdadeiras.jsonl")
ARQ_SAIDA = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
SEED = 0
SPLITS = ("treino", "validacao", "teste")

# --- limpeza de `texto_entrada` do lado FALSO --------------------------------
# O `texto_viral_bruto` fica INTACTO; só a entrada do experimento é limpa. As
# regras (introdução do checador, "Versão N" no meio, lixo) vivem em
# limpeza_viral.py -- o MESMO módulo que o pipeline de extração usa na origem,
# para as duas pontas não divergirem. Aqui elas cobrem os registros que não
# passaram pela extração nova; nos reprocessados, já não há o que tirar.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limpeza_viral import limpar_viral, degenerada, narrativa_checador, cabecalho_checador   # noqa: E402

# Veredito dentro da entrada: o vazamento que a auditoria mede. Vira
#    `entrada_com_veredito`, mas NÃO sai do pareado. Usa a MESMA lista da
#    auditoria (ATRIBUTIVOS), e não palavras soltas -- uma lista de palavras
#    marcou 12,7% das falsas, quase tudo viral legítimo. Mesmo assim a regra
#    não separa quem escreveu: "É falso que programas..." é o checador, mas
#    "FAKE NEWS É DESMENTIDA Circulou nas redes..." é o próprio boato. Excluir
#    tiraria ~11% das falsas, muitas legítimas, e o corte enviesaria a base.
#    Decisão (a mesma do documento de passagem): declarar e MEDIR -- o
#    experimento roda com e sem estes registros, e a diferença é o quanto o
#    vazamento explica da acurácia (RQ4).
# --- normalização da assinatura da fonte, NAS DUAS CLASSES -------------------
# O modelo bruto pôs "hoje" e "agência brasil" entre os termos de maior peso
# para VERDADEIRO. "hoje (12)" é o estilo da casa (15,7% das verdadeiras contra
# 0,1% das falsas); a menção "à Agência Brasil" é o veículo se nomeando. Os dois
# dizem QUEM escreveu, não se é verdade. Tira-se o "(N)" e mantém-se a palavra;
# tira-se a menção com a preposição que a introduz. Vale para as duas classes
# (144 falsas também têm "hoje (N)"): regra de um lado só seria assimetria.
# Maiúsculas e pontuação NÃO são tocadas: caixa alta e "!!!" são estilo do
# boato, e apagá-los perderia informação legítima.
DIA_PAREN = re.compile(
    r"\b(hoje|ontem|amanhã|(?:segunda|terça|quarta|quinta|sexta)(?:-feira)?|sábado|domingo)"
    # "(12)", "(1º)", "( 8)", "(15/06/2021)"
    r"\s*\(\s*\d{1,2}\s*[º°]?(?:/\d{1,2}(?:/\d{2,4})?)?\s*\)", re.I)
# \b depois de "Brasil" impede casar "Agência Brasileira de Cooperação".
# Cada padrão consome o próprio espaço, para não precisar de limpeza global de
# espaçamento -- que mexeria na pontuação do texto inteiro ("URGENTE !!!").
# "Em entrevista à Agência Brasil, X" -> "Em entrevista, X";
# "Agência Brasil explica como..." -> "explica como...".
MENCAO_FONTE = re.compile(
    r"\s*\b(?:à|pela|da|para a|com a|a)\s+ag[êe]ncia brasil\b|\bag[êe]ncia brasil\b\s*", re.I)


def normalizar_fonte(texto):
    feitas = []
    novo = DIA_PAREN.sub(r"\1", texto)
    if novo != texto:
        feitas.append("dia_entre_parenteses")
    texto = novo
    novo = MENCAO_FONTE.sub("", texto)
    if novo != texto:
        feitas.append("mencao_agencia_brasil")
    return novo.strip(), feitas


from auditar_corpus import ATRIBUTIVOS                             # noqa: E402
VEREDITO = re.compile(ATRIBUTIVOS.pattern + r"|#boato\b|#fake\b|#fakenews\b", re.I)


def mil(n):
    return "{:,}".format(n).replace(",", ".")


def ler_jsonl(caminho):
    with open(caminho, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def identificador(prefixo, *partes):
    """Estável entre execuções e independente da ordem do arquivo. URL + viral
    não basta: em artigo-lista o LLM às vezes repete o MESMO viral para
    alegações diferentes (104 grupos medidos), então a alegação entra na chave.
    As colisões que sobram (registros idênticos) recebem sufixo em `montar`."""
    return prefixo + hashlib.sha1("|".join(partes).encode("utf-8")).hexdigest()[:12]


def normalizar(texto):
    return re.sub(r"\s+", " ", (texto or "").lower()).strip()


def limpar_entrada_falsa(texto):
    """Introdução do checador fora e só a primeira versão (limpeza_viral).
    Devolve o texto e os prefixos removidos, que ficam no registro."""
    r = limpar_viral(texto)
    return r["texto"], r["prefixos"]


def do_falso(r):
    entrada, prefixos = limpar_entrada_falsa(r["texto_viral_bruto"])
    comum = {
        "id": identificador("F", r["url_origem_fake"], r["texto_viral_bruto"],
                            r.get("claim_refinada") or ""),
        "veredito": "FALSO",
        "texto_entrada": entrada,
        "entrada_prefixos_removidos": prefixos,
        "fonte": r["fonte_fake"],
        "url": r["url_origem_fake"],
        "data_publicacao": r.get("data_publicacao"),
        "tema_pareamento": r["tema_pareamento"],
        "tema_pareamento_origem": r["tema_pareamento_origem"],
        "split": r["split"],
        "grupo_duplicata": r.get("grupo_duplicata"),
        "duplicata_canonica": r.get("duplicata_canonica", True),
        # O ideia_central atual foi gerado com "resuma a afirmação FALSA": o
        # modelo sabia o rótulo. Marcado para ninguém usá-lo no braço refinado.
        "refinamento_versao": "v1_prompt_assimetrico",
    }
    return comum, {k: v for k, v in r.items() if k not in comum}


def do_verdadeiro(r):
    comum = {
        "id": identificador("V", r["url_origem_real"], r["titulo"]),
        "veredito": "VERDADEIRO",
        "texto_entrada": f"{r['titulo'].strip()} {r['lide'].strip()}",
        "entrada_prefixos_removidos": [],
        "fonte": r["fonte_real"],
        "url": r["url_origem_real"],
        "data_publicacao": r["data_publicacao"],
        "tema_pareamento": r["tema_pareamento"],
        "tema_pareamento_origem": r["tema_pareamento_origem"],
        "split": r["split"],
        "grupo_duplicata": r.get("grupo_duplicata"),
        "duplicata_canonica": r.get("duplicata_canonica", True),
        "refinamento_versao": None,
    }
    return comum, {k: v for k, v in r.items() if k not in comum}


def montar(falsas, verdadeiras):
    """Registro = campos comuns na frente + os campos originais de cada lado,
    que ficam intactos para rastreabilidade (inclusive os proibidos como
    entrada -- o datasheet diz quais são)."""
    saida, vistos = [], Counter()
    for fonte, conv in ((falsas, do_falso), (verdadeiras, do_verdadeiro)):
        for r in fonte:
            comum, resto = conv(r)
            vistos[comum["id"]] += 1
            if vistos[comum["id"]] > 1:                 # registro idêntico a outro
                comum["id"] += f"-{vistos[comum['id']]}"
            comum["texto_entrada"], comum["entrada_normalizacoes"] = normalizar_fonte(comum["texto_entrada"])
            comum["entrada_degenerada"] = degenerada(comum["texto_entrada"])
            # Narração do checador ("diz a mensagem", "de acordo com o texto"): não é
            # o boato, e carrega o vocabulário de quem desmente. Sai do pareado, como a
            # entrada degenerada. Vale nas duas classes (notícia sobre boato também casa).
            comum["entrada_narrativa_checador"] = narrativa_checador(comum["texto_entrada"])
            # Título da checagem com "#boato", assinatura do autor, rodapé do site:
            # é o rótulo dentro da entrada. Sai do pareado (05/10/2026).
            comum["entrada_cabecalho_checador"] = cabecalho_checador(comum["texto_entrada"])
            # Vale para as duas classes: notícia que cita "fake news" também
            # sai, para o filtro não ser assimétrico.
            comum["entrada_com_veredito"] = bool(VEREDITO.search(comum["texto_entrada"]))
            comum["entrada_repetida"] = False
            comum["subconjunto_pareado"] = False
            comum.update(resto)
            saida.append(comum)

    # Mesma entrada em dois canônicos da mesma classe: o MinHash não os pega
    # quando o texto é curto demais para ter shingles de 5 palavras. Fica o
    # primeiro por id; os outros saem do pareado.
    primeiro = {}
    for r in sorted(saida, key=lambda x: x["id"]):
        if not r["duplicata_canonica"] or r["entrada_degenerada"]:
            continue
        chave = (r["veredito"], normalizar(r["texto_entrada"]))
        if chave in primeiro:
            r["entrada_repetida"] = True
        else:
            primeiro[chave] = r["id"]
    return saida


def elegivel(r):
    # `entrada_com_veredito` fica DE FORA deste filtro de propósito (ver item 3).
    return (r["duplicata_canonica"] and r["data_publicacao"]
            and not r["entrada_degenerada"] and not r["entrada_repetida"]
            and not r["entrada_narrativa_checador"] and not r["entrada_cabecalho_checador"])


def parear(corpus):
    celulas = defaultdict(lambda: {"FALSO": [], "VERDADEIRO": []})
    for r in corpus:
        if elegivel(r):
            chave = (r["tema_pareamento"], r["data_publicacao"][:4], r["split"])
            celulas[chave][r["veredito"]].append(r)
    sobra = Counter()
    for chave, lados in sorted(celulas.items()):
        n = min(len(lados["FALSO"]), len(lados["VERDADEIRO"]))
        for classe, regs in lados.items():
            regs = sorted(regs, key=lambda x: x["id"])          # ordem estável
            random.Random(f"{SEED}|{'|'.join(chave)}|{classe}").shuffle(regs)
            for r in regs[:n]:
                r["subconjunto_pareado"] = True
            sobra[(chave[0], classe)] += len(regs) - n
    return sobra


def main():
    aplicar = "--aplicar" in sys.argv
    print("=" * 72)
    print("  FUSÃO DO CORPUS  " + ("[APLICANDO]" if aplicar else "[DRY-RUN — nada será gravado]"))
    print("=" * 72)

    with open(ARQ_FALSAS, encoding="utf-8") as f:
        falsas = json.load(f)
    verdadeiras = ler_jsonl(ARQ_VERDADEIRAS)
    print(f"\nentrada: {mil(len(falsas))} falsas + {mil(len(verdadeiras))} verdadeiras")

    # Pré-condições: a fusão não corrige nada, só junta. Se algo faltar, é
    # sinal de que um passo anterior não rodou -- melhor parar do que fundir
    # uma base pela metade.
    faltas = Counter()
    for r in falsas:
        for k in ("split", "tema_pareamento", "texto_viral_bruto"):
            faltas[f"falsa sem {k}"] += not r.get(k)
    for r in verdadeiras:
        for k in ("split", "tema_pareamento", "titulo", "lide"):
            faltas[f"verdadeira sem {k}"] += not r.get(k)
    faltas = {k: v for k, v in faltas.items() if v}
    if faltas:
        print(f"\n[!] pré-condição falhou: {faltas}")
        print("    rode corrigir_corpus.py / preparar_verdadeiras.py antes.")
        sys.exit(1)

    corpus = montar(falsas, verdadeiras)
    ids = Counter(r["id"] for r in corpus)
    repetidos = {k: n for k, n in ids.items() if n > 1}
    print(f"ids únicos: {mil(len(ids))} de {mil(len(corpus))}"
          + (f"  [!] {len(repetidos)} repetidos" if repetidos else ""))

    sobra = parear(corpus)
    for tipo in ("dia_entre_parenteses", "mencao_agencia_brasil"):
        c = Counter(r["veredito"] for r in corpus if tipo in r["entrada_normalizacoes"])
        print(f"normalização {tipo}: falsas {c['FALSO']} · verdadeiras {c['VERDADEIRO']}")
    amostra = [r["texto_entrada"] for r in corpus
               if "mencao_agencia_brasil" in r["entrada_normalizacoes"] and r["veredito"] == "VERDADEIRO"][:3]
    for t in amostra:
        print(f"    ex.: {t[:110]!r}")
    limpos = [r for r in corpus if r["entrada_prefixos_removidos"]]
    print(f"introdução do checador removida: {mil(len(limpos))} falsas")
    print("    prefixos:", Counter(re.sub(r"\d+", "N", p.lower())[:30]
                                  for r in limpos for p in r["entrada_prefixos_removidos"]).most_common(6))
    for rotulo, campo in (("entrada degenerada (lixo de extração)", "entrada_degenerada"),
                          ("narração do checador (sai do pareado)", "entrada_narrativa_checador"),
                          ("cabeçalho do checador, #boato (sai do pareado)", "entrada_cabecalho_checador"),
                          ("entrada com veredito (marcada, fica no pareado)", "entrada_com_veredito"),
                          ("entrada repetida entre canônicos", "entrada_repetida")):
        c = Counter(r["veredito"] for r in corpus if r[campo])
        print(f"{rotulo}: falsas {c['FALSO']} · verdadeiras {c['VERDADEIRO']}")

    # ---- relatório ----
    def tabela(filtro, titulo):
        print(f"\n{titulo}")
        print(f"    {'':10s} {'falsas':>8s} {'verdadeiras':>12s}   % verdadeiras")
        for s in SPLITS:
            c = Counter(r["veredito"] for r in corpus if r["split"] == s and filtro(r))
            tot = c["FALSO"] + c["VERDADEIRO"]
            print(f"    {s:10s} {mil(c['FALSO']):>8s} {mil(c['VERDADEIRO']):>12s}"
                  f"   {100 * c['VERDADEIRO'] / max(1, tot):5.1f}%")

    tabela(lambda r: True, "[1] corpus completo (inclui duplicatas marcadas)")
    tabela(elegivel, "[2] elegíveis (canônicos, com data, entrada válida e não repetida)")
    tabela(lambda r: r["subconjunto_pareado"] and r["entrada_com_veredito"],
           "[2b] dentro do pareado, marcados com veredito (rodar o experimento sem eles também)")
    tabela(lambda r: r["subconjunto_pareado"], "[3] subconjunto pareado (tema x ano x split)")

    print("\n    o que fica FORA do pareado, por tema (canônicos sem par):")
    temas = sorted({t for t, _ in sobra}, key=lambda t: -(sobra[(t, "FALSO")] + sobra[(t, "VERDADEIRO")]))
    for t in temas:
        f, v = sobra[(t, "FALSO")], sobra[(t, "VERDADEIRO")]
        if f or v:
            print(f"      {t:22s} falsas {mil(f):>6}   verdadeiras {mil(v):>6}")

    import statistics as st
    print("\n[4] texto_entrada no pareado -- mediana de caracteres:")
    for classe in ("FALSO", "VERDADEIRO"):
        ls = [len(r["texto_entrada"]) for r in corpus if r["subconjunto_pareado"] and r["veredito"] == classe]
        print(f"    {classe:10s} {int(st.median(ls)):>5}   (n={mil(len(ls))})")

    # Mesmo texto nas duas classes seria rótulo contraditório.
    por_texto = defaultdict(set)
    for r in corpus:
        por_texto[r["texto_entrada"].lower().strip()].add(r["veredito"])
    conflito = sum(1 for v in por_texto.values() if len(v) > 1)
    print(f"\n[5] textos idênticos com vereditos opostos: {conflito}")

    if not aplicar:
        print("\n" + "-" * 72)
        print("DRY-RUN: nada foi gravado. Rode com --aplicar para efetivar.")
        return

    tmp = ARQ_SAIDA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in corpus:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, ARQ_SAIDA)
    print(f"\nGravado: {ARQ_SAIDA}")
    print(f"{mil(len(corpus))} registros; {mil(sum(r['subconjunto_pareado'] for r in corpus))} no pareado.")


if __name__ == "__main__":
    main()
