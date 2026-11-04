# -*- coding: utf-8 -*-
"""Correção do corpus -- aplica os defeitos A, B, D e os campos derivados.

Roda em UMA passada, com backup e dry-run por padrão. Ver AUDITORIA_CORPUS.md
para a medição que justifica cada etapa.

    python scripts/corrigir_corpus.py              # simula, não grava
    python scripts/corrigir_corpus.py --aplicar    # grava (com backup)

NÃO rode com o main_nlp() em execução: os dois escrevem os mesmos arquivos.
"""
import os
import re
import sys
import json
import random
import shutil
import hashlib
import statistics
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import taxonomia                                        # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DADOS = os.path.join(_RAIZ, "datasets")
ARQ_BRUTO = os.path.join(PASTA_DADOS, "dataset_bruto.json")
ARQ_FINAL = os.path.join(PASTA_DADOS, "dataset_tcc_fake_news_final.json")

# Divisão temporal: treino nos mais antigos, teste nos mais recentes. Divisão
# aleatória infla a acurácia porque o mesmo boato reciclado cai dos dois lados.
FRACAO_TREINO, FRACAO_VAL = 0.70, 0.10
LIMIAR_JACCARD = 0.70
MINHASH_N, BANDAS = 64, 16          # 16 bandas x 4 linhas
SEED = 0

# --- padrões do extrator textual (iguais aos de web_scraping_limpo.py) ------
INICIO_VIRAL = re.compile(
    r"(?:confira|leia|veja)\s+(?:o|a|as|os|algumas)?[^.:;]{0,60}"
    r"(?:texto|mensage|conteúdo|conteudo|transcri|post|publicaç)[^.:;]{0,40}:\s*", re.I)
FIM_VIRAL = re.compile(r"\s(?:Checagem|Conclusão|Análise|Fontes?:)\b")

PALAVRAS_VEREDITO = [
    "é falso", "e falso", "é falsa", "é fake", "trata-se de um boato", "é boato",
    "não é verdade", "nao e verdade", "é mentira", "desmentid", "fake news",
    "não procede", "nao procede", "é #fake", "classificamos como", "veredito",
    "checagem do boatos", "boatos.org classific",
]

SUPORTE = {
    "video": r"\bv[íi]deos?\b|\bgrava[çc][ãa]o\b|\bfilmagem\b|\blive\b",
    "imagem": r"\bfotos?\b|\bimagens?\b|\bprints?\b|\bfotografias?\b|\bmontagem\b|\bcartaz\b",
    "audio": r"\b[áa]udios?\b|\bmensagem\s+de\s+voz\b",
    "texto": r"\bcorrente\b|\bmensagem\b|\btexto\b|\bcarta\b|\blista\b",
}
DATA_ARTIGO = re.compile(r"\b(\d{2})/(\d{2})/(\d{4})\b")


def filtrar_veredito(texto):
    if not texto:
        return texto
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", texto)
                    if not any(p in s.lower() for p in PALAVRAS_VEREDITO)).strip()


def shingles(texto, k=5):
    p = re.findall(r"\w+", (texto or "").lower())
    return {" ".join(p[i:i + k]) for i in range(max(0, len(p) - k + 1))}


def mil(n):
    """Separador de milhar brasileiro. Como função, e não .replace na frase
    inteira, porque isso também trocaria as vírgulas do texto."""
    return "{:,}".format(n).replace(",", ".")


# ==========================================================================
# A -- devolver ao LLM os virais sem âncora de fim
# ==========================================================================
def etapa_A(bruto, final):
    """O extrator textual abre em "Leia a mensagem:" e fecha em
    Checagem|Conclusão|Análise|Fontes:. Quando o fechamento não existe no artigo,
    o recorte corre até o fim e leva a apuração junto -- 95,7% desses trazem
    veredito do checador dentro da entrada de treino, contra 5,6% dos que têm
    fechamento. Medido: 2.716 de 4.070.

    Não dá para consertar por corte léxico: boato e checador usam o mesmo
    vocabulário, e toda âncora testada amputou mensagem legítima (correntes de
    golpe trazem link; boatos escrevem "não é verdade" e "fake news"). Então
    devolvemos esses artigos ao caminho LLM, que já entrega mediana de 247c com
    3,1% de contaminação porque decide pelo sentido, não por palavra-chave."""
    alvos = set()
    for item in bruto:
        if item.get("viral_metodo") != "texto" or not item.get("texto_viral_estrutural"):
            continue
        artigo = item.get("texto_bruto_jornalista") or ""
        m = INICIO_VIRAL.search(artigo)
        if not m:
            continue
        if not FIM_VIRAL.search(artigo[m.end():]):
            alvos.add(item.get("url_origem"))

    removidos = [r for r in final if r.get("url_origem_fake") in alvos]
    mantidos = [r for r in final if r.get("url_origem_fake") not in alvos]
    return alvos, removidos, mantidos


def aplicar_A(bruto, alvos):
    """Limpa o viral textual e devolve o artigo à fila. Sem apagar o artigo:
    o texto bruto continua salvo, só a decisão de extração é desfeita."""
    for item in bruto:
        if item.get("url_origem") in alvos:
            item.pop("texto_viral_estrutural", None)
            item.pop("viral_metodo", None)
            item["processado_pela_ia"] = False
            item["boatos_extraidos"] = 0
            item["reprocessar_motivo"] = "viral_textual_sem_ancora_de_fim"


# ==========================================================================
# B -- tirar a alegação de dentro da evidência
# ==========================================================================
def etapa_B(registros):
    """`justificativa_bruta` deveria conter só a apuração. No caminho sem
    <blockquote>, `_fechar_justificativa()` não removia o viral porque
    `texto_viral_estrutural` era None naquele momento -- o viral só é conhecido
    depois, quando o LLM o extrai. Resultado: 87% dos registros desse caminho
    carregam a alegação dentro da evidência (sobreposição mediana 1,00), o que
    invalida o campo como padrão-ouro de recuperação na RQ3.2.

    Aqui o viral já é conhecido, então a remoção é trivial e determinística."""
    tocados = 0
    for r in registros:
        jb, viral = r.get("justificativa_bruta"), r.get("texto_viral_bruto")
        if not jb or not viral or len(viral) < 40:
            continue
        novo = jb.replace(viral, " ") if viral in jb else jb

        # Remoção literal falha sempre que o viral extraído pelo LLM difere do
        # artigo por normalização de espaço, aspas curvas ou emoji perdido --
        # 13% dos registros do caminho `llm` sobreviviam assim. Comparar por
        # sobreposição de n-gramas não depende de igualdade caractere a
        # caractere: a sentença que REAPARECE no viral sai, o resto fica.
        alvo = shingles(viral, k=4)
        if alvo:
            manter = []
            for s in re.split(r"(?<=[.!?])\s+", novo):
                si = shingles(s, k=4)
                if si and len(si & alvo) / len(si) > 0.50:
                    continue          # esta sentença é a alegação, não a apuração
                manter.append(s)
            novo = " ".join(manter)

        novo = re.sub(r"\s{2,}", " ", filtrar_veredito(novo) or "").strip()
        if novo != jb:
            tocados += 1
            r["justificativa_bruta"] = novo if len(novo) >= 150 else None
            if not r["justificativa_bruta"]:
                r["justificativa_origem"] = "llm"
    return tocados


# ==========================================================================
# campos derivados: data e suporte
# ==========================================================================
def etapa_campos(registros, por_url):
    """Dois campos que a fonte já informa e ninguém tinha lido.

    `data_publicacao` sai do cabeçalho do artigo ("Política 10/08/2026 por ...")
    e é o que permite dividir treino/teste por tempo em vez de aleatoriamente.

    `suporte` sai da linha-resumo "Boato - <alegação>", escrita pelo jornalista.
    Cinco estados, e `nao_declarado` é o mais honesto dos cinco: em 76% dos
    artigos o jornalista simplesmente não diz se era vídeo, foto ou texto, e
    inventar um rótulo aí seria repetir o erro do tema inflado."""
    com_data = com_suporte = 0
    for r in registros:
        artigo = (por_url.get(r.get("url_origem_fake"), {})
                  .get("texto_bruto_jornalista") or "")

        m = DATA_ARTIGO.search(artigo[:600])
        if m:
            d, mes, a = m.groups()
            try:
                r["data_publicacao"] = datetime(int(a), int(mes), int(d)).strftime("%Y-%m-%d")
                com_data += 1
            except ValueError:
                r["data_publicacao"] = None
        else:
            r["data_publicacao"] = None

        mb = re.search(r"\bBoato\s*[–—-]\s*", artigo)
        if mb:
            resto = artigo[mb.end():]
            p = re.search(r"(?<=[.!?])\s", resto)
            linha = (resto[:p.start()] if p else resto[:300]).lower()
            achados = [k for k, rx in SUPORTE.items() if re.search(rx, linha)]
            r["suporte"] = achados[0] if len(achados) == 1 else (
                "+".join(sorted(achados)) if achados else "nao_declarado")
            r["suporte_origem"] = "linha_resumo"
            if achados:
                com_suporte += 1
        else:
            r["suporte"] = "nao_declarado"
            r["suporte_origem"] = "ausente"

        # Camada grossa, usada SÓ para equilibrar as classes por assunto. O
        # `tema` granular fica intacto: é dado de pesquisa, e é contra ele que
        # se mede a concordância com o LLM (RQ2). Ver taxonomia.py.
        r["tema_pareamento"] = taxonomia.pareamento_do_falso(r.get("tema"))
        r["tema_pareamento_origem"] = ("mapa_categoria_editorial"
                                       if r.get("tema_origem") == "url" else "mapa_tema_llm")
    return com_data, com_suporte


# ==========================================================================
# D -- duplicatas e divisão temporal
# ==========================================================================
def _minhash(conjunto, n=MINHASH_N):
    if not conjunto:
        return None
    sig = []
    for i in range(n):
        sig.append(min(int(hashlib.md5(f"{i}:{s}".encode()).hexdigest()[:8], 16)
                       for s in conjunto))
    return tuple(sig)


def etapa_D(registros):
    """Agrupa quase-duplicatas e divide por tempo.

    Não apaga nada: marca `grupo_duplicata` e `duplicata_canonica`. Apagar seria
    decisão irreversível tomada por um script; marcar deixa a escolha no
    experimento, e permite relatar quanto a deduplicação muda o resultado.

    LSH por bandas em vez de comparar todos contra todos: 15 mil registros dão
    115 milhões de pares, o que não termina em tempo útil."""
    random.seed(SEED)
    sh = {i: shingles(r.get("texto_viral_bruto")) for i, r in enumerate(registros)}
    assinaturas = {i: _minhash(s) for i, s in sh.items()}

    baldes = defaultdict(list)
    linhas = MINHASH_N // BANDAS
    for i, sig in assinaturas.items():
        if not sig:
            continue
        for b in range(BANDAS):
            baldes[(b, sig[b * linhas:(b + 1) * linhas])].append(i)

    # União-busca sobre os pares candidatos confirmados por Jaccard real.
    pai = list(range(len(registros)))

    def raiz(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    for grupo in baldes.values():
        if len(grupo) < 2 or len(grupo) > 200:
            continue
        for a in range(len(grupo)):
            for b in range(a + 1, len(grupo)):
                i, j = grupo[a], grupo[b]
                if raiz(i) == raiz(j):
                    continue
                si, sj = sh[i], sh[j]
                if si and sj and len(si & sj) / len(si | sj) >= LIMIAR_JACCARD:
                    pai[raiz(i)] = raiz(j)

    grupos = defaultdict(list)
    for i in range(len(registros)):
        grupos[raiz(i)].append(i)

    n_grupos = n_excedente = 0
    for membros in grupos.values():
        if len(membros) < 2:
            registros[membros[0]]["grupo_duplicata"] = None
            registros[membros[0]]["duplicata_canonica"] = True
            continue
        n_grupos += 1
        n_excedente += len(membros) - 1
        gid = f"g{n_grupos:05d}"
        # Canônico = o mais antigo, para a divisão temporal ficar coerente.
        membros.sort(key=lambda i: (registros[i].get("data_publicacao") or "9999",
                                    registros[i].get("url_origem_fake") or ""))
        for pos, i in enumerate(membros):
            registros[i]["grupo_duplicata"] = gid
            registros[i]["duplicata_canonica"] = (pos == 0)

    # --- divisão temporal, com o grupo inteiro do mesmo lado ---------------
    def chave(r):
        return r.get("data_publicacao") or "9999-99-99"

    data_do_grupo = {}
    for r in registros:
        g = r.get("grupo_duplicata")
        if g:
            data_do_grupo[g] = min(data_do_grupo.get(g, "9999-99-99"), chave(r))

    ordenados = sorted(range(len(registros)),
                       key=lambda i: data_do_grupo.get(registros[i].get("grupo_duplicata"),
                                                       chave(registros[i])))
    n = len(ordenados)
    corte_t, corte_v = int(n * FRACAO_TREINO), int(n * (FRACAO_TREINO + FRACAO_VAL))
    split_do_grupo = {}
    for pos, i in enumerate(ordenados):
        s = "treino" if pos < corte_t else ("validacao" if pos < corte_v else "teste")
        g = registros[i].get("grupo_duplicata")
        if g:
            s = split_do_grupo.setdefault(g, s)
        registros[i]["split"] = s
    return n_grupos, n_excedente


# ==========================================================================
def main():
    aplicar = "--aplicar" in sys.argv
    print("=" * 72)
    print("  CORREÇÃO DO CORPUS  " + ("[APLICANDO]" if aplicar else "[DRY-RUN — nada será gravado]"))
    print("=" * 72)

    with open(ARQ_BRUTO, encoding="utf-8") as f:
        bruto = json.load(f)
    with open(ARQ_FINAL, encoding="utf-8") as f:
        final = json.load(f)
    print(f"\nentrada: {len(final):,} instâncias / {len(bruto):,} artigos"
          .replace(",", "."))

    # ---- A ----
    alvos, removidos, mantidos = etapa_A(bruto, final)
    print(f"\n[A] viral textual sem âncora de fim")
    print(f"    artigos a reprocessar ......... {len(alvos):,}".replace(",", "."))
    print(f"    instâncias retiradas .......... {len(removidos):,}".replace(",", "."))
    print(f"    instâncias remanescentes ...... {len(mantidos):,}".replace(",", "."))

    # ---- B ----
    tocados = etapa_B(mantidos)
    print(f"\n[B] alegação dentro da evidência")
    print(f"    justificativa_bruta corrigida .. {tocados:,}".replace(",", "."))

    # ---- campos ----
    por_url = {x.get("url_origem"): x for x in bruto}
    com_data, com_sup = etapa_campos(mantidos, por_url)
    print(f"\n[C] campos derivados")
    print(f"    com data_publicacao ............ {com_data:,} "
          f"({100 * com_data / max(1, len(mantidos)):.1f}%)".replace(",", "."))
    print(f"    com suporte declarado .......... {com_sup:,} "
          f"({100 * com_sup / max(1, len(mantidos)):.1f}%)".replace(",", "."))
    print("    suporte:", dict(Counter(r.get("suporte") for r in mantidos).most_common(6)))
    print("    tema_pareamento:")
    for t, n in Counter(r.get("tema_pareamento") for r in mantidos).most_common():
        orig = Counter(r.get("tema") for r in mantidos if r.get("tema_pareamento") == t)
        print(f"      {t:22s} {mil(n):>7}  <- {', '.join(k for k, _ in orig.most_common(3))}")
    llm = sum(1 for r in mantidos if r.get("tema_pareamento_origem") == "mapa_tema_llm")
    print(f"    baldes derivados de tema atribuído por LLM: {mil(llm)} "
          f"({100 * llm / max(1, len(mantidos)):.1f}%)")

    # ---- D ----
    n_grupos, n_exc = etapa_D(mantidos)
    print(f"\n[D] duplicatas e divisão temporal")
    print(f"    grupos de quase-duplicatas ..... {n_grupos:,}".replace(",", "."))
    print(f"    registros excedentes ........... {n_exc:,} "
          f"({100 * n_exc / max(1, len(mantidos)):.1f}%)".replace(",", "."))
    print("    split:", dict(Counter(r.get("split") for r in mantidos).most_common()))
    # Percentis, não mín/máx: um punhado de registros atravessa a fronteira de
    # propósito, porque quase-duplicatas têm de ficar do mesmo lado da divisão,
    # e um grupo pode reunir um boato de 2013 com sua reciclagem de 2026. Mín e
    # máx mostram só esses outliers e fazem a divisão parecer vazada.
    for s in ("treino", "validacao", "teste"):
        d = sorted(r["data_publicacao"] for r in mantidos
                   if r.get("split") == s and r.get("data_publicacao"))
        if d:
            p5, p50, p95 = d[len(d) // 20], d[len(d) // 2], d[-max(1, len(d) // 20)]
            print(f"      {s:10s} p5 {p5} · mediana {p50} · p95 {p95}   ({len(d):,})"
                  .replace(",", "."))
    deslocados = sum(1 for r in mantidos if r.get("grupo_duplicata")
                     and not r.get("duplicata_canonica"))
    sem_data = sum(1 for r in mantidos if not r.get("data_publicacao"))
    print(f"    registros movidos para manter o grupo junto: {deslocados:,}"
          .replace(",", "."))
    if sem_data:
        print(f"    sem data (vão para 'teste'): {sem_data}")

    if not aplicar:
        print("\n" + "-" * 72)
        print("DRY-RUN: nada foi gravado. Rode com --aplicar para efetivar.")
        print("Depois, para reprocessar os artigos devolvidos à fila:")
        print("    w.LIMITE_ARTIGOS_NLP = None ; w.main_nlp()")
        return

    # ---- gravação, com backup ----
    carimbo = datetime.now().strftime("%Y%m%d_%H%M")
    for caminho in (ARQ_BRUTO, ARQ_FINAL):
        destino = caminho.replace(".json", f"_BACKUP_{carimbo}.json")
        shutil.copy2(caminho, destino)
        print(f"\nbackup: {os.path.basename(destino)}")

    aplicar_A(bruto, alvos)
    with open(ARQ_FINAL, "w", encoding="utf-8") as f:
        json.dump(mantidos, f, indent=4, ensure_ascii=False)
    with open(ARQ_BRUTO, "w", encoding="utf-8") as f:
        json.dump(bruto, f, indent=4, ensure_ascii=False)

    na_fila = sum(1 for i in bruto if not i.get("processado_pela_ia"))
    print(f"\nGravado. {len(mantidos):,} instâncias; {len(alvos):,} artigos devolvidos à fila "
          f"pela etapa A; {na_fila:,} na fila do main_nlp() no total.".replace(",", "."))
    # Só manda reprocessar quando esta execução devolveu algo -- antes a mensagem
    # aparecia sempre, inclusive com a fila vazia.
    if alvos:
        print("Agora rode:  w.LIMITE_ARTIGOS_NLP = None ; w.main_nlp()")


if __name__ == "__main__":
    main()
