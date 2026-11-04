# -*- coding: utf-8 -*-
"""Prepara o lado VERDADEIRO para a fusão: duplicatas, lides quebrados e split.

Mesmo espírito do `corrigir_corpus.py`: dry-run por padrão, backup ao aplicar,
e duplicatas MARCADAS, não apagadas. Apagar seria decisão irreversível tomada
por script; marcar deixa a escolha para o experimento e permite relatar quanto
a deduplicação muda o resultado. A fusão usa só `duplicata_canonica == True`.

    python scripts/preparar_verdadeiras.py              # simula, não grava
    python scripts/preparar_verdadeiras.py --aplicar    # grava (com backup)

NÃO rode com o coletar_verdadeiras.py em execução: os dois usam o mesmo arquivo.
"""
import os
import re
import sys
import json
import zlib
import shutil
import statistics
from datetime import datetime
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_FALSAS = os.path.join(_RAIZ, "datasets", "dataset_tcc_fake_news_final.json")
ARQ_VERDADEIRAS = os.path.join(_RAIZ, "datasets", "dataset_verdadeiras.jsonl")

# Mesmo limiar do lado falso, para as duas classes serem deduplicadas pelo
# mesmo critério -- critério diferente por classe seria mais uma assimetria.
LIMIAR_JACCARD = 0.70
K_SHINGLE = 5
BOTTOM_K = 12          # candidatos: pares que compartilham um dos 12 menores hashes
MAX_BALDE = 200        # hash presente em muitos textos é boilerplate, não indício
MIN_LIDE = 40
# Nas páginas de 2014-2019 o og:description traz a LEGENDA e o CRÉDITO da foto
# grudados antes do lide: "...Osmar TerraWilson Dias/Agência Brasil/Arquivo O
# ministro do Desenvolvimento...". MEDIDO: 325 lides. O lide real começa depois
# do crédito. "/AP" fica de fora de propósito: é também sigla de estado
# ("Macapá/AP"), e cortar ali amputaria lide legítimo.
CREDITO_FOTO = re.compile(
    r"(?:\s*/\s*|Arquivo\s*/?\s*)"
    r"(?:Ag[êe]ncia Brasil|Reuters|AFP|EFE|Divulga[çc][ãa]o|Arquivo)"
    r"(?:\s*/\s*(?:Arquivo|Ag[êe]ncia Brasil))*")
MAX_POS_CREDITO = 0.8      # crédito no fim do lide não tem lide depois dele


def ler_jsonl(caminho):
    saida = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if linha.strip():
                try:
                    saida.append(json.loads(linha))
                except Exception:
                    pass
    return saida


def mil(n):
    return "{:,}".format(n).replace(",", ".")


def shingles(texto, k=K_SHINGLE):
    """Sequências de k palavras, como inteiros. crc32 e não hash(): o hash de
    str do Python muda a cada processo, e o resultado tem de ser reproduzível."""
    p = re.findall(r"\w+", (texto or "").lower())
    return {zlib.crc32(" ".join(p[i:i + k]).encode()) for i in range(max(0, len(p) - k + 1))}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


# ==========================================================================
# 1 -- lides quebrados
# ==========================================================================
def corrigir_lides(regs):
    """O og:description às vezes vem vazio de sentido ("x", um fragmento). Como
    o lide é metade da entrada do experimento (título + lide), um lide de
    1 caractere vira um exemplo degenerado. Troca pela 1ª sentença do corpo,
    que é o que o lide seria no texto jornalístico, e registra a troca."""
    trocados = []
    for r in regs:
        r.setdefault("lide_origem", "og:description")
        if len(r.get("lide") or "") >= MIN_LIDE:
            continue
        m = re.search(r"^(.{80,400}?[.!?])\s", r["conteudo_real"])
        novo = m.group(1) if m else r["conteudo_real"][:300]
        trocados.append((r["lide"], novo))
        r["lide"], r["lide_origem"] = novo, "primeira_sentenca_do_corpo"
    return trocados


def tirar_legenda_da_foto(regs):
    """Corta legenda + crédito do começo do lide. Só age quando sobra um lide
    de verdade depois do crédito; o lide original fica em `lide_original`."""
    tocados = []
    for r in regs:
        if r.get("lide_original"):
            continue                          # já tratado numa execução anterior
        m = CREDITO_FOTO.search(r["lide"])
        if not m or m.end() / len(r["lide"]) > MAX_POS_CREDITO:
            continue
        resto = r["lide"][m.end():].strip()
        # Crédito em dois níveis ("Fulano/EFE/Direitos Reservados"): tira os
        # sufixos conhecidos. Se ainda sobrar "/Algo", não dá para saber por
        # regra onde o crédito acaba ("/Theatro Municipal Mesmo enfrentando..."),
        # então o lide vira a 1ª sentença do corpo -- a mesma saída dos lides curtos.
        resto = re.sub(r"^(?:\s*/\s*(?:direitos reservados|PR)\s*\)?)+\s*", "", resto, flags=re.I)
        origem = "og:description_sem_legenda"
        if resto.startswith("/") or len(resto) < MIN_LIDE:
            s = re.search(r"^(.{80,400}?[.!?])\s", r["conteudo_real"])
            resto = s.group(1) if s else r["conteudo_real"][:300]
            origem = "primeira_sentenca_do_corpo"
        tocados.append((r["lide"], resto))
        r["lide_original"], r["lide"] = r["lide"], resto
        r["lide_origem"] = origem
    return tocados


# ==========================================================================
# 2 -- duplicatas
# ==========================================================================
def agrupar_duplicatas(regs):
    """Jaccard sobre shingles de 5 palavras do CORPO INTEIRO.

    Não usa título nem URL: a mesma matéria pode ter o título corrigido e ganhar
    URL nova, e matérias diferentes podem ter título igual (boletim diário).

    Comparar todos contra todos seriam 86 milhões de pares. Os candidatos saem
    de um esboço bottom-k: dois textos com Jaccard J compartilham o menor hash
    com probabilidade J, então com 12 hashes um par a 0,70 quase nunca escapa.
    Todo candidato é conferido pelo Jaccard exato antes de entrar no grupo."""
    sh = [shingles(r["conteudo_real"]) for r in regs]
    baldes = defaultdict(list)
    for i, s in enumerate(sh):
        for h in sorted(s)[:BOTTOM_K]:
            baldes[h].append(i)

    pai = list(range(len(regs)))

    def raiz(x):
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    vistos, pares = set(), []
    for membros in baldes.values():
        if len(membros) < 2 or len(membros) > MAX_BALDE:
            continue
        for a in range(len(membros)):
            for b in range(a + 1, len(membros)):
                i, j = membros[a], membros[b]
                if (i, j) in vistos:
                    continue
                vistos.add((i, j))
                jac = jaccard(sh[i], sh[j])
                pares.append((jac, i, j))
                if jac >= LIMIAR_JACCARD:
                    pai[raiz(i)] = raiz(j)

    grupos = defaultdict(list)
    for i in range(len(regs)):
        grupos[raiz(i)].append(i)

    n = 0
    for membros in grupos.values():
        if len(membros) < 2:
            regs[membros[0]]["grupo_duplicata"] = None
            regs[membros[0]]["duplicata_canonica"] = True
            continue
        n += 1
        gid = f"v{n:05d}"
        # Canônico = o mais antigo; empate, o de corpo mais longo (a versão
        # corrigida costuma acrescentar, não cortar).
        membros.sort(key=lambda i: (regs[i]["data_publicacao"], -len(regs[i]["conteudo_real"]),
                                    regs[i]["url_origem_real"]))
        for pos, i in enumerate(membros):
            regs[i]["grupo_duplicata"] = gid
            regs[i]["duplicata_canonica"] = (pos == 0)
    return pares


# ==========================================================================
# 3 -- split temporal, com os MESMOS cortes do lado falso
# ==========================================================================
def cortes_do_lado_falso():
    """O split das falsas foi feito por posição (70/10/20 dos mais antigos).
    As verdadeiras não podem ter corte próprio: com datas de corte diferentes,
    uma faixa de meses teria verdadeiras no treino e falsas no teste, e a
    época passaria a separar as classes. O corte é a primeira data de cada
    faixa entre as falsas que NÃO foram deslocadas para manter grupo junto."""
    with open(ARQ_FALSAS, encoding="utf-8") as f:
        falsas = json.load(f)

    def inicio(s):
        return min(r["data_publicacao"] for r in falsas
                   if r.get("split") == s and not r.get("grupo_duplicata")
                   and r.get("data_publicacao"))
    return inicio("validacao"), inicio("teste"), Counter(r.get("split") for r in falsas)


def aplicar_split(regs, c_val, c_teste):
    def faixa(d):
        return "treino" if d < c_val else ("validacao" if d < c_teste else "teste")
    # Grupo inteiro do mesmo lado, pela data do canônico -- senão a cópia de
    # uma matéria de treino apareceria no teste.
    data_do_grupo = {}
    for r in regs:
        g = r.get("grupo_duplicata")
        if g:
            data_do_grupo[g] = min(data_do_grupo.get(g, "9999"), r["data_publicacao"])
    for r in regs:
        r["split"] = faixa(data_do_grupo.get(r.get("grupo_duplicata"), r["data_publicacao"]))


# ==========================================================================
def main():
    aplicar = "--aplicar" in sys.argv
    print("=" * 72)
    print("  PREPARAÇÃO DAS VERDADEIRAS  " + ("[APLICANDO]" if aplicar else "[DRY-RUN — nada será gravado]"))
    print("=" * 72)

    regs = ler_jsonl(ARQ_VERDADEIRAS)
    print(f"\nentrada: {mil(len(regs))} registros")

    # ---- 1 ----
    trocados = corrigir_lides(regs)
    print(f"\n[1] lides com menos de {MIN_LIDE} caracteres: {len(trocados)}")
    for velho, novo in trocados:
        print(f"    {velho!r:32.32s} -> {novo[:70]!r}")

    legendas = tirar_legenda_da_foto(regs)
    ja = sum(1 for r in regs if r.get("lide_original")) - len(legendas)
    print(f"\n[1b] legenda + crédito de foto no começo do lide: {len(legendas)} corrigidos agora"
          + (f" ({ja} já corrigidos antes)" if ja else ""))
    for velho, novo in legendas[:4]:
        print(f"    ANTES  {velho[:95]!r}")
        print(f"    DEPOIS {novo[:95]!r}")

    # ---- 2 ----
    pares = agrupar_duplicatas(regs)
    exced = sum(1 for r in regs if r.get("grupo_duplicata") and not r["duplicata_canonica"])
    grupos = {r["grupo_duplicata"] for r in regs if r.get("grupo_duplicata")}
    print(f"\n[2] duplicatas (Jaccard >= {LIMIAR_JACCARD} no corpo inteiro)")
    print(f"    pares candidatos conferidos .... {mil(len(pares))}")
    print(f"    grupos ......................... {mil(len(grupos))}")
    print(f"    registros excedentes ........... {mil(exced)} ({100 * exced / len(regs):.2f}%)")
    # A faixa logo abaixo do limiar mostra se o corte está num vale ou no meio
    # de uma encosta: se houver muitos pares em 0,60-0,70, o limiar importa.
    faixas = Counter()
    for jac, _, _ in pares:
        if jac >= 0.3:
            faixas[f"{int(jac * 10) / 10:.1f}"] += 1
    print("    distribuição dos pares candidatos por Jaccard:")
    for f in sorted(faixas):
        print(f"      {f}-{float(f) + 0.1:.1f}  {mil(faixas[f]):>6}")
    # Exemplos, para conferir a olho: os mais baixos acima do limiar e os mais
    # altos abaixo dele -- é na fronteira que o critério erra, se errar.
    acima = sorted(p for p in pares if p[0] >= LIMIAR_JACCARD)[:3]
    abaixo = sorted((p for p in pares if p[0] < LIMIAR_JACCARD), reverse=True)[:3]
    for rotulo, lista in (("menos similares ACIMA do limiar (marcados):", acima),
                          ("mais similares ABAIXO do limiar (mantidos):", abaixo)):
        print(f"    {rotulo}")
        for jac, i, j in lista:
            print(f"      J={jac:.2f}  {regs[i]['titulo'][:58]}")
            print(f"              {regs[j]['titulo'][:58]}")
    mesmo_titulo = Counter(r["titulo"] for r in regs)
    tit_nao_dup = sum(1 for r in regs if mesmo_titulo[r["titulo"]] > 1 and not r.get("grupo_duplicata"))
    print(f"    registros com título repetido mas corpo diferente (mantidos): {tit_nao_dup}")

    # ---- 3 ----
    c_val, c_teste, dist_f = cortes_do_lado_falso()
    aplicar_split(regs, c_val, c_teste)
    dist_v = Counter(r["split"] for r in regs if r["duplicata_canonica"])
    print(f"\n[3] split temporal -- cortes do lado falso: validação a partir de {c_val},"
          f" teste a partir de {c_teste}")
    for s in ("treino", "validacao", "teste"):
        print(f"    {s:10s} falsas {mil(dist_f[s]):>7}   verdadeiras {mil(dist_v[s]):>7}"
              f"   ({100 * dist_v[s] / (dist_v[s] + dist_f[s]):.0f}% verdadeiras)")

    if not aplicar:
        print("\n" + "-" * 72)
        print("DRY-RUN: nada foi gravado. Rode com --aplicar para efetivar.")
        return

    carimbo = datetime.now().strftime("%Y%m%d_%H%M")
    destino = ARQ_VERDADEIRAS.replace(".jsonl", f"_BACKUP_{carimbo}.jsonl")
    shutil.copy2(ARQ_VERDADEIRAS, destino)
    print(f"\nbackup: {os.path.basename(destino)}")
    # Só JSONL: nenhum script lê uma cópia .json, e ela dobrava o espaço em disco.
    tmp = ARQ_VERDADEIRAS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in regs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, ARQ_VERDADEIRAS)
    print(f"Gravado. {mil(len(regs))} registros, {mil(len(regs) - exced)} canônicos.")


if __name__ == "__main__":
    main()
