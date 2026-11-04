# -*- coding: utf-8 -*-
"""Refinamento CEGO -- o braço refinado da RQ1.

Um único prompt, que não sabe o rótulo, reescreve o `texto_entrada` de cada
instância (as duas classes do corpus e as duas do Fake.br) como uma frase
neutra com a alegação central. O braço refinado do experimento usa essa frase
no lugar do texto original: o estilo (caixa alta, "!", "você", "disse hoje")
some, a alegação fica.

Três modos, em três momentos:

    # 1. AQUI (tem o corpus e o Fake.br): monta a fila portátil, alguns MB
    python scripts/refinar_cego.py --preparar

    # 2. NA MÁQUINA COM GPU (só precisa da fila e do Ollama com qwen3.5:9b)
    python scripts/refinar_cego.py --piloto 25     # 25 por grupo, NÃO grava na saída
    python scripts/refinar_cego.py                 # fila inteira; retomável

    # 3. AQUI: a saída volta e é juntada ao corpus pelo id (passo seguinte)

Arquivos:
    datasets/refinamento_fila.jsonl    id, origem, grupo, split, texto  (entrada)
    datasets/refinamento_cego.jsonl    id, ideia_central, prompt_sha... (saída, append)
    relatorios/piloto_refinamento_<versão>.json  estatísticas do piloto, por grupo

REGRAS DO DESENHO (não mudar depois do pré-registro do braço refinado):
  - o prompt é o MESMO para todos os grupos e não menciona veracidade, checagem
    ou origem do texto;
  - a entrada é o `texto_entrada` do braço bruto (corpus) e a entrada A da
    avaliação cruzada (Fake.br) -- o refinamento muda a representação, não o
    recorte;
  - cada saída grava `prompt_versao` e `prompt_sha`: a prova de que os dois lados
    e o Fake.br passaram pelo mesmo prompt.

O RISCO QUE O PILOTO MEDE: mesmo cego, o modelo pode inferir que uma alegação é
implausível e escrever com distanciamento ("teria", "supostamente"), ou manter
a atribuição jornalística ("o ministro afirmou") só nas verdadeiras. Qualquer
uma dessas marcas, diferente entre as classes, é um atalho NOVO criado pelo
refinamento. O piloto conta as marcas por grupo antes de soltar a fila.
"""
import os
import re
import sys
import json
import time
import random
import hashlib
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_FILA = os.path.join(_RAIZ, "datasets", "refinamento_fila.jsonl")
ARQ_SAIDA = os.path.join(_RAIZ, "datasets", "refinamento_cego.jsonl")
ARQ_TRAVA = os.path.join(_RAIZ, "datasets", ".refinar_cego.lock")
ARQ_PILOTO = os.path.join(_RAIZ, "relatorios", "piloto_refinamento_{versao}.json")   # um por versão do prompt
SEED = 0
SALVAR_A_CADA = 20

# ==========================================================================
# O PROMPT
# ==========================================================================
# Histórico (o piloto do cego-v1, 25 por grupo, em 05/10/2026, motivou a v2):
#   cego-v1 -- "disse/afirmou só quando a declaração for o fato central": as
#              notícias mantiveram a atribuição ("a Anvisa afirma", "segundo
#              estudo") mais que as correntes; 8% das falsas saíram "O texto
#              alerta..."; verdadeiras 2-4 palavras mais longas, frases > 30.
#   cego-v2 -- atribuição proibida nos DOIS lados; frase começa pelo sujeito
#              do fato; 12 a 25 palavras.
PROMPT_VERSAO = "cego-v2"
PROMPT = """Leia o texto abaixo e escreva, em português, UMA ÚNICA FRASE com o fato ou a afirmação central que ele apresenta.

Regras da frase:
1. Uma só frase declarativa, de 12 a 25 palavras, em terceira pessoa.
2. Comece pelo sujeito do fato: a pessoa, a instituição, o lugar ou o acontecimento. Nunca comece nem fale sobre o próprio texto ("o texto", "a mensagem", "a notícia", "o vídeo", "o alerta", "a publicação").
3. Escreva o conteúdo diretamente, sem dizer quem o declarou, anunciou ou alertou. Não use "disse", "diz", "afirmou", "afirma", "declarou", "alerta", "anunciou", "informou", "segundo" nem "de acordo com".
4. Apresente o conteúdo como o texto o apresenta. Não acrescente distanciamento ("teria", "supostamente", "alegadamente") nem confirmação ("de fato", "comprovadamente").
5. Não avalie se o conteúdo é verdadeiro, falso, exagerado ou duvidoso.
6. Linguagem neutra e padrão: sem letras maiúsculas para ênfase, sem exclamações, sem emojis, sem hashtags, sem links, sem se dirigir ao leitor.
7. Mantenha os nomes de pessoas, lugares, instituições, números e datas que forem essenciais.

TEXTO:
{texto}"""
PROMPT_SHA = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()[:16]

SCHEMA = {
    "type": "object",
    "properties": {"ideia_central": {"type": "string",
                                     "description": "Uma única frase declarativa e neutra."}},
    "required": ["ideia_central"],
}

# ==========================================================================
# Marcas que NÃO podem diferir entre as classes depois do refinamento
# ==========================================================================
# Corrigidas depois do piloto do cego-v1: a atribuição só contava o passado
# ("afirmou") e deixava passar "afirma", "alerta", "diz"; o distanciamento
# misturava hedge com conteúdo ("áudio falso" é a própria alegação) e nome
# próprio ("Comissão da Verdade"). Agora são quatro marcas separadas.
_VERBOS_DECLARACAO = (r"diz|dizem|disse|disseram|afirma|afirmam|afirmou|afirmaram|declara|declarou|"
                      r"alerta|alertam|alertou|anuncia|anunciou|informa|informou|garante|garantiu|"
                      r"relata|relatou|ressalta|ressaltou|destaca|destacou|revela|revelou|"
                      r"denuncia|denunciou|defende|defendeu|admite|admitiu|nega|negou")
MARCAS = {
    "atribuicao": re.compile(rf"\b(?:{_VERBOS_DECLARACAO}|segundo|de acordo com|conforme)\b", re.I),
    "distanciamento": re.compile(r"\b(?:teria|teriam|supostamente|suposto|suposta|supostos|supostas|"
                                 r"alegadamente|alegado|alegada|possivelmente|aparentemente)\b", re.I),
    "veredito": re.compile(r"\b(?:é falso|é falsa|são falsos|são falsas|é mentira|boato|boatos|fake news|"
                           r"desmentid\w*|não procede|não é verdade)\b", re.I),
    "metatexto": re.compile(r"\b(?:o texto|a mensagem|a notícia|a publicação|o post|a matéria|"
                            r"o alerta|a corrente|o vídeo (?:mostra|diz|alerta))\b", re.I),
}


def medir(frase):
    palavras = re.findall(r"\w+", frase)
    letras = [c for c in frase if c.isalpha()]
    return {
        "palavras": len(palavras),
        "frases": len([f for f in re.split(r"(?<=[.!?])\s+", frase.strip()) if f]),
        "pct_maiusculas": sum(c.isupper() for c in letras) / max(1, len(letras)),
        "exclamacao": frase.count("!"),
        **{k: bool(rx.search(frase)) for k, rx in MARCAS.items()},
    }


def ler_jsonl(caminho):
    if not os.path.exists(caminho):
        return []
    saida = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if linha.strip():
                try:
                    saida.append(json.loads(linha))
                except json.JSONDecodeError:
                    pass                  # linha truncada por interrupção: ignora
    return saida


# ==========================================================================
# 1. --preparar (aqui)
# ==========================================================================
def preparar():
    """Fila = subconjunto pareado do corpus + os 7.200 textos do Fake.br.
    Os textos já são os do braço bruto: `texto_entrada` e a entrada A da
    avaliação cruzada (mesmo recorte e normalização)."""
    from avaliacao_cruzada import carregar_fakebr, entrada_texto
    fila = []
    with open(ARQ_CORPUS, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["subconjunto_pareado"]:
                grupo = "corpus_falso" if r["veredito"] == "FALSO" else "corpus_verdadeiro"
                fila.append({"id": r["id"], "origem": "corpus", "grupo": grupo,
                             "split": r["split"], "texto": r["texto_entrada"]})
    pares, _ = carregar_fakebr()
    for p in pares:
        for classe, grupo in (("fake", "fakebr_falso"), ("true", "fakebr_verdadeiro")):
            fila.append({"id": f"FB-{classe}-{p['n']}", "origem": "fakebr", "grupo": grupo,
                         "split": "fakebr", "texto": entrada_texto(p[classe]["bruto"])})
    # Ordem embaralhada com semente: uma interrupção no meio deixa a saída
    # parcial com as quatro classes misturadas, não uma inteira e outra vazia.
    random.Random(SEED).shuffle(fila)
    tmp = ARQ_FILA + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in fila:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, ARQ_FILA)
    print(f"fila: {len(fila):,} textos -> {ARQ_FILA}".replace(",", "."))
    print("  por grupo:", dict(Counter(r["grupo"] for r in fila)))
    print(f"  tamanho: {os.path.getsize(ARQ_FILA) / 1e6:.1f} MB")
    print(f"  prompt {PROMPT_VERSAO} sha {PROMPT_SHA}")


# ==========================================================================
# chamada ao modelo
# ==========================================================================
# Limpeza da SAÍDA (não do prompt -- o hash não muda). No piloto do cego-v2, 1
# de 200 frases veio com a autoverificação do modelo colada: "Tico Santa Cruz...
# **Análise da frase gerada:** * Uma só frase declarativa? Sim. * 12 a 25
# palavras?...". Corta no primeiro sinal de texto que não é a frase; cada corte
# fica registrado em `saida_ajustes`, para medir quantas foram mexidas.
CORTES_SAIDA = [
    ("quebra_de_linha", re.compile(r"\n")),
    ("markdown", re.compile(r"\*\*|^\s*[*#-]\s", re.M)),
    # Só expressões que nenhuma frase de notícia ou boato usa -- "verificação"
    # sozinha cortaria "O teste de verificação das urnas...".
    ("autoanalise", re.compile(r"\b(?:análise da frase|frase gerada|checagem das regras|"
                               r"contagem de palavras|regras? atendidas?)\b", re.I)),
]


def limpar_saida(frase):
    """Devolve (frase limpa, lista de ajustes)."""
    ajustes = []
    for nome, rx in CORTES_SAIDA:
        m = rx.search(frase)
        if m and m.start() > 0:
            frase = frase[:m.start()]
            ajustes.append(nome)
    limpa = re.sub(r"\s+", " ", frase).strip().strip("\"'“”*").strip()
    if limpa != re.sub(r"\s+", " ", frase).strip():
        ajustes.append("aspas_ou_marcadores")
    # Se o que sobrou ainda é autoanálise ou markdown (a resposta COMEÇOU por
    # ela), não há frase aproveitável: devolve vazio e o modelo refaz.
    if "**" in limpa or CORTES_SAIDA[2][1].search(limpa):
        return "", ajustes + ["descartada"]
    return limpa, ajustes


def refinar(texto, chamar_llm):
    """Devolve (frase, segundos, ajustes) ou (None, segundos, []) se o modelo
    falhou. Frase que fica vazia depois da limpeza conta como falha e é refeita."""
    t0 = time.time()
    for _ in range(3):
        resposta = chamar_llm(PROMPT.format(texto=texto), format_schema=SCHEMA)
        if resposta is None:
            break
        try:
            frase = (json.loads(resposta).get("ideia_central") or "").strip()
        except json.JSONDecodeError:
            continue
        frase, ajustes = limpar_saida(frase)
        if len(re.findall(r"\w+", frase)) >= 3:
            return frase, time.time() - t0, ajustes
    return None, time.time() - t0, []


def carregar_llm():
    import web_scraping_limpo as w            # mesmo cliente, modelo e think=False da extração
    return w.chamar_llm, w.OLLAMA_MODEL


# ==========================================================================
# 2a. --piloto N (máquina com GPU; não grava na saída)
# ==========================================================================
def piloto(n):
    """N textos por grupo, só do split de TREINO no corpus (o piloto avalia o
    formato da frase; não deve olhar o teste)."""
    fila = ler_jsonl(ARQ_FILA)
    if not fila:
        sys.exit(f"[!] {ARQ_FILA} não existe -- rode --preparar na máquina que tem o corpus.")
    chamar_llm, modelo = carregar_llm()
    por_grupo = defaultdict(list)
    for r in fila:
        if r["split"] in ("treino", "fakebr") and len(por_grupo[r["grupo"]]) < n:
            por_grupo[r["grupo"]].append(r)

    print(f"PILOTO -- {n} por grupo | modelo {modelo} | prompt {PROMPT_VERSAO} sha {PROMPT_SHA}\n")
    resultados = defaultdict(list)
    for grupo in sorted(por_grupo):
        print("=" * 78)
        print(grupo)
        for r in por_grupo[grupo]:
            frase, seg, ajustes = refinar(r["texto"], chamar_llm)
            if frase is None:
                print(f"   [FALHA] {r['id']}")
                continue
            m = medir(frase)
            m["segundos"] = seg
            m["ajustada"] = bool(ajustes)
            resultados[grupo].append(m)
            alerta = [k for k in MARCAS if m[k]] + [f"saída cortada: {'+'.join(ajustes)}"] * bool(ajustes)
            print(f"   ({seg:4.1f}s) {r['texto'][:90]!r}")
            print(f"        -> {frase}" + (f"   <{', '.join(alerta)}>" if alerta else ""))

    print("\n" + "=" * 78)
    print("RESUMO POR GRUPO -- as marcas precisam ser parecidas entre falso e verdadeiro")
    colunas = ["palavras", ">25 pal.", ">1 frase", "cortada"] + list(MARCAS) + ["s/texto"]
    print(f"  {'grupo':20s} {'n':>3s} " + " ".join(f"{c[:10]:>10s}" for c in colunas))
    resumo = {}
    for grupo, ms in sorted(resultados.items()):
        if not ms:
            continue
        k = len(ms)
        linha = {
            "n": k,
            "palavras_media": round(sum(m["palavras"] for m in ms) / k, 1),
            "pct_mais_de_25_palavras": round(100 * sum(m["palavras"] > 25 for m in ms) / k, 1),
            "pct_mais_de_uma_frase": round(100 * sum(m["frases"] > 1 for m in ms) / k, 1),
            "pct_saida_cortada": round(100 * sum(m["ajustada"] for m in ms) / k, 1),
            **{f"pct_{c}": round(100 * sum(m[c] for m in ms) / k, 1) for c in MARCAS},
            "segundos_media": round(sum(m["segundos"] for m in ms) / k, 2),
        }
        resumo[grupo] = linha
        valores = ([f"{linha['palavras_media']:10.1f}", f"{linha['pct_mais_de_25_palavras']:9.1f}%",
                    f"{linha['pct_mais_de_uma_frase']:9.1f}%", f"{linha['pct_saida_cortada']:9.1f}%"]
                   + [f"{linha[f'pct_{c}']:9.1f}%" for c in MARCAS]
                   + [f"{linha['segundos_media']:10.2f}"])
        print(f"  {grupo:20s} {k:3d} " + " ".join(valores))

    # Diferença falso − verdadeiro DENTRO de cada base: é ela que vira atalho.
    print("\n  diferença FALSO − VERDADEIRO (pontos percentuais; palavras em palavras):")
    for base in ("corpus", "fakebr"):
        f, v = resumo.get(f"{base}_falso"), resumo.get(f"{base}_verdadeiro")
        if not (f and v):
            continue
        difs = {"palavras": f["palavras_media"] - v["palavras_media"],
                **{c: f[f"pct_{c}"] - v[f"pct_{c}"] for c in MARCAS}}
        alertas = [c for c, d in difs.items() if (abs(d) >= 3 if c == "palavras" else abs(d) >= 10)]
        print(f"    {base:8s} " + "  ".join(f"{c} {d:+.1f}" for c, d in difs.items())
              + (f"   <-- ATENÇÃO: {', '.join(alertas)}" if alertas else "   ok"))
        resumo[f"diferenca_{base}"] = {c: round(d, 1) for c, d in difs.items()}
    seg = [m["segundos"] for ms in resultados.values() for m in ms]
    if seg:
        media = sum(seg) / len(seg)
        print(f"\n  estimativa da fila inteira: {len(fila):,} × {media:.2f} s = {len(fila) * media / 3600:.1f} h"
              .replace(",", "."))
    destino = ARQ_PILOTO.format(versao=PROMPT_VERSAO)
    with open(destino, "w", encoding="utf-8") as f:
        json.dump({"modelo": modelo, "prompt_versao": PROMPT_VERSAO, "prompt_sha": PROMPT_SHA,
                   "n_por_grupo": n, "resumo": resumo}, f, indent=2, ensure_ascii=False)
    print(f"\nresumo salvo em {destino}")


# ==========================================================================
# 2b. execução completa (máquina com GPU; retomável)
# ==========================================================================
def executar(limite=None):
    fila = ler_jsonl(ARQ_FILA)
    if not fila:
        sys.exit(f"[!] {ARQ_FILA} não existe -- rode --preparar na máquina que tem o corpus.")
    feitos = ler_jsonl(ARQ_SAIDA)
    shas = {r.get("prompt_sha") for r in feitos}
    if feitos and shas != {PROMPT_SHA}:
        # Saídas de prompts diferentes no mesmo arquivo invalidariam o braço inteiro.
        sys.exit(f"[!] {ARQ_SAIDA} tem saídas de outro prompt ({shas - {PROMPT_SHA}}). "
                 f"Mova o arquivo antes de rodar o prompt {PROMPT_SHA}.")
    ja = {r["id"] for r in feitos}
    pendentes = [r for r in fila if r["id"] not in ja]
    if limite:
        pendentes = pendentes[:limite]
    print(f"fila {len(fila):,} | feitos {len(ja):,} | nesta execução {len(pendentes):,}".replace(",", "."))
    if not pendentes:
        return

    try:
        fd = os.open(ARQ_TRAVA, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        sys.exit(f"[!] Outra execução parece estar rodando ({ARQ_TRAVA}). Se não houver, apague o arquivo.")

    chamar_llm, modelo = carregar_llm()
    print(f"modelo {modelo} | prompt {PROMPT_VERSAO} sha {PROMPT_SHA}")
    print("(Ctrl+C a qualquer momento: a próxima execução continua de onde parou)\n")
    saida = open(ARQ_SAIDA, "a", encoding="utf-8")
    inicio, ok, falhas = time.time(), 0, 0
    try:
        for i, r in enumerate(pendentes, 1):
            frase, seg, ajustes = refinar(r["texto"], chamar_llm)
            if frase is None:
                falhas += 1               # não grava: continua pendente na próxima execução
                print(f"[{i}/{len(pendentes)}] [X] {r['id']}")
                continue
            saida.write(json.dumps({"id": r["id"], "origem": r["origem"], "grupo": r["grupo"],
                                    "ideia_central": frase, "modelo": modelo,
                                    "prompt_versao": PROMPT_VERSAO, "prompt_sha": PROMPT_SHA,
                                    "saida_ajustes": ajustes,
                                    "segundos": round(seg, 2)}, ensure_ascii=False) + "\n")
            ok += 1
            if ok % SALVAR_A_CADA == 0:
                saida.flush()
                os.fsync(saida.fileno())
                passou = time.time() - inicio
                resta = (len(pendentes) - i) * passou / i / 3600
                print(f"[{i}/{len(pendentes)}] [salvo] {len(ja) + ok:,} feitos | restam ~{resta:.1f} h"
                      .replace(",", "."))
    except KeyboardInterrupt:
        print("\n[interrompido] gravando o que já foi refinado...")
    finally:
        saida.flush()
        os.fsync(saida.fileno())
        saida.close()
        os.remove(ARQ_TRAVA)
    print(f"\n-> {ok:,} refinados nesta execução, {falhas} falhas (seguem pendentes).".replace(",", "."))


def main():
    if "--preparar" in sys.argv:
        preparar()
    elif "--piloto" in sys.argv:
        piloto(int(sys.argv[sys.argv.index("--piloto") + 1]))
    else:
        limite = int(sys.argv[sys.argv.index("--limite") + 1]) if "--limite" in sys.argv else None
        executar(limite)


if __name__ == "__main__":
    main()
