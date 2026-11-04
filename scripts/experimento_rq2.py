# -*- coding: utf-8 -*-
"""RQ2 -- experimento PAREADO: extração da mensagem viral pela estrutura × por LLM,
nos MESMOS artigos. Pré-registro: `relatorios/PREREGISTRO_RQ2_PAREADO.md`.

    # 1. AQUI (tem o corpus): sorteia os artigos e monta a fila portátil
    python scripts/experimento_rq2.py --preparar

    # 2. NA MÁQUINA COM GPU (Ollama + qwen3.5:9b): roda a via do LLM, retomável
    python scripts/experimento_rq2.py --rodar

    # 3. AQUI: compara item a item e grava relatorios/resultado_rq2_pareado.json
    python scripts/experimento_rq2.py --apurar

Por que existe: os números da RQ2 até aqui (literalidade html 99,4% × llm 87,1%)
vêm de conjuntos DISJUNTOS -- o LLM só rodava onde a página não tinha marcação,
que podem ser justamente os artigos mais difíceis. Aqui a via do LLM roda em
artigos que JÁ têm o recorte determinístico, e a comparação é no mesmo artigo.

DESENHO (fixado no pré-registro)
  - População: falsas do subconjunto pareado com origem_viral `html` ou `texto`,
    um bloco por artigo, uma única entrada pareada por URL.
  - Amostra: 200 por via (400), semente 20261008.
  - Via do LLM: a MESMA função do pipeline (`extrair_boatos_do_artigo`), com o
    artigo sem o recorte estrutural -- exatamente o que ela recebe quando a
    página não tem marcação. A resposta BRUTA do modelo também é gravada, para
    medir o que ele faz antes da rede de segurança (`limpar_viral`).
  - Referência: o `texto_viral_bruto` determinístico já gravado no corpus.

Arquivos:
    datasets/rq2_fila.jsonl          id, via, url, fonte, titulo, artigo, viral_det
    datasets/rq2_saida.jsonl         id, resposta_bruta, boatos, segundos, versão (append)
    relatorios/resultado_rq2_pareado.json
"""
import os
import re
import sys
import json
import time
import random
import hashlib
import inspect
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_FILA = os.path.join(_RAIZ, "datasets", "rq2_fila.jsonl")
ARQ_SAIDA = os.path.join(_RAIZ, "datasets", "rq2_saida.jsonl")
ARQ_TRAVA = os.path.join(_RAIZ, "datasets", ".rq2.lock")
ARQ_RESULTADO = os.path.join(_RAIZ, "relatorios", "resultado_rq2_pareado.json")
SEED = 20261008
N_POR_VIA = 200
CORTE_ARTIGO = 3800          # o que a via do LLM lê do artigo (extrair_boatos_do_artigo)

# Faixas da concordância com o recorte determinístico (pré-registradas).
IGUAL, PARCIAL = 0.90, 0.50
TITULO_COPIADO = 0.80


def mil(n):
    return "{:,}".format(n).replace(",", ".")


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


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
                    pass                  # linha truncada por interrupção
    return saida


# ==========================================================================
# 1. --preparar (aqui)
# ==========================================================================
def preparar():
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    falsas = [r for r in corpus if r["veredito"] == "FALSO" and r["subconjunto_pareado"]]
    por_url = Counter(r["url"] for r in falsas)
    pop = defaultdict(list)
    for r in falsas:
        via = str(r.get("origem_viral"))
        if via in ("html", "texto") and r.get("blocos_no_artigo", 1) == 1 and por_url[r["url"]] == 1:
            pop[via].append(r)
    rng = random.Random(SEED)
    fila = []
    for via in ("html", "texto"):
        regs = sorted(pop[via], key=lambda r: r["id"])
        for r in rng.sample(regs, N_POR_VIA):
            fila.append({"id": r["id"], "via": via, "url": r["url"], "fonte": r.get("fonte_fake") or r["fonte"],
                         "titulo": r.get("titulo_artigo_checagem") or "",
                         "artigo": r["texto_bruto_jornalista"], "viral_det": r["texto_viral_bruto"]})
    rng.shuffle(fila)                 # interrupção no meio deixa as duas vias misturadas
    with open(ARQ_FILA + ".tmp", "w", encoding="utf-8") as f:
        for r in fila:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(ARQ_FILA + ".tmp", ARQ_FILA)
    print(f"população: html {mil(len(pop['html']))} | texto {mil(len(pop['texto']))}")
    print(f"fila: {len(fila)} artigos ({N_POR_VIA} por via), semente {SEED} -> {ARQ_FILA}")
    print(f"  sha256 {sha256(ARQ_FILA)}")


# ==========================================================================
# 2. --rodar (máquina com GPU; retomável; não importa scikit-learn)
# ==========================================================================
def versao_codigo(w):
    """sha da função de extração do LLM: a prova de que a via testada é a do pipeline."""
    return hashlib.sha256(inspect.getsource(w.extrair_boatos_do_artigo).encode("utf-8")).hexdigest()[:16]


def rodar():
    fila = ler_jsonl(ARQ_FILA)
    if not fila:
        sys.exit(f"[!] {ARQ_FILA} não existe -- rode --preparar na máquina que tem o corpus.")
    import web_scraping_limpo as w
    sha = versao_codigo(w)
    feitos = ler_jsonl(ARQ_SAIDA)
    shas = {r.get("codigo_sha") for r in feitos}
    if feitos and shas != {sha}:
        sys.exit(f"[!] {ARQ_SAIDA} tem saídas de outra versão do código ({shas - {sha}}). "
                 f"Mova o arquivo antes de rodar a versão {sha}.")
    ja = {r["id"] for r in feitos}
    pendentes = [r for r in fila if r["id"] not in ja]
    print(f"fila {len(fila)} | feitos {len(ja)} | nesta execução {len(pendentes)}")
    print(f"modelo {w.OLLAMA_MODEL} | extração {w.VERSAO_EXTRACAO} | código sha {sha}")
    if not pendentes:
        return
    try:
        fd = os.open(ARQ_TRAVA, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        sys.exit(f"[!] Outra execução parece estar rodando ({ARQ_TRAVA}). Se não houver, apague o arquivo.")

    # Captura a resposta BRUTA do modelo, antes de limpar_viral() e dos filtros.
    respostas = []
    original = w.chamar_llm

    def capturar(prompt, format_schema=None):
        r = original(prompt, format_schema=format_schema)
        respostas.append(r)
        return r
    w.chamar_llm = capturar

    print("(Ctrl+C a qualquer momento: a próxima execução continua de onde parou)\n")
    saida = open(ARQ_SAIDA, "a", encoding="utf-8")
    inicio, ok, falhas = time.time(), 0, 0
    try:
        for i, r in enumerate(pendentes, 1):
            # Sem texto_viral_estrutural: é o que a função recebe numa página sem marcação.
            item = {"texto_bruto_jornalista": r["artigo"], "titulo": r["titulo"],
                    "url_origem": r["url"], "fonte": r["fonte"]}
            respostas.clear()
            t0 = time.time()
            boatos = w.extrair_boatos_do_artigo(item)
            seg = time.time() - t0
            if boatos is None:
                falhas += 1               # falha do modelo: segue pendente
                print(f"[{i}/{len(pendentes)}] [X] {r['id']}")
                continue
            saida.write(json.dumps({"id": r["id"], "via": r["via"], "boatos": boatos,
                                    "resposta_bruta": respostas[-1] if respostas else None,
                                    "tentativas": len(respostas), "segundos": round(seg, 2),
                                    "modelo": w.OLLAMA_MODEL, "extracao_versao": w.VERSAO_EXTRACAO,
                                    "codigo_sha": sha}, ensure_ascii=False) + "\n")
            saida.flush()
            ok += 1
            if ok % 10 == 0:
                passou = time.time() - inicio
                print(f"[{i}/{len(pendentes)}] {len(ja) + ok} feitos | restam ~{(len(pendentes) - i) * passou / i / 60:.0f} min")
    except KeyboardInterrupt:
        print("\n[interrompido] o que já foi feito está gravado.")
    finally:
        saida.close()
        os.remove(ARQ_TRAVA)
        w.chamar_llm = original
    print(f"\n-> {ok} artigos nesta execução, {falhas} falhas (seguem pendentes).")


# ==========================================================================
# 3. --apurar (aqui)
# ==========================================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    m = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [round(100 * max(0.0, c - m), 1), round(100 * min(1.0, c + m), 1)]


def pct(k, n):
    return {"n": k, "de": n, "pct": round(100 * k / n, 1) if n else None, "ic95": wilson(k, n)}


def apurar():
    from difflib import SequenceMatcher
    from scipy.stats import binomtest
    from limpeza_viral import limpar_viral
    import web_scraping_limpo as w

    fila = {r["id"]: r for r in ler_jsonl(ARQ_FILA)}
    saidas = {r["id"]: r for r in ler_jsonl(ARQ_SAIDA)}
    print(f"fila {len(fila)} | com saída {len(saidas)} | sem saída {len(fila) - len(saidas)}")
    norm = w._normalizar

    def sim(a, b):
        return SequenceMatcher(None, norm(a), norm(b), autojunk=False).ratio()

    linhas = []
    for i, q in fila.items():
        s = saidas.get(i)
        lin = {"id": i, "via": q["via"], "com_saida": s is not None,
               "det_literal": w.classificar_literalidade(q["viral_det"], q["artigo"])[1] == "conferido",
               "viral_no_corte": norm(q["viral_det"])[:80] in norm(q["artigo"][:CORTE_ARTIGO])}
        if s is not None:
            virais = [b["texto_viral_bruto"] for b in s["boatos"]]
            lin["rendeu"] = bool(virais)
            if virais:
                # O item do LLM mais parecido com a referência (artigo de um bloco só).
                melhor = max(virais, key=lambda v: sim(v, q["viral_det"]))
                lin["similaridade"] = sim(melhor, q["viral_det"])
                lin["llm_literal"] = w.classificar_literalidade(melhor, q["artigo"])[1] == "conferido"
                lin["titulo_copiado"] = bool(q["titulo"]) and sim(melhor, q["titulo"]) >= TITULO_COPIADO
                lin["razao_tamanho"] = len(melhor) / max(1, len(q["viral_det"]))
            # Antes da rede de segurança: o que o modelo devolveu de fato.
            brutos = []
            try:
                brutos = [b.get("texto_viral_bruto") or "" for b in json.loads(s["resposta_bruta"] or "{}").get("boatos", [])]
            except (json.JSONDecodeError, AttributeError):
                pass
            lp = [limpar_viral(v) for v in brutos if v.strip()]
            lin["bruto_itens"] = len(lp)
            lin["bruto_prefixo_checador"] = any(l["prefixos"] for l in lp)
            lin["bruto_voz_checador"] = any(l["checador"] for l in lp)
            lin["bruto_degenerado"] = any(l["degenerada"] for l in lp)
            lin["bruto_veredito"] = any(any(p in v.lower() for p in w.PALAVRAS_VEREDITO) for v in brutos)
            lin["descartado_pela_limpeza"] = bool(lp) and not virais
        linhas.append(lin)

    res = {"n_fila": len(fila), "n_saida": len(saidas), "por_via": {}}
    for via in ("todas", "html", "texto"):
        ls = [l for l in linhas if via == "todas" or l["via"] == via]
        cs = [l for l in ls if l["com_saida"]]
        rend = [l for l in cs if l.get("rendeu")]
        sims = [l["similaridade"] for l in rend]
        bloco = {
            "M1_rendimento_llm": pct(len(rend), len(cs)),
            "M2_literalidade_llm": pct(sum(l["llm_literal"] for l in rend), len(rend)),
            "M2_literalidade_det_mesmos": pct(sum(l["det_literal"] for l in rend), len(rend)),
            "M3_igual": pct(sum(s >= IGUAL for s in sims), len(cs)),
            "M3_parcial": pct(sum(PARCIAL <= s < IGUAL for s in sims), len(cs)),
            "M3_diferente": pct(sum(s < PARCIAL for s in sims), len(cs)),
            "M3_sem_viral": pct(len(cs) - len(rend), len(cs)),
            "M3_similaridade_mediana": round(sorted(sims)[len(sims) // 2], 3) if sims else None,
            "M4_bruto_prefixo_checador": pct(sum(l["bruto_prefixo_checador"] for l in cs), len(cs)),
            "M4_bruto_voz_checador": pct(sum(l["bruto_voz_checador"] for l in cs), len(cs)),
            "M4_bruto_degenerado": pct(sum(l["bruto_degenerado"] for l in cs), len(cs)),
            "M4_bruto_veredito": pct(sum(l["bruto_veredito"] for l in cs), len(cs)),
            "M4_descartado_pela_limpeza": pct(sum(l["descartado_pela_limpeza"] for l in cs), len(cs)),
            "M5_titulo_copiado": pct(sum(l.get("titulo_copiado", False) for l in rend), len(rend)),
            "viral_det_dentro_do_corte": pct(sum(l["viral_no_corte"] for l in cs), len(cs)),
        }
        # McNemar pareado: literalidade det × llm nos artigos em que o LLM rendeu.
        b = sum(l["det_literal"] and not l["llm_literal"] for l in rend)
        c = sum(l["llm_literal"] and not l["det_literal"] for l in rend)
        bloco["M2_mcnemar"] = {"so_det": b, "so_llm": c,
                               "p": binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0}
        # Sensibilidade: só os artigos cujo viral cabe no trecho que o LLM lê.
        dentro = [l for l in cs if l["viral_no_corte"]]
        bloco["sens_M1_rendimento_viral_no_corte"] = pct(sum(bool(l.get("rendeu")) for l in dentro), len(dentro))
        bloco["sens_M3_igual_viral_no_corte"] = pct(sum(l.get("similaridade", 0) >= IGUAL for l in dentro), len(dentro))
        res["por_via"][via] = bloco
        print(f"\n== {via} (n com saída = {len(cs)})")
        for k, v in bloco.items():
            print(f"  {k:38s} {v}")

    with open(ARQ_RESULTADO, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nsalvo em {ARQ_RESULTADO}")


if __name__ == "__main__":
    if "--preparar" in sys.argv:
        preparar()
    elif "--rodar" in sys.argv:
        rodar()
    elif "--apurar" in sys.argv:
        apurar()
    else:
        print(__doc__)
