# -*- coding: utf-8 -*-
"""Junta a saída do refinamento cego ao corpus, pelo id -- passo 1 da §1.6 do
`relatorios/PREREGISTRO_BRACO_REFINADO.md`.

    python scripts/juntar_refinamento.py          # confere e grava datasets/braco_refinado.jsonl

    # teste com outra saída/destino (não toca nos arquivos de verdade)
    python scripts/juntar_refinamento.py --entrada X.jsonl --destino Y.jsonl --conferencia Z.json

O que o pré-registro permite ANTES de o comparativo estar commitado: conferir o
sha256 da saída, a contagem por grupo, o `prompt_sha` único, as falhas e os
`saida_ajustes` por grupo. Este script faz só isso -- conta, não mostra nenhuma
frase refinada.

Arquivos:
    datasets/refinamento_fila.jsonl     entrada do refinamento (id, grupo, split, texto)
    datasets/refinamento_cego.jsonl     saída do refinamento (id, ideia_central, prompt_sha...)
    datasets/braco_refinado.jsonl       uma linha por id refinado: texto bruto + frase refinada
    relatorios/conferencia_refinamento.json   as contagens desta conferência
"""
import os
import sys
import json
import hashlib
from collections import Counter, defaultdict

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from refinar_cego import ARQ_FILA, ARQ_SAIDA, PROMPT_SHA, PROMPT_VERSAO, ler_jsonl   # noqa: E402

ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_DESTINO = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
ARQ_CONFERENCIA = os.path.join(_RAIZ, "relatorios", "conferencia_refinamento.json")

# Fixados no pré-registro do braço refinado (§1.1).
SHA_FILA = "3533440ab268976dd9d9ef7f71c14fd7bb1590f5c3f7397f230678f43e35220a"
SHA_CORPUS = "901d8373a1f5452c"
GRUPOS = ("corpus_falso", "corpus_verdadeiro", "fakebr_falso", "fakebr_verdadeiro")


def mil(n):
    return "{:,}".format(n).replace(",", ".")


def sha256(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def arg(nome, padrao):
    return sys.argv[sys.argv.index(nome) + 1] if nome in sys.argv else padrao


def categorias_fakebr():
    """id da fila (FB-fake-n / FB-true-n) -> categoria do Fake.br. Vazio se a pasta não existir."""
    try:
        from avaliacao_cruzada import carregar_fakebr
    except Exception:
        return {}
    if not os.path.isdir(os.path.join(_RAIZ, "datasets", "externos", "fakebr")):
        return {}
    pares, _ = carregar_fakebr()
    return {f"FB-{c}-{p['n']}": p[c]["categoria"] for p in pares for c in ("fake", "true")}


def main():
    entrada = arg("--entrada", ARQ_SAIDA)
    destino = arg("--destino", ARQ_DESTINO)
    arq_conf = arg("--conferencia", ARQ_CONFERENCIA)
    problemas = []

    # ---- fila ---------------------------------------------------------------
    sha_fila = sha256(ARQ_FILA)
    print(f"fila  {ARQ_FILA}\n  sha256 {sha_fila}  "
          f"{'OK' if sha_fila == SHA_FILA else '[!] DIFERE do pré-registrado'}")
    if sha_fila != SHA_FILA:
        problemas.append("sha da fila difere do pré-registrado")
    fila = {r["id"]: r for r in ler_jsonl(ARQ_FILA)}

    # ---- saída --------------------------------------------------------------
    if not os.path.exists(entrada):
        sys.exit(f"[!] {entrada} não existe.")
    sha_saida = sha256(entrada)
    brutas = open(entrada, encoding="utf-8").read().splitlines()
    saida, invalidas = [], 0
    for l in brutas:
        if not l.strip():
            continue
        try:
            saida.append(json.loads(l))
        except json.JSONDecodeError:
            invalidas += 1
    print(f"\nsaída {entrada}\n  sha256 {sha_saida}\n  linhas {mil(len(saida))}"
          f" | JSON inválido {invalidas}")

    shas = Counter(r.get("prompt_sha") for r in saida)
    versoes = Counter(r.get("prompt_versao") for r in saida)
    modelos = Counter(r.get("modelo") for r in saida)
    print(f"  prompt_sha {dict(shas)}  {'OK' if set(shas) == {PROMPT_SHA} else '[!] MISTURADO OU DIFERENTE'}")
    print(f"  prompt_versao {dict(versoes)} | modelo {dict(modelos)}")
    if set(shas) != {PROMPT_SHA}:
        problemas.append(f"prompt_sha diferente de {PROMPT_SHA}: {dict(shas)}")

    vistos, repetidos, fora_da_fila = {}, 0, 0
    for r in saida:
        if r["id"] not in fila:
            fora_da_fila += 1
        elif r["id"] in vistos:
            repetidos += 1           # não deveria acontecer (o script pula os feitos); fica o primeiro
        else:
            vistos[r["id"]] = r
    print(f"  ids fora da fila {fora_da_fila} | repetidos {repetidos} (fica a primeira saída)")
    if fora_da_fila:
        problemas.append(f"{fora_da_fila} ids da saída não estão na fila")

    # ---- por grupo ----------------------------------------------------------
    conf = {"fila_sha256": sha_fila, "saida_sha256": sha_saida, "linhas_saida": len(saida),
            "json_invalido": invalidas, "prompt_sha": dict(shas), "prompt_versao": dict(versoes),
            "modelo": dict(modelos), "ids_fora_da_fila": fora_da_fila, "repetidos": repetidos,
            "por_grupo": {}}
    na_fila = Counter(r["grupo"] for r in fila.values())
    feitos = Counter(fila[i]["grupo"] for i in vistos)
    ajustes = defaultdict(Counter)
    com_ajuste = Counter()
    for i, r in vistos.items():
        g = fila[i]["grupo"]
        if r.get("saida_ajustes"):
            com_ajuste[g] += 1
            ajustes[g].update(r["saida_ajustes"])
    print(f"\n  {'grupo':18s} {'fila':>7s} {'refinados':>9s} {'faltam':>7s} {'% ajustada':>10s}  ajustes")
    for g in GRUPOS:
        falta = na_fila[g] - feitos[g]
        pct = 100 * com_ajuste[g] / max(1, feitos[g])
        print(f"  {g:18s} {mil(na_fila[g]):>7s} {mil(feitos[g]):>9s} {mil(falta):>7s} {pct:9.2f}%  "
              f"{dict(ajustes[g])}")
        conf["por_grupo"][g] = {"fila": na_fila[g], "refinados": feitos[g], "faltam": falta,
                                "pct_saida_ajustada": round(pct, 2), "ajustes": dict(ajustes[g])}
    falta_total = len(fila) - len(vistos)
    print(f"  {'total':18s} {mil(len(fila)):>7s} {mil(len(vistos)):>9s} {mil(falta_total):>7s}")
    if falta_total:
        print(f"  [!] {mil(falta_total)} sem saída: saem dos TRÊS braços (§1.3). Se forem muitos, "
              f"rode refinar_cego.py de novo antes -- ele refaz só os pendentes.")

    # ---- corpus -------------------------------------------------------------
    sha_corpus = sha256(ARQ_CORPUS)[:16]
    print(f"\ncorpus sha {sha_corpus}  {'OK' if sha_corpus == SHA_CORPUS else '[!] DIFERE do pré-registrado'}")
    if sha_corpus != SHA_CORPUS:
        problemas.append("sha do corpus difere do pré-registrado")
    corpus = {}
    with open(ARQ_CORPUS, encoding="utf-8") as f:
        for l in f:
            r = json.loads(l)
            if r["subconjunto_pareado"]:
                corpus[r["id"]] = r
    cats = categorias_fakebr()
    if not cats:
        print("  [!] Fake.br não encontrado: o destino sai sem categoria do Fake.br")

    divergentes = 0
    with open(destino + ".tmp", "w", encoding="utf-8") as f:
        for i, s in vistos.items():
            q = fila[i]
            falso = q["grupo"].endswith("_falso")
            linha = {"id": i, "origem": q["origem"], "grupo": q["grupo"], "split": q["split"],
                     "veredito": "FALSO" if falso else "VERDADEIRO",
                     "texto_entrada": q["texto"], "ideia_central": s["ideia_central"],
                     "saida_ajustes": s.get("saida_ajustes", []),
                     "entrada_com_veredito": False, "categoria_fakebr": None}
            if q["origem"] == "corpus":
                c = corpus.get(i)
                if c is None or c["texto_entrada"] != q["texto"] or c["split"] != q["split"] \
                        or c["veredito"] != linha["veredito"]:
                    divergentes += 1           # a fila não bate com o corpus atual
                    continue
                linha["entrada_com_veredito"] = bool(c.get("entrada_com_veredito"))
            else:
                linha["categoria_fakebr"] = cats.get(i)
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")
    os.replace(destino + ".tmp", destino)
    print(f"  registros da fila que não batem com o corpus (texto, split ou veredito): {divergentes}")
    if divergentes:
        problemas.append(f"{divergentes} registros da fila não batem com o corpus")
    conf.update({"corpus_sha": sha_corpus, "divergentes_corpus": divergentes,
                 "gravados": len(vistos) - divergentes, "problemas": problemas})

    os.makedirs(os.path.dirname(arq_conf), exist_ok=True)
    with open(arq_conf, "w", encoding="utf-8") as f:
        json.dump(conf, f, indent=2, ensure_ascii=False)
    print(f"\n-> {mil(len(vistos) - divergentes)} registros em {destino}")
    print(f"-> conferência em {arq_conf}")
    print("\n" + ("[!] PROBLEMAS: " + "; ".join(problemas) if problemas else
                  f"conferência OK (prompt {PROMPT_VERSAO} {PROMPT_SHA}). Próximo: comparativo_bracos.py"))


if __name__ == "__main__":
    main()
