# -*- coding: utf-8 -*-
"""Marca as falsas cuja entrada é o DESMENTIDO, e não o boato.
Pré-registro: `relatorios/PREREGISTRO_DESMENTIDO.md`.

    python scripts/marcar_desmentido.py              # grava a marcação
    python scripts/marcar_desmentido.py --dev        # mostra os acertos fora da amostra (desenvolvimento)

Um artigo de checagem vira um registro por bloco de mensagem, e às vezes o
bloco não traz o boato, e sim o que o desmente: a nota de uma empresa ou órgão
("a empresa esclarece que não está realizando…"), a fala de quem foi citado
("há um texto circulando que não é meu"), ou o próprio texto do checador em
artigos de lista ("É falso que…", "6 fake news sobre…"). Esses registros
entraram como FALSO. Marcar, não apagar: a lista fica em
`relatorios/marcacoes/desmentido.json`, e a marcação vai para o meta da
anotação das 200 falsas, que mede a precisão e a cobertura dela.

A regra foi escrita olhando SÓ artigos fora da amostra de anotação (nenhum
registro de um artigo sorteado foi lido), e é congelada no commit do
pré-registro.
"""
import os
import re
import sys
import json
from collections import Counter

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_META = os.path.join(_RAIZ, "relatorios", "anotacao", "amostra_meta.json")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "marcacoes", "desmentido.json")

# Três tipos de desmentido, cada um com o seu padrão. Os padrões exigem a
# NEGAÇÃO junto da voz oficial ou do próprio citado: o boato também usa "nota
# oficial", "comunicado" e "é falso" ("a Globo diz que é mentira"), e esses
# casos não podem ser marcados.
PADROES = {
    # Nota de empresa ou órgão: voz institucional + negação explícita.
    "nota_oficial": re.compile(
        r"\b(esclarece|esclarecemos|informa(?:mos)?|vem a p[úu]blico|nota (?:oficial|de esclarecimento|[àa] imprensa)"
        r"|comunicado oficial|assessoria de imprensa)\b"
        r".{0,200}?"
        r"\b(n[ãa]o procede|n[ãa]o [ée] verdade|[ée] falsa?|n[ãa]o (?:est[áa]|estamos|est[ãa]o) (?:realizando|promovendo|oferecendo|distribuindo)"
        r"|n[ãa]o (?:haver[áa]|houve|existe)|n[ãa]o (?:[ée]|s[ãa]o) (?:de )?(?:sua|nossa)? ?titularidade|n[ãa]o tem (?:qualquer|nenhuma) rela[çc][ãa]o)",
        re.I | re.S),
    # Fala de quem foi citado, negando a autoria ou o fato.
    "nega_autoria": re.compile(
        r"\b(n[ãa]o [ée] (?:meu|minha) (?:texto|voz|[áa]udio|v[íi]deo|mensagem)|n[ãa]o [ée] minha a voz"
        r"|(?:texto|[áa]udio|v[íi]deo|mensagem) que n[ãa]o [ée] (?:meu|minha)|n[ãa]o [ée] de minha autoria"
        r"|nunca (?:disse|escrevi|falei|afirmei) (?:isso|nada|essa|esse|tal)|escrito por mim"
        r"|fizeram uma montagem|est[ãa]o usando (?:meu nome|minha imagem)|n[ãa]o sou eu\b[.!,])",
        re.I),
    # Texto do checador em artigos de lista ou de cobertura ao vivo.
    "texto_checador": re.compile(
        r"^\s*(?:\d+\s*(?:h|h\d{2})?\s*[–-]\s*)?(?:[ée] falso que|[ée] #?fake que|#?fake[:!]|aos fatos\s*[–-]|boato\s*[–-]"
        r"|\d+\s+(?:fake news|boatos|alertas falsos|mentiras|not[íi]cias falsas)\b)",
        re.I),
}


def tipo_desmentido(texto):
    for tipo, p in PADROES.items():
        if p.search(texto or ""):
            return tipo
    return None


def main():
    meta = json.load(open(ARQ_META, encoding="utf-8"))
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    falsas = [r for r in corpus if r["veredito"] == "FALSO"]
    registros = Counter(r["url"] for r in falsas)
    marcados = []
    for r in falsas:
        t = tipo_desmentido(r["texto_entrada"])
        if t:
            marcados.append({"id": r["id"], "tipo": t, "registros_no_artigo": registros[r["url"]],
                             "split": r["split"], "subconjunto_pareado": r["subconjunto_pareado"], "url": r["url"]})

    if "--dev" in sys.argv:
        # Desenvolvimento: só artigos fora da amostra de anotação.
        url_amostra = {r["url"] for r in falsas if r["id"] in meta["itens"]}
        textos = {r["id"]: r["texto_entrada"] for r in falsas}
        dev = [m for m in marcados if m["url"] not in url_amostra]
        print(f"marcados fora da amostra: {len(dev)} | por tipo {dict(Counter(m['tipo'] for m in dev))}")
        for m in dev:
            print(f"[{m['tipo']}|{m['registros_no_artigo']}] {textos[m['id']][:180]!r}")
        return

    os.makedirs(os.path.dirname(ARQ_SAIDA), exist_ok=True)
    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        json.dump({"marcacao": "desmentido", "n": len(marcados),
                   "por_tipo": dict(Counter(m["tipo"] for m in marcados)),
                   "pareado": sum(m["subconjunto_pareado"] for m in marcados),
                   "de_artigo_com_2_ou_mais": sum(m["registros_no_artigo"] > 1 for m in marcados),
                   "itens": marcados}, f, indent=1, ensure_ascii=False)
    # A marcação vai para o meta da anotação, escondida do anotador.
    ids = {m["id"]: m["tipo"] for m in marcados}
    for i, item in meta["itens"].items():
        item["flag_desmentido"] = ids.get(i)
    with open(ARQ_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"{len(marcados)} falsas marcadas ({sum(m['subconjunto_pareado'] for m in marcados)} no pareado) -> {ARQ_SAIDA}")
    print(f"  por tipo: {dict(Counter(m['tipo'] for m in marcados))}")
    print(f"  na amostra de anotação: {sum(1 for i in meta['itens'] if i in ids)} (só a contagem; o meta não é mostrado ao anotador)")


if __name__ == "__main__":
    main()
