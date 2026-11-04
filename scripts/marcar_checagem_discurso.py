# -*- coding: utf-8 -*-
"""Marca as falsas que vêm de checagens de DISCURSO (G1 Fato ou Fake e Lupa), não de boatos.
Pré-registro: `relatorios/PREREGISTRO_SENSIBILIDADE_DISCURSO.md`.

    python scripts/marcar_checagem_discurso.py

Esses portais checam frases de políticos em entrevistas e debates, com um
veredito por frase (#FATO / #FAKE no G1; "erra", "exagera", "omite contexto" na
Lupa). A extração pela via LLM juntou as frases como mensagens falsas, inclusive
frases verdadeiras e trechos do jornalista. Marcar, não apagar: a lista fica em
`relatorios/marcacoes/checagem_de_discurso.json`; a integração ao
`fundir_corpus.py` fica para a versão publicada.
"""
import os
import json
from collections import Counter

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "marcacoes", "checagem_de_discurso.json")
FONTES = ("G1 Fato ou Fake", "Lupa")
MOTIVO = ("checagem de discurso de político (frase a frase, vereditos mistos ou graduados), não de boato; "
          "extraída pela via llm; pode conter frase verdadeira ou texto do jornalista")


def main():
    itens = []
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        r = json.loads(l)
        if r["veredito"] == "FALSO" and r.get("fonte") in FONTES:
            itens.append({"id": r["id"], "fonte": r["fonte"], "titulo_checagem": r.get("titulo_artigo_checagem"),
                          "split": r["split"], "subconjunto_pareado": r["subconjunto_pareado"],
                          "data_publicacao": r["data_publicacao"], "url": r.get("url")})
    os.makedirs(os.path.dirname(ARQ_SAIDA), exist_ok=True)
    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        json.dump({"marcacao": "checagem_de_discurso", "motivo": MOTIVO, "n": len(itens),
                   "por_fonte": dict(Counter(i["fonte"] for i in itens)),
                   "por_split_pareado": {f"{s}|{p}": n for (s, p), n in
                                         Counter((i["split"], i["subconjunto_pareado"]) for i in itens).items()},
                   "itens": itens}, f, indent=1, ensure_ascii=False)
    print(f"{len(itens)} instâncias marcadas -> {ARQ_SAIDA}")


if __name__ == "__main__":
    main()
