# -*- coding: utf-8 -*-
"""Gera o pacote de publicação da BRACE.br (Zenodo, DOI 10.5281/zenodo.23932000).

    python scripts/exportar_publicacao.py             # simula: mostra contagens, não grava
    python scripts/exportar_publicacao.py --aplicar   # grava em datasets/publicacao/

O que entra e o que não entra:
  - só instâncias da BRACE.br: as 7.200 linhas do Fake.br do braco_refinado.jsonl
    ficam de fora (base de terceiros, usada só para avaliação);
  - sem o texto integral do artigo de checagem (texto_bruto_jornalista) nem o
    recorte literal da apuração (justificativa_bruta): são conteúdo do portal;
  - ideia_central é a versão cega (cego-v2) do braco_refinado.jsonl, a usada nos
    experimentos; o corpus fundido ainda guarda a versão antiga;
  - as 111 entradas inventadas pelo LLM (relatorios/marcacoes/entrada_inventada.json)
    saem do arquivo;
  - subconjunto_pareado exclui também as checagens de discurso; o pareado usado no
    TCC fica em subconjunto_pareado_tcc, para reprodução;
  - registros_no_artigo (falsas) e as marcações de qualidade vão como atributos.
"""
import os
import sys
import json
import gzip
import hashlib
from collections import Counter

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_BRACO = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
PASTA_MARCAS = os.path.join(_RAIZ, "relatorios", "marcacoes")
SAIDA = os.path.join(_RAIZ, "datasets", "publicacao")
VERSAO = "1.0.0"
DOI = "10.5281/zenodo.23932000"
ARQ_DADOS = f"brace_br_v{VERSAO}.jsonl.gz"
SEMENTE = 0

CAMPOS = ["id", "veredito", "texto_entrada", "ideia_central", "justificativa", "palavras_chave",
          "url", "fonte", "data_publicacao", "tema", "tema_pareamento", "split",
          "subconjunto_pareado", "subconjunto_pareado_tcc", "origem_viral", "verificacao_viral",
          "registros_no_artigo", "duplicata_canonica", "grupo_duplicata", "entrada_degenerada",
          "entrada_repetida", "entrada_com_veredito", "entrada_narrativa_checador",
          "entrada_cabecalho_checador", "checagem_de_discurso", "desmentido"]
PROIBIDOS = {"texto_bruto_jornalista", "justificativa_bruta"}


def ids_marcados(nome):
    return {i["id"] for i in json.load(open(os.path.join(PASTA_MARCAS, f"{nome}.json"), encoding="utf-8"))["itens"]}


def montar():
    corpus = [json.loads(l) for l in open(ARQ_CORPUS, encoding="utf-8") if l.strip()]
    cego = {}
    for l in open(ARQ_BRACO, encoding="utf-8"):
        r = json.loads(l)
        if r["split"] != "fakebr":
            cego[r["id"]] = r["ideia_central"]
    inventadas, discurso, desmentido = ids_marcados("entrada_inventada"), ids_marcados("checagem_de_discurso"), ids_marcados("desmentido")
    registros = Counter((r.get("url") or "").rstrip("/") for r in corpus if r["veredito"] == "FALSO")

    saida, retiradas = [], 0
    for r in corpus:
        if r["id"] in inventadas:
            retiradas += 1
            continue
        x = {c: r.get(c) for c in CAMPOS}
        x["ideia_central"] = cego.get(r["id"])                # versão cega; None fora do pareado
        x["subconjunto_pareado_tcc"] = bool(r.get("subconjunto_pareado"))
        x["checagem_de_discurso"] = r["id"] in discurso
        x["desmentido"] = r["id"] in desmentido
        x["subconjunto_pareado"] = x["subconjunto_pareado_tcc"] and not x["checagem_de_discurso"]
        x["registros_no_artigo"] = registros[(r.get("url") or "").rstrip("/")] if r["veredito"] == "FALSO" else None
        assert not PROIBIDOS & set(x)
        saida.append(x)
    reequilibrar(saida)
    return corpus, saida, retiradas, cego


def celula(x):
    return (x.get("tema_pareamento"), (x.get("data_publicacao") or "")[:4], x.get("split"))


def reequilibrar(saida):
    """Depois das retiradas, devolve o equilíbrio de cada célula tema × ano × parte do
    pareado, tirando do lado maior um sorteio reproduzível (semente fixa)."""
    import random
    por = {}
    for x in saida:
        if x["subconjunto_pareado"]:
            por.setdefault(celula(x), {"FALSO": [], "VERDADEIRO": []})[x["veredito"]].append(x)
    rng = random.Random(SEMENTE)
    for chave in sorted(por, key=str):
        f, v = por[chave]["FALSO"], por[chave]["VERDADEIRO"]
        maior, menor = (v, f) if len(v) > len(f) else (f, v)
        for x in rng.sample(sorted(maior, key=lambda y: y["id"]), len(maior) - len(menor)):
            x["subconjunto_pareado"] = False


def resumo(corpus, saida, retiradas, cego):
    v = Counter(x["veredito"] for x in saida)
    par = [x for x in saida if x["subconjunto_pareado"]]
    par_tcc = [x for x in corpus if x.get("subconjunto_pareado")]
    res = {
        "corpus_original": len(corpus),
        "inventadas_retiradas": retiradas,
        "publicadas": len(saida),
        "por_classe": dict(v),
        "pareado_tcc": dict(Counter(x["veredito"] for x in par_tcc)),
        "pareado_publicado": dict(Counter(x["veredito"] for x in par)),
        "pareado_publicado_por_split": dict(Counter((x["split"], x["veredito"]) for x in par)),
        "ideia_central_cega_no_pareado": sum(x["ideia_central"] is not None for x in par),
        "ideia_central_cega_total": len(cego),
        "checagens_de_discurso": sum(x["checagem_de_discurso"] for x in saida),
        "desmentidos_marcados": sum(x["desmentido"] for x in saida),
    }
    # conferências contra os números do TCC
    assert res["corpus_original"] == 28002, res["corpus_original"]
    assert res["pareado_tcc"] == {"FALSO": 11975, "VERDADEIRO": 11975}, res["pareado_tcc"]
    assert res["inventadas_retiradas"] == 111
    assert len(cego) == 23949
    return res


LEIAME = """# BRACE.br v{versao} — Boatos Reunidos de Agências de Checagem, em Escala

Base de desinformação em português, construída automaticamente a partir de
portais de checagem (classe FALSO) e da Agência Brasil (classe VERDADEIRO).

- DOI: https://doi.org/{doi}
- Código e documentação: https://github.com/FSMota/BRACE.br
- Autor: Filipe Simões Mota (UFAL). Orientação: Profa. Dra. Roberta Vilhena
  Vieira Lopes e Prof. Me. Lucas Lisboa (UFAL).

## Conteúdo

`{arquivo}`: uma instância por linha (JSON Lines, gzip).

| | instâncias |
|---|---:|
| total | {publicadas} ({falsas} falsas + {verdadeiras} verdadeiras) |
| subconjunto pareado | {par_total} ({par_f} falsas + {par_v} verdadeiras) |

## Campos

- `id`, `veredito` (`FALSO` ou `VERDADEIRO`).
- `texto_entrada`: a entrada bruta. FALSO: a mensagem viral, como transcrita pelo
  checador, limpa da introdução do jornalista. VERDADEIRO: título + lide da matéria.
- `ideia_central`: o texto refinado, uma frase neutra com a alegação central,
  gerada por LLM local (qwen3.5:9b) com um único prompt cego à classe (`cego-v2`).
  Presente nas instâncias do pareado usado no TCC.
- `justificativa`: resumo, gerado por LLM, das evidências com que o checador
  desmentiu a alegação (só FALSO). **Não use como entrada de um classificador de
  veracidade**: só existe numa classe e entrega o rótulo.
- `palavras_chave`: termos escolhidos pelo LLM.
- `url`, `fonte`, `data_publicacao`: procedência. Nas falsas, a URL é a do artigo
  de checagem, e a data é a da checagem, não a da circulação da mensagem.
- `tema`, `tema_pareamento`, `split` (`treino` até 25/07/2023, `validacao` até
  05/08/2024, `teste` a partir de 06/08/2024).
- `subconjunto_pareado`: as duas classes com o mesmo número de instâncias em
  cada célula tema × ano × parte da divisão, sem as checagens de discurso nem
  as entradas inventadas; nas células afetadas por essas retiradas, o lado
  maior foi reduzido por sorteio com semente fixa. `subconjunto_pareado_tcc`:
  o pareado exato dos experimentos do TCC, para reprodução.
- `origem_viral` (`html`, `texto` ou `llm`), `verificacao_viral`.
- `registros_no_artigo`: quantas instâncias falsas vieram do mesmo artigo de
  checagem. Artigos com vários registros têm mais entradas que não são a
  mensagem (11,1% contra 2,9% na anotação manual).
- Marcações (nada foi apagado por regra): `duplicata_canonica`,
  `grupo_duplicata`, `entrada_degenerada`, `entrada_repetida`,
  `entrada_com_veredito`, `entrada_narrativa_checador`,
  `entrada_cabecalho_checador`, `checagem_de_discurso`, `desmentido`.

## O que foi retirado

- 111 entradas falsas da via LLM que não existem no artigo de origem (texto
  inventado pelo modelo), identificadas por medida pré-registrada.
- O texto integral dos artigos de checagem e os recortes literais da apuração.

## Limitações medidas

- As verdadeiras são matérias de agência, e não mensagens virais: o gênero dos
  textos separa as classes e é o atalho que os modelos captam primeiro.
- 99,6% das falsas vêm do Boatos.org.
- Leitura humana de 200 falsas: 94,5% são a mensagem que circulou.

## Licenças e uso

- VERDADEIRO: textos da Agência Brasil, CC BY 3.0 BR; cite a fonte (`url`).
- FALSO: mensagens transcritas pelos portais de checagem, disponibilizadas
  apenas para pesquisa, com o link para cada artigo de checagem (`url`).

## Como citar

MOTA, Filipe Simões. BRACE.br: uma base de desinformação em português, construída
automaticamente e auditada. Versão {versao}. Zenodo, 2026. DOI: {doi}.
"""


def main():
    corpus, saida, retiradas, cego = montar()
    res = resumo(corpus, saida, retiradas, cego)
    print(json.dumps({k: (v if not isinstance(v, dict) else {str(a): b for a, b in v.items()}) for k, v in res.items()},
                     indent=1, ensure_ascii=False))
    if "--aplicar" not in sys.argv:
        print("\nsimulação: nada gravado (use --aplicar)")
        return
    os.makedirs(SAIDA, exist_ok=True)
    caminho = os.path.join(SAIDA, ARQ_DADOS)
    with gzip.open(caminho, "wt", encoding="utf-8") as f:
        for x in saida:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    par = res["pareado_publicado"]
    leiame = LEIAME.format(versao=VERSAO, doi=DOI, arquivo=ARQ_DADOS, publicadas=res["publicadas"],
                           falsas=res["por_classe"]["FALSO"], verdadeiras=res["por_classe"]["VERDADEIRO"],
                           par_total=sum(par.values()), par_f=par["FALSO"], par_v=par["VERDADEIRO"])
    open(os.path.join(SAIDA, "LEIAME.md"), "w", encoding="utf-8").write(leiame)
    linhas = []
    for nome in (ARQ_DADOS, "LEIAME.md"):
        h = hashlib.sha256(open(os.path.join(SAIDA, nome), "rb").read()).hexdigest()
        linhas.append(f"{h}  {nome}")
    open(os.path.join(SAIDA, "SHA256SUMS"), "w").write("\n".join(linhas) + "\n")
    print("\ngravado em", SAIDA)
    for nome in (ARQ_DADOS, "LEIAME.md", "SHA256SUMS"):
        print(f"  {nome}  {os.path.getsize(os.path.join(SAIDA, nome)) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
