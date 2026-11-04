# -*- coding: utf-8 -*-
"""Sonda os portais em busca de checagens com veredito VERDADEIRO.

Por que existe: o lado FALSO do corpus é mensagem viral transcrita por checador,
mediana ~250 caracteres, registro de corrente. Parear isso com notícia de veículo
profissional confunde veracidade com GÊNERO -- e a divisão temporal que o corpus
agora tem tornaria o descasamento pior ainda, porque o GDELT com timespan=3m
traria só 2026 contra uma classe falsa de mediana 2021.

Checagem com veredito verdadeiro resolve os três de uma vez: mesma forma
(alegação transcrita), mesmo período, mesmo gênero editorial.

A pergunta que esta sonda responde é se existe VOLUME. Checadores raramente
verificam o que é verdade, então o núcleo limpo pode render centenas em vez de
milhares -- e isso muda o desenho do experimento, não só o tamanho da base.

Não coleta e não altera nada: baixa uma amostra, classifica e relata. Roda em
paralelo com a GPU, porque só usa rede.

    python scripts/sondar_verdadeiras.py
"""
import os
import re
import json
import time
import requests
from collections import Counter
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
N_AMOSTRA = 25          # artigos amostrados por portal
PAUSA = 1.0
_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAIDA = os.path.join(_RAIZ, "relatorios", "coleta", "sonda_verdadeiras.json")

# Os seletores são CHUTE INICIAL de propósito: quando não casam, a sonda despeja
# a estrutura de links agrupada por tag/classe do pai, que é o que permite
# descobrir o certo. Foi assim que a Lupa foi corrigida da primeira vez.
PORTAIS = [
    {"nome": "G1 Fato ou Fake", "listagem": "https://g1.globo.com/fato-ou-fake/",
     "seletor_link": "a.feed-post-link", "pagina": "{base}index/feed/pagina-{n}.ghtml"},
    {"nome": "Lupa", "listagem": "https://lupa.uol.com.br/",
     "seletor_link": "article a", "pagina": None},
    {"nome": "AFP Checamos", "listagem": "https://checamos.afp.com/",
     "seletor_link": "article a, h3 a", "pagina": "{base}?page={n}"},
    {"nome": "Projeto Comprova", "listagem": "https://checamos.projetocomprova.com.br/",
     "seletor_link": "article a, h2 a", "pagina": "{base}page/{n}/"},
    {"nome": "Estadão Verifica", "listagem": "https://www.estadao.com.br/estadao-verifica/",
     "seletor_link": "article a, h3 a", "pagina": None},
    {"nome": "Aos Fatos", "listagem": "https://www.aosfatos.org/noticias/",
     "seletor_link": "article a", "pagina": None},
]

# Rótulos de veredito. Cada portal usa o seu, então a lista é a união -- e a
# classificação exige rótulo EXPLÍCITO, não inferência: "é fato que" no meio de
# um texto não é veredito, é conectivo.
ROTULOS = {
    "VERDADEIRO": [r"#\s?fato\b", r"\b[ée]\s+fato\b", r"\bverdadeiro\b", r"\bcomprovado\b",
                   r"\bconfirmado\s+pel", r"\bverdade\b"],
    "FALSO": [r"#\s?fake\b", r"\bfalso\b", r"\bfalsa\b", r"\bboato\b", r"\benganoso\b",
              r"\bdistorcido\b", r"\bsatíric", r"\binsustentável\b", r"\bfake\s+news\b"],
}
# Marcas de que o portal TRANSCREVE a alegação original -- sem isso o registro
# não é comparável ao lado falso, por mais correto que o veredito seja.
TRANSCREVE = re.compile(
    r"\bleia\s+(?:a\s+)?(?:mensagem|o\s+texto|a\s+publica)"
    r"|\bconfira\s+(?:a\s+mensagem|o\s+texto|a\s+publica)"
    r"|\bcircula\s+(?:nas\s+redes|no\s+whatsapp)"
    r"|\bpublica[çc][õo]es\s+nas\s+redes"
    r"|\bcorrente\s+(?:de|no)\s+whatsapp", re.I)


def baixar(url, timeout=20):
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
        return r.text if r.status_code == 200 else None
    except Exception:
        return None


def diagnosticar(soup, seletor):
    """Descreve os links da página para permitir consertar o seletor."""
    grupos, exemplos = Counter(), {}
    for a in soup.find_all("a", href=True):
        txt = a.get_text(" ", strip=True)
        if len(txt) < 25:
            continue
        pai = a.find_parent()
        chave = (pai.name if pai else "?",
                 " ".join(pai.get("class", [])) if pai else "",
                 " ".join(a.get("class", [])))
        grupos[chave] += 1
        exemplos.setdefault(chave, txt[:70])
    print(f"   [X] seletor {seletor!r} não casou. Candidatos:")
    linhas = []
    for (pt, pc, ac), n in grupos.most_common(6):
        print(f"       <{pt}> pai_classe={pc[:34]!r} link_classe={ac[:24]!r} ({n})")
        print(f"           ex: {exemplos[(pt, pc, ac)]}")
        linhas.append({"pai_tag": pt, "pai_classe": pc, "classe_link": ac, "qtd": n,
                       "exemplo": exemplos[(pt, pc, ac)]})
    return linhas


def classificar_veredito(titulo, corpo):
    """Veredito a partir do rótulo explícito, com o TÍTULO tendo prioridade.

    A ordem importa: o corpo de uma checagem verdadeira quase sempre menciona a
    palavra 'falso' ao descrever o que outros diziam. Decidir pelo corpo
    classificaria como FALSO quase tudo."""
    for fonte, peso in ((titulo, 2), (corpo[:900], 1)):
        baixo = (fonte or "").lower()
        v = sum(1 for p in ROTULOS["VERDADEIRO"] if re.search(p, baixo))
        f = sum(1 for p in ROTULOS["FALSO"] if re.search(p, baixo))
        if v and not f:
            return "VERDADEIRO", peso
        if f and not v:
            return "FALSO", peso
        if v and f:
            return "MISTO", peso
    return "INDEFINIDO", 0


def sondar(portal):
    print("\n" + "=" * 74)
    print(f"PORTAL: {portal['nome']}")
    print("=" * 74)
    rel = {"portal": portal["nome"], "listagem": portal["listagem"],
           "diagnostico": None, "artigos": [], "resumo": {}}

    html = baixar(portal["listagem"])
    if not html:
        print("   [X] listagem inacessível (bloqueio, timeout ou 404).")
        rel["resumo"] = {"erro": "listagem_inacessivel"}
        return rel

    soup = BeautifulSoup(html, "html.parser")
    achados = soup.select(portal["seletor_link"])
    if not achados:
        rel["diagnostico"] = diagnosticar(soup, portal["seletor_link"])
        rel["resumo"] = {"erro": "seletor_nao_casou"}
        return rel

    urls = []
    for t in achados:
        href = t.get("href", "")
        if not href:
            continue
        if not href.startswith("http"):
            href = requests.compat.urljoin(portal["listagem"], href)
        if href not in urls and portal["listagem"].split("/")[2] in href:
            urls.append(href)
    urls = urls[:N_AMOSTRA]
    print(f"   {len(urls)} artigos na amostra.\n")

    vereditos, transcritos = Counter(), 0
    for i, url in enumerate(urls, 1):
        art = baixar(url)
        if not art:
            continue
        s = BeautifulSoup(art, "html.parser")
        for x in s(["script", "style", "nav", "footer", "header", "aside", "form"]):
            x.decompose()
        titulo = (s.find("h1").get_text(" ", strip=True) if s.find("h1") else "")
        corpo = s.get_text(" ", strip=True)
        ver, conf = classificar_veredito(titulo, corpo)
        bq = len(s.find_all("blockquote"))
        tem_transc = bool(bq) or bool(TRANSCREVE.search(corpo))
        if tem_transc:
            transcritos += 1
        vereditos[ver] += 1
        rel["artigos"].append({"url": url, "titulo": titulo[:120], "veredito": ver,
                               "confianca": conf, "blockquotes": bq,
                               "transcreve": tem_transc})
        if ver == "VERDADEIRO" and sum(1 for a in rel["artigos"]
                                       if a["veredito"] == "VERDADEIRO") <= 3:
            print(f"   [VERDADEIRO] {titulo[:88]}")
            print(f"       transcreve={tem_transc} blockquotes={bq}")
        time.sleep(PAUSA)

    n = sum(vereditos.values())
    v = vereditos["VERDADEIRO"]
    rel["resumo"] = {"amostrados": n, **dict(vereditos),
                     "taxa_verdadeiro": round(100 * v / n, 1) if n else 0,
                     "taxa_transcreve": round(100 * transcritos / n, 1) if n else 0}
    print(f"\n   RESUMO {portal['nome']}:")
    print(f"      amostrados ........ {n}")
    print(f"      vereditos ......... {dict(vereditos)}")
    print(f"      taxa VERDADEIRO ... {rel['resumo']['taxa_verdadeiro']}%")
    print(f"      transcrevem ....... {rel['resumo']['taxa_transcreve']}%")
    return rel


def main():
    print("Sonda de checagens VERDADEIRAS. Somente leitura; nada é coletado.")
    print(f"Amostra de até {N_AMOSTRA} artigos por portal.\n")
    resultados = [sondar(p) for p in PORTAIS]

    os.makedirs(os.path.dirname(SAIDA), exist_ok=True)
    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 74)
    print("  VIABILIDADE DO NÚCLEO LIMPO")
    print("=" * 74)
    print(f"  {'portal':22s} {'amostra':>8s} {'%VERD':>7s} {'%transcreve':>12s}  situação")
    viaveis = []
    for r in resultados:
        s = r["resumo"]
        if s.get("erro"):
            print(f"  {r['portal']:22s} {'—':>8s} {'—':>7s} {'—':>12s}  {s['erro']}")
            continue
        # Só serve o portal que dá as duas coisas: veredito verdadeiro E
        # transcrição da alegação. Veredito sem transcrição não é comparável ao
        # lado falso; transcrição sem veredito verdadeiro não ajuda aqui.
        ok = s["taxa_verdadeiro"] >= 10 and s["taxa_transcreve"] >= 50
        print(f"  {r['portal']:22s} {s['amostrados']:>8d} {s['taxa_verdadeiro']:>6.1f}% "
              f"{s['taxa_transcreve']:>11.1f}%  {'VIÁVEL' if ok else 'fraco'}")
        if ok:
            viaveis.append(r["portal"])
    print("\n  portais viáveis:", ", ".join(viaveis) if viaveis else "nenhum na amostra")
    print(f"\n  Detalhe em {SAIDA}")
    print("\n  Envie este resumo: com ele eu dimensiono o núcleo limpo e decido")
    print("  quanto o GDELT precisa completar no braço de escala.")


if __name__ == "__main__":
    main()
