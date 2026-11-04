# -*- coding: utf-8 -*-
"""Sonda v2 -- mede a estrutura HTML das fontes de checagem em AMOSTRA.

O que mudou em relação à v1 (que olhava 1 artigo por fonte):

  1. AMOSTRA de vários artigos por fonte, para medir COBERTURA do <blockquote>
     em vez de confiar num único exemplo.
  2. Quando o seletor da listagem falha (Aos Fatos e Lupa hoje), despeja a
     estrutura de links da página para dar de reparar o seletor.
  3. Classifica cada blockquote como "parece viral" ou "parece fonte
     jornalística" -- distinção que importa porque no G1 os blockquotes são
     declarações de assessoria/especialista, não a mensagem falsa.

Não altera nada: só baixa e relata. Uso:  python sondar_estrutura_fontes.py
"""
import os
import re
import json
import time
import requests
from collections import Counter
from bs4 import BeautifulSoup

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
N_ARTIGOS = 12          # artigos amostrados por fonte
SAIDA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "relatorios", "coleta", "sonda_estrutura.json")

# Fontes candidatas à diversificação. Os seletores das novas são CHUTE INICIAL --
# e isso não é problema: quando o seletor não casa, a sonda despeja a estrutura de
# links da listagem agrupada por tag e classe do elemento pai, que é justamente o
# que permite descobrir o seletor certo. Foi assim que a Lupa foi corrigida.
#
# 'pagina_padrao' descreve como a fonte pagina, para a sonda conseguir amostrar
# além da primeira página. None = não tenta paginar.
FONTES = [
    # --- já em produção, mantidas para comparação ---
    {"nome": "Boatos.org", "listagem": "https://www.boatos.org/",
     "seletor_link": "article h2 a", "pagina_padrao": "{base}page/{n}/"},

    # --- candidata principal: mesmo gênero do Boatos.org (corrente e post viral) ---
    # URLs são planas (ex.: /origem-da-foto-negao-whatsapp-18.html), sem categoria
    # no caminho -- então aqui NÃO dará para derivar tema pela URL como no
    # Boatos.org; o tema virá do LLM.
    {"nome": "E-farsas", "listagem": "https://www.e-farsas.com/",
     "seletor_link": "article h2 a", "pagina_padrao": "{base}page/{n}/"},

    # --- outras candidatas, para saber se transcrevem texto viral ---
    {"nome": "Projeto Comprova", "listagem": "https://projetocomprova.com.br/publicações/",
     "seletor_link": "article a", "pagina_padrao": "{base}page/{n}/"},
    {"nome": "Estadão Verifica", "listagem": "https://www.estadao.com.br/estadao-verifica/",
     "seletor_link": "article a", "pagina_padrao": None},

    # --- já avaliadas: G1 e Lupa não transcrevem viral; Aos Fatos é JS ---
    {"nome": "Lupa", "listagem": "https://www.agencialupa.org/checagem/",
     "seletor_link": "article.feed a.feed-link", "pagina_padrao": None},
]

# Sinais de que um trecho é a MENSAGEM VIRAL (corrente de rede social).
SINAIS_VIRAL = ["whatsapp", "repasse", "compartilhe", "encaminhe", "urgente",
                "atenção", "passe adiante", "grupo", "recebi", "amigos",
                "corre informação", "acabei de receber"]
# Sinais de que é FONTE JORNALÍSTICA (declaração citada pelo checador).
SINAIS_FONTE = ["assessoria", "em nota", "informou", "declarou", "segundo o",
                "afirmou", "disse ao", "comunicado", "confirmou", "esclarece",
                "professor", "especialista", "pesquisador"]


def classificar_bloco(txt):
    """Heurística: a mensagem viral tem marcas de corrente; a fonte
    jornalística tem marcas de citação institucional."""
    baixo = txt.lower()
    v = sum(1 for s in SINAIS_VIRAL if s in baixo)
    f = sum(1 for s in SINAIS_FONTE if s in baixo)
    letras = [c for c in txt if c.isalpha()]
    prop_maiuscula = sum(1 for c in letras if c.isupper()) / len(letras) if letras else 0
    exclamacoes = txt.count("!")

    if prop_maiuscula > 0.30 or exclamacoes >= 3:
        v += 2
    if v > f:
        return "VIRAL?", v, f
    if f > v:
        return "FONTE?", v, f
    return "indef.", v, f


def links_da_listagem(fonte, paginas=2):
    urls, diagnostico = [], None
    for p in range(1, paginas + 1):
        # Paginação declarada por fonte, em vez de if/elif por nome -- assim
        # acrescentar uma fonte nova é só editar a lista FONTES.
        padrao = fonte.get("pagina_padrao")
        if p == 1 or not padrao:
            url = fonte["listagem"]
        else:
            url = padrao.format(base=fonte["listagem"], n=p)

        try:
            html = requests.get(url, headers=UA, timeout=20).text
        except Exception as e:
            diagnostico = f"falha ao baixar a listagem: {type(e).__name__}: {e}"
            break

        soup = BeautifulSoup(html, "html.parser")
        achados = soup.select(fonte["seletor_link"])

        if not achados and diagnostico is None:
            # Seletor quebrado: descreve os links da página para reparo.
            diagnostico = diagnosticar_listagem(soup, fonte)

        for t in achados:
            href = t.get("href", "")
            if not href:
                continue
            if not href.startswith("http"):
                href = requests.compat.urljoin(fonte["listagem"], href)
            if href not in urls:
                urls.append(href)

        if len(urls) >= N_ARTIGOS or not fonte.get("pagina_padrao"):
            break
        time.sleep(1)

    return urls[:N_ARTIGOS], diagnostico


def diagnosticar_listagem(soup, fonte):
    """Descreve a estrutura de links para permitir consertar o seletor."""
    print(f"  [X] seletor {fonte['seletor_link']!r} não casou. Estrutura da listagem:")

    # Agrupa links por (tag do pai, classe do pai) para revelar o padrão do feed.
    grupos = Counter()
    exemplos = {}
    for a in soup.find_all("a", href=True):
        texto = a.get_text(" ", strip=True)
        if len(texto) < 25:           # ignora menu/rodapé
            continue
        pai = a.find_parent()
        chave = (pai.name if pai else "?", " ".join(pai.get("class", [])) if pai else "",
                 " ".join(a.get("class", [])))
        grupos[chave] += 1
        exemplos.setdefault(chave, (texto[:70], a["href"]))

    linhas = []
    print("      pai_tag | pai_classe | classe_do_link | qtd")
    for (ptag, pclasse, aclasse), n in grupos.most_common(8):
        print(f"      <{ptag}> | {pclasse[:38]!r} | {aclasse[:28]!r} | {n}")
        print(f"          ex: {exemplos[(ptag, pclasse, aclasse)][0]}")
        print(f"              {exemplos[(ptag, pclasse, aclasse)][1][:95]}")
        linhas.append({"pai_tag": ptag, "pai_classe": pclasse, "classe_link": aclasse,
                       "qtd": n, "exemplo_texto": exemplos[(ptag, pclasse, aclasse)][0],
                       "exemplo_url": exemplos[(ptag, pclasse, aclasse)][1]})
    return {"seletor_quebrado": fonte["seletor_link"], "candidatos": linhas}


def sondar(fonte):
    print("\n" + "=" * 78)
    print(f"FONTE: {fonte['nome']}")
    print("=" * 78)

    urls, diagnostico = links_da_listagem(fonte)
    relatorio = {"fonte": fonte["nome"], "diagnostico_listagem": diagnostico,
                 "artigos": [], "resumo": {}}

    if not urls:
        print("  (sem artigos para amostrar)")
        return relatorio

    print(f"  amostrando {len(urls)} artigos...\n")
    com_bq = 0
    tamanhos = []
    classes_bq = Counter()

    for idx, url in enumerate(urls, 1):
        try:
            html = requests.get(url, headers=UA, timeout=20).text
        except Exception:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for x in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            x.decompose()

        bqs = soup.find_all("blockquote")
        info = {"url": url, "n_blockquotes": len(bqs), "blocos": []}
        if bqs:
            com_bq += 1

        for bq in bqs:
            txt = bq.get_text(" ", strip=True)
            if len(txt) < 40:
                continue
            classe = " ".join(bq.get("class", [])) or "(sem classe)"
            classes_bq[classe] += 1
            tamanhos.append(len(txt))
            rotulo, v, f = classificar_bloco(txt)
            info["blocos"].append({"classe": classe, "tamanho": len(txt),
                                   "rotulo": rotulo, "sinais_viral": v,
                                   "sinais_fonte": f, "inicio": txt[:180]})
            if idx <= 3:
                print(f"   [art {idx}] <blockquote class={classe[:34]!r}> {len(txt)}c "
                      f"-> {rotulo} (viral={v} fonte={f})")
                print(f"            {txt[:150]}")

        relatorio["artigos"].append(info)

    total = len(relatorio["artigos"])
    todos = [b for a in relatorio["artigos"] for b in a["blocos"]]
    virais = sum(1 for b in todos if b["rotulo"] == "VIRAL?")
    fontes_ = sum(1 for b in todos if b["rotulo"] == "FONTE?")

    relatorio["resumo"] = {
        "artigos_amostrados": total,
        "artigos_com_blockquote": com_bq,
        "cobertura_pct": round(100 * com_bq / total, 1) if total else 0,
        "blocos_totais": len(todos),
        "blocos_parecem_viral": virais,
        "blocos_parecem_fonte": fontes_,
        "tamanho_medio": round(sum(tamanhos) / len(tamanhos)) if tamanhos else 0,
        "classes": dict(classes_bq),
    }

    print(f"\n  RESUMO {fonte['nome']}:")
    print(f"     artigos com <blockquote> : {com_bq}/{total} ({relatorio['resumo']['cobertura_pct']}%)")
    print(f"     blocos: {len(todos)} (parecem viral: {virais} | parecem fonte: {fontes_})")
    print(f"     tamanho médio do bloco   : {relatorio['resumo']['tamanho_medio']} chars")
    print(f"     classes usadas           : {dict(classes_bq)}")
    return relatorio


if __name__ == "__main__":
    print(f"Sonda v2 -- amostrando até {N_ARTIGOS} artigos por fonte.")
    print("(nada é alterado no dataset)")

    resultados = [sondar(f) for f in FONTES]
    for _ in range(1):
        time.sleep(1)

    with open(SAIDA, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78)
    print(f"Relatório salvo em {SAIDA}. Envie o arquivo para eu escrever os extratores.")
