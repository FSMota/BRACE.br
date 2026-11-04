# -*- coding: utf-8 -*-
"""Auditoria do corpus -- diagnostica os defeitos POR TIPO, sem alterar nada.

Por que existe: o log do `main_nlp()` reporta quantos registros gravou, não se
eles servem. Os três defeitos mais caros deste projeto (vazamento de rótulo,
tema inflado, duplicação de viral) não apareceram em nenhum log -- só em medição
sobre a base inteira. Este script é essa medição, empacotada.

É SOMENTE LEITURA. Pode rodar com o pipeline em execução: se pegar o arquivo no
meio de uma gravação, avisa e pede para repetir, em vez de estourar.

Uso:
    python scripts/auditar_corpus.py

Saídas:
    relatorios/AUDITORIA_CORPUS.md     -- relatório legível
    relatorios/auditoria_afetados.json -- URLs por tipo de defeito, para o
                                          script de correção consumir depois
"""
import os
import re
import sys
import json
import time
import random
import statistics
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# Caminhos resolvidos a partir do arquivo, não do diretório de trabalho: o
# projeto foi reorganizado em scripts/ e relatorios/, e caminho relativo a cwd
# já criou dataset vazio uma vez.
# --------------------------------------------------------------------------
_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA_DADOS = os.path.join(_RAIZ, "datasets")
ARQ_BRUTO = os.path.join(PASTA_DADOS, "dataset_bruto.json")
ARQ_FINAL = os.path.join(PASTA_DADOS, "dataset_tcc_fake_news_final.json")
PASTA_REL = os.path.join(_RAIZ, "relatorios")
SAIDA_MD = os.path.join(PASTA_REL, "AUDITORIA_CORPUS.md")
SAIDA_JSON = os.path.join(PASTA_REL, "auditoria_afetados.json")

AMOSTRA_DUPLICATAS = 1500     # O(n^2) -- amostrar é obrigatório acima disso
LIMIAR_JACCARD = 0.70
SEED = 0

# --------------------------------------------------------------------------
# Vocabulário de checagem.
#
# Duas listas, e a distinção é o ponto mais importante deste arquivo:
#
#   TERMOS_SOLTOS  -- "boato", "fake news". Podem aparecer legitimamente DENTRO
#                     da mensagem viral: o próprio autor do boato escreve "estão
#                     rolando boatos". Contar isso como vazamento superestima.
#
#   ATRIBUTIVOS    -- "é falso que", "trata-se de um boato", "desmentido".
#                     São construções que EMITEM veredito sobre a alegação. Quem
#                     escreve assim é o checador, não quem espalha. É esta lista
#                     que mede vazamento de verdade.
# --------------------------------------------------------------------------
TERMOS_SOLTOS = ["boato", "fake news", "checagem", "desinformação", "inverídic"]

ATRIBUTIVOS = re.compile(
    r"\b[ée]\s+falso\s+que\b"
    r"|\b[ée]\s+fals[ao]\b"
    r"|\btrata-se\s+de\s+um\s+boato\b"
    r"|\bn[ãa]o\s+[ée]\s+verdade\s+que\b"
    r"|\bdesmentid[oa]s?\b"
    r"|\bchecagem\s+do\b"
    r"|\bboatos\.org\b"
    r"|\bfato\s+ou\s+fake\b"
    r"|\bn[ãa]o\s+procede\b"
    r"|\bclassificamos\s+como\b"
    r"|\bao\s+contr[áa]rio\s+do\s+que\b",
    re.I)

# Marcadores que o Boatos.org usa para INTRODUZIR a transcrição da mensagem.
# Servem de âncora de INÍCIO -- e a ausência de âncora de FIM equivalente é
# exatamente a causa do defeito do método 'texto'.
MARCADORES_INICIO = re.compile(
    r"\bleia\s*(a\s+mensagem)?\s*:"
    r"|\bconfira\s*(o\s+texto|a\s+mensagem|abaixo)?\s*:"
    r"|\bveja\s*(o\s+texto|a\s+mensagem)?\s*:"
    r"|\bmensagem\s+(que\s+circula|circulante)\s*:"
    r"|\btexto\s+(que\s+circula|circulante)\s*:"
    r"|\btranscri[çc][ãa]o\s*:",
    re.I)

# Candidatos a âncora de FIM: onde o jornalista retoma a palavra depois da
# transcrição. A auditoria mede a cobertura de cada um para o conserto poder
# escolher com dado, não com palpite.
CANDIDATOS_FIM = {
    "Análise": r"\bAn[áa]lise\b",
    "Checagem": r"\bChecagem\b(?!\s+do\s+Boatos)",
    "Conclusão": r"\bConclus[ãa]o\b",
    "(Fim da mensagem)": r"\(fim\s+da\s+mensagem\)|fim\s+da\s+mensagem",
    "Sobre o boato": r"\bsobre\s+(o\s+boato|essa\s+hist[óo]ria)\b",
    "atributivo": ATRIBUTIVOS.pattern,
}

# Marcadores de suporte declarado na alegação (campo que ainda não existe).
SUPORTE = {
    "video": r"\bv[íi]deos?\b|\bgrava[çc][ãa]o\b|\bfilmagem\b|\blive\b",
    "imagem": r"\bfotos?\b|\bimagens?\b|\bprints?\b|\bfotografias?\b|\bmontagem\b|\bcartaz\b",
    "audio": r"\b[áa]udios?\b|\bmensagem\s+de\s+voz\b",
    "texto": r"\bcorrente\b|\bmensagem\b|\btexto\b|\bcarta\b|\blista\b",
}


# ==========================================================================
# utilidades
# ==========================================================================
def carregar(caminho, tentativas=3):
    """Lê JSON tolerando gravação concorrente do pipeline.

    O `_salvar()` do main_nlp escreve por cima do arquivo sem gravação atômica,
    então há uma janela de fração de segundo em que o JSON está truncado. Em vez
    de estourar no meio da auditoria, espera e tenta de novo."""
    for n in range(tentativas):
        try:
            with open(caminho, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            if n < tentativas - 1:
                print(f"   [aguardando] {os.path.basename(caminho)} está sendo gravado; "
                      f"nova tentativa em 5s...")
                time.sleep(5)
            else:
                print(f"\nERRO: não consegui ler {caminho} de forma consistente.")
                print("O pipeline provavelmente está gravando. Rode de novo em alguns segundos,")
                print("ou espere a linha '[salvo]' aparecer no log do main_nlp.")
                sys.exit(1)
        except FileNotFoundError:
            print(f"\nERRO: {caminho} não encontrado.")
            sys.exit(1)


def shingles(texto, k=5):
    palavras = re.findall(r"\w+", (texto or "").lower())
    return {tuple(palavras[i:i + k]) for i in range(max(0, len(palavras) - k + 1))}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def mil(n):
    """Separador de milhar no padrão brasileiro. Existe como função porque
    fazer .replace(',', '.') na frase inteira também troca as vírgulas do texto
    -- foi o que aconteceu na primeira versão."""
    return "{:,}".format(n).replace(",", ".")


def pct(n, total):
    return f"{100 * n / total:.1f}%" if total else "-"


def resumo_numerico(valores):
    if not valores:
        return {}
    q = statistics.quantiles(valores, n=10) if len(valores) >= 10 else None
    return {"n": len(valores),
            "mediana": round(statistics.median(valores)),
            "p10": round(q[0]) if q else min(valores),
            "p90": round(q[8]) if q else max(valores),
            "max": max(valores)}


# ==========================================================================
# seções da auditoria
# ==========================================================================
def secao_inventario(final, bruto, L):
    L("## 1. Inventário\n")
    n = len(final)
    L(f"- Instâncias no arquivo final: **{mil(n)}**")
    L(f"- Artigos-fonte distintos: {mil(len(set(r.get('url_origem_fake') for r in final)))}"
      )
    L(f"- Registros por artigo (média): {n / max(1, len(set(r.get('url_origem_fake') for r in final))):.2f}")
    for campo in ("veredito", "idioma", "fonte_fake", "modelo_refinamento"):
        c = Counter(r.get(campo) for r in final).most_common(4)
        L(f"- `{campo}`: " + ", ".join(f"{k} {mil(v)}" for k, v in c))

    # Fila restante -- diz se a auditoria está olhando um retrato parcial.
    falta = sum(1 for x in bruto if not x.get("processado_pela_ia"))
    L(f"\n- **Fila restante no `dataset_bruto`: {mil(falta)} artigos**")
    if falta:
        L(f"  > Retrato PARCIAL. A ~8,6 s/artigo faltam ~{falta * 8.6 / 3600:.1f} h de GPU.")
        L("  > As proporções abaixo são estáveis; os totais absolutos vão crescer.")
    else:
        L("  > Fila vazia: o refinamento terminou. Números definitivos.")

    L("\n### Cobertura de campos\n")
    L("| campo | preenchidos | % |")
    L("|---|---:|---:|")
    for campo in ("texto_viral_bruto", "claim_refinada", "ideia_central", "justificativa",
                  "justificativa_bruta", "palavras_chave", "tema"):
        k = sum(1 for r in final if r.get(campo))
        L(f"| `{campo}` | {mil(k)} | {pct(k, n)} |")
    return {"n": n, "fila_restante": falta}


def secao_proveniencia(final, L):
    L("\n## 2. Proveniência do texto viral\n")
    L("O `texto_viral_bruto` é a entrada de treino do lado FALSO. Ele vem de três")
    L("extratores diferentes, e as colunas abaixo dizem se eles produzem a mesma coisa.\n")
    L("| método | n | mediana | p90 | literal | veredito atributivo |")
    L("|---|---:|---:|---:|---:|---:|")
    por_metodo = {}
    for m in ("html", "html_sem_indice", "texto", "llm"):
        s = [r for r in final if r.get("origem_viral") == m]
        if not s:
            continue
        tam = [len(r.get("texto_viral_bruto") or "") for r in s]
        lit = sum(1 for r in s if r.get("verificacao_viral") == "conferido")
        atr = sum(1 for r in s if ATRIBUTIVOS.search(r.get("texto_viral_bruto") or ""))
        rn = resumo_numerico(tam)
        por_metodo[m] = {"n": len(s), **rn, "literal": lit, "atributivo": atr}
        L(f"| `{m}` | {mil(len(s))} | {rn['mediana']} | {rn['p90']} | {pct(lit, len(s))} "
          f"| **{pct(atr, len(s))}** |")
    L("\n> `literal` do método `texto` é **tautológico**: o trecho foi obtido procurando")
    L("> texto DENTRO do artigo, então conferi-lo no artigo sempre passa. Não leia como qualidade.")
    return por_metodo


def secao_defeito_overcapture(final, L, afetados):
    L("\n## 3. Defeito A — extrator textual invadindo a análise\n")
    s = [r for r in final if r.get("origem_viral") == "texto"]
    if not s:
        L("Nenhum registro do método `texto`. Nada a tratar.")
        return {}
    atr = [r for r in s if ATRIBUTIVOS.search(r.get("texto_viral_bruto") or "")]
    L(f"- Registros do método `texto`: **{mil(len(s))}**")
    L(f"- Com construção atributiva de veredito no viral: **{mil(len(atr))} ({pct(len(atr), len(s))})**"
      )

    # Onde a marca de veredito cai dentro do trecho? Se cai no COMEÇO, o
    # extrator começou cedo demais (pegou a introdução do jornalista). Se cai no
    # FIM, ele terminou tarde demais (entrou na análise). A distinção decide o
    # conserto.
    posicoes = []
    for r in atr:
        t = r["texto_viral_bruto"]
        m = ATRIBUTIVOS.search(t)
        posicoes.append(m.start() / max(1, len(t)))
    if posicoes:
        inicio = sum(1 for p in posicoes if p < 0.33)
        meio = sum(1 for p in posicoes if 0.33 <= p < 0.66)
        fim = sum(1 for p in posicoes if p >= 0.66)
        L(f"\n**Onde a marca de veredito aparece dentro do trecho extraído:**\n")
        L("| posição | n | % | leitura |")
        L("|---|---:|---:|---|")
        L(f"| terço inicial | {inicio} | {pct(inicio, len(posicoes))} | extrator começou cedo demais |")
        L(f"| terço central | {meio} | {pct(meio, len(posicoes))} | trecho mistura os dois |")
        L(f"| terço final | {fim} | {pct(fim, len(posicoes))} | extrator terminou tarde demais |")

    # Cobertura dos candidatos a âncora de FIM: qual marcador serviria para
    # cortar o trecho no lugar certo.
    L(f"\n**Âncoras de fim disponíveis nesses {mil(len(s))} registros:**\n")
    L("| candidato | presente no viral | posição mediana |")
    L("|---|---:|---:|")
    for nome, padrao in CANDIDATOS_FIM.items():
        rx = re.compile(padrao, re.I)
        achados = []
        for r in s:
            t = r.get("texto_viral_bruto") or ""
            m = rx.search(t)
            if m:
                achados.append(m.start() / max(1, len(t)))
        if achados:
            L(f"| {nome} | {mil(len(achados))} ({pct(len(achados), len(s))}) "
              f"| {statistics.median(achados):.0%} do trecho |")
        else:
            L(f"| {nome} | 0 | — |")

    inicio_ok = sum(1 for r in s if MARCADORES_INICIO.search(r.get("texto_viral_bruto") or ""))
    # O .replace de separador de milhar só pode incidir sobre o número, nunca
    # sobre a frase -- senão vírgulas do texto viram pontos.
    L(f'\n- Registros que ainda carregam o marcador de INÍCIO dentro do trecho '
      f'("Leia:", "Confira:"): {mil(inicio_ok)} ({pct(inicio_ok, len(s))})')
    L("  > Se o marcador de início está DENTRO do trecho, ele deveria ter sido o ponto de corte.")

    afetados["A_overcapture_texto"] = [r["url_origem_fake"] for r in atr]
    L(f"\n**Tratamento:** passada determinística sobre `dataset_bruto` (artigo salvo), "
      f"reancorando o fim. Sem GPU.")
    return {"n": len(s), "contaminados": len(atr)}


def secao_defeito_justificativa(final, L, afetados):
    L("\n## 4. Defeito B — a alegação dentro da evidência\n")
    L("O `justificativa_bruta` deveria conter só a apuração. Quando ele contém também")
    L("a mensagem viral, deixa de servir como padrão-ouro de evidência (RQ3.2): a")
    L("alegação está dentro da própria evidência que deveria comprová-la ou refutá-la.\n")
    s = [r for r in final if r.get("justificativa_bruta") and r.get("texto_viral_bruto")]
    linhas = []
    contaminados = []
    for m in ("html", "texto", "llm"):
        sub = [r for r in s if r.get("origem_viral") == m]
        if not sub:
            continue
        fr = []
        for r in sub:
            a = shingles(r["texto_viral_bruto"])
            b = shingles(r["justificativa_bruta"])
            f = len(a & b) / len(a) if a else 0.0
            fr.append(f)
            if f > 0.50:
                contaminados.append(r["url_origem_fake"])
        alto = sum(1 for x in fr if x > 0.50)
        linhas.append((m, len(sub), alto, statistics.median(fr)))
    L("| método | n | viral contido na evidência (>50%) | % | sobreposição mediana |")
    L("|---|---:|---:|---:|---:|")
    for m, n, alto, med in linhas:
        L(f"| `{m}` | {mil(n)} | {mil(alto)} | **{pct(alto, n)}** | {med:.2f} |")
    afetados["B_alegacao_na_evidencia"] = contaminados
    L(f"\n**Causa:** `_fechar_justificativa()` só remove o viral quando "
      f"`texto_viral_estrutural` existe — e nos artigos sem `<blockquote>` ele é `None`.")
    L("**Tratamento:** passada determinística. Sem GPU.")
    return {"contaminados": len(contaminados)}


def secao_defeito_comprimento(final, por_metodo, L):
    L("\n## 5. Defeito C — heterogeneidade de comprimento dentro da classe\n")
    L("Comprimento já é o confundidor clássico entre classes. Aqui ele aparece **dentro**")
    L("da classe FALSO, correlacionado com o método de extração — o que significa que um")
    L("classificador pode separar registros por extrator em vez de por conteúdo.\n")
    medianas = {m: v["mediana"] for m, v in por_metodo.items() if v.get("mediana")}
    if len(medianas) >= 2:
        mx, mn = max(medianas.values()), min(medianas.values())
        L(f"- Mediana máxima / mínima entre métodos: **{mx} / {mn} = {mx / max(1, mn):.1f}x**")
    v = [len(r.get("texto_viral_bruto") or "") for r in final]
    rn = resumo_numerico(v)
    L(f"- Distribuição geral do viral: p10 {rn['p10']} · mediana {rn['mediana']} · p90 {rn['p90']} · máx {rn['max']}")
    curtos = sum(1 for x in v if x < 100)
    L(f"- Virais com menos de 100 caracteres: {mil(curtos)} ({pct(curtos, len(v))}) "
      f"— curtos demais para carregar sinal")
    teto = sum(1 for x in v if x >= 5000)
    if teto:
        L(f"- Virais no teto de 5.000 caracteres: {teto} — truncados, não completos")
    L("\n**Tratamento:** decorre do Defeito A. Remedir depois de reancorar.")


def secao_duplicatas(final, L, afetados):
    L("\n## 6. Duplicatas\n")
    exatas = Counter((r.get("texto_viral_bruto") or "").strip() for r in final)
    excedente = sum(n - 1 for n in exatas.values() if n > 1)
    L(f"- Virais idênticos (excedente): **{mil(excedente)} ({pct(excedente, len(final))})** "
      f"— distintos: {mil(len(exatas))}")
    piores = [(t[:70], n) for t, n in exatas.most_common(5) if n > 1]
    if piores:
        L("\n| viral repetido | vezes |")
        L("|---|---:|")
        for t, n in piores:
            L(f"| {t.replace('|', '/')}… | {n} |")

    random.seed(SEED)
    am = random.sample(final, min(AMOSTRA_DUPLICATAS, len(final)))
    S = [(r, shingles(r.get("texto_viral_bruto") or "", k=3)) for r in am]
    quase = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if jaccard(S[i][1], S[j][1]) > LIMIAR_JACCARD:
                quase.add(S[i][0]["url_origem_fake"])
                quase.add(S[j][0]["url_origem_fake"])
    L(f"\n- Quase-duplicatas (Jaccard > {LIMIAR_JACCARD}, amostra de {mil(len(am))}): "
      f"**{len(quase)} ({pct(len(quase), len(am))})**")
    L("\n> Quase-duplicata é vazamento treino/teste: o mesmo boato reciclado cai dos dois")
    L("> lados da divisão e infla a acurácia. Deduplicar ANTES de dividir, e dividir por")
    L("> tempo, não aleatoriamente.")
    afetados["C_quase_duplicatas_amostra"] = sorted(quase)


def secao_vazamento(final, L, afetados):
    L("\n## 7. Vazamento de rótulo na entrada de treino\n")
    n = len(final)
    atr = [r for r in final if ATRIBUTIVOS.search(r.get("texto_viral_bruto") or "")]
    solto = [r for r in final
             if any(p in (r.get("texto_viral_bruto") or "").lower() for p in TERMOS_SOLTOS)]
    L(f"- **Construção atributiva de veredito** (assinatura do checador): "
      f"**{mil(len(atr))} ({pct(len(atr), n)})**")
    L(f"- Termo solto (\"boato\", \"fake news\") em qualquer contexto: "
      f"{mil(len(solto))} ({pct(len(solto), n)})")
    L("\n> A segunda linha **superestima**: o próprio autor do boato escreve \"estão rolando")
    L("> boatos\". Só a primeira mede vazamento de fato. A diferença entre as duas é a razão")
    L("> de a lista de palavras proibidas do LIAR-PLUS não bastar sozinha.")
    L(f"\n- Vazamento na `justificativa` (não é entrada, mas contamina a RQ3.1): "
      f"{mil(sum(1 for r in final if ATRIBUTIVOS.search(r.get('justificativa') or '')))}"
      )
    afetados["D_vazamento_atributivo"] = [r["url_origem_fake"] for r in atr]


def secao_claim(final, L):
    L("\n## 8. A `claim_refinada` herdou a contaminação?\n")
    L("Esta é a seção que decide se o conserto custa minutos ou uma noite. A")
    L("`claim_refinada` do método `texto` foi gerada a partir de um bloco que já continha")
    L("prosa do jornalista. Reancorar o viral conserta a entrada crua — **não** conserta o")
    L("resumo derivado dela. Se as taxas abaixo forem parecidas entre os métodos, o resumo")
    L("resistiu e basta a passada determinística. Se o método `texto` destoar, são")
    L("~N registros para refinar de novo na GPU.\n")
    L("| método | n | atributivo na claim | termo solto | mediana (chars) |")
    L("|---|---:|---:|---:|---:|")
    for m in ("html", "texto", "llm"):
        s = [r for r in final if r.get("origem_viral") == m]
        if not s:
            continue
        a = sum(1 for r in s if ATRIBUTIVOS.search(r.get("claim_refinada") or ""))
        t = sum(1 for r in s if any(p in (r.get("claim_refinada") or "").lower()
                                    for p in TERMOS_SOLTOS))
        med = round(statistics.median([len(r.get("claim_refinada") or "") for r in s]))
        L(f"| `{m}` | {mil(len(s))} | **{pct(a, len(s))}** | {pct(t, len(s))} | {med} |"
          )
    L("\n**Critério de decisão:** se `texto` tiver taxa atributiva mais que o dobro dos")
    L("outros dois, refinar de novo. Caso contrário, manter e só documentar.")


def secao_tema(final, L):
    L("\n## 9. Tema\n")
    n = len(final)
    L("| tema | n | % |")
    L("|---|---:|---:|")
    for t, k in Counter(r.get("tema") for r in final).most_common():
        L(f"| {t} | {mil(k)} | {pct(k, n)} |")
    origem = Counter(r.get("tema_origem") for r in final)
    L(f"\n- Origem: " + ", ".join(f"`{k}` {mil(v)} ({pct(v, n)})"
                                  for k, v in origem.most_common()))
    com_url = [r for r in final if r.get("tema_origem") == "url"]
    if com_url:
        ac = sum(1 for r in com_url if r.get("tema") == r.get("tema_llm"))
        L(f"- **Concordância categoria editorial × LLM: {mil(ac)}/{mil(len(com_url))} "
          f"({pct(ac, len(com_url))})**")
        L("\n> Este número é resultado da RQ2, não um problema a corrigir: mede o quanto o")
        L("> modelo diverge de um rótulo humano quando ambos veem o mesmo artigo.")


def secao_suporte(final, bruto, L):
    L("\n## 10. Suporte da mensagem (campo ainda inexistente)\n")
    L("Estimativa a partir da linha-resumo `Boato – <alegação>`, escrita pelo jornalista.")
    L("Não é o campo definitivo — é a medida que diz se vale criá-lo.\n")
    por_url = {x.get("url_origem"): x for x in bruto}
    cont = Counter()
    vistos = set()
    for r in final:
        u = r.get("url_origem_fake")
        if u in vistos:
            continue
        vistos.add(u)
        art = por_url.get(u)
        if not art:
            continue
        texto = art.get("texto_bruto_jornalista") or ""
        m = re.search(r"\bBoato\s*[–—-]\s*", texto)
        if not m:
            cont["(sem linha-resumo)"] += 1
            continue
        resto = texto[m.end():]
        p = re.search(r"(?<=[.!?])\s", resto)
        linha = (resto[:p.start()] if p else resto[:300]).lower()
        achados = [k for k, rx in SUPORTE.items() if re.search(rx, linha)]
        cont["+".join(sorted(achados)) if achados else "nao_declarado"] += 1
    tot = sum(cont.values())
    L("| suporte declarado | artigos | % |")
    L("|---|---:|---:|")
    for k, v in cont.most_common(8):
        L(f"| {k} | {mil(v)} | {pct(v, tot)} |")
    L("\n**Tratamento:** passada determinística, rótulo de cinco estados incluindo")
    L("`nao_declarado`. Precisa de 100 casos anotados à mão para medir precisão e recall")
    L("antes de entrar no TCC.")


def secao_classe_verdadeira(final, L):
    L("\n## 11. Classe VERDADEIRO\n")
    v = Counter(r.get("veredito") for r in final)
    pareados = sum(1 for r in final if r.get("pareado"))
    L(f"- Vereditos presentes: " + ", ".join(f"{k} {mil(n)}" for k, n in v.items()))
    L(f"- Registros marcados como pareados: {mil(pareados)}")
    if "VERDADEIRO" not in v:
        L("\n> **A classe VERDADEIRO não existe.** Sem ela: RQ1 não tem o que comparar, RQ4")
        L("> não tem baseline trivial, e o `relatorio_corpus.py` mantém essas seções inativas.")
        L("> É o item que destrava mais coisa por unidade de esforço.")


def secao_plano(L, inv, ov, jb):
    L("\n## 12. Plano de correção, por tipo\n")
    L("| # | defeito | escopo | custo | bloqueia |")
    L("|---|---|---:|---|---|")
    L(f"| A | viral do método `texto` invadindo a análise | ~{mil(ov.get('contaminados', 0))} reg. "
      f"| determinístico, minutos | RQ1, RQ4 |")
    L(f"| B | alegação dentro do `justificativa_bruta` | ~{mil(jb.get('contaminados', 0))} reg. "
      f"| determinístico, minutos | RQ3.2 |")
    L("| C | heterogeneidade de comprimento | decorre de A | remedir após A | RQ1, RQ4 |")
    L("| D | duplicatas e divisão aleatória | corpus todo | determinístico, minutos | RQ1, RQ4 |")
    L("| E | `claim_refinada` contaminada | ver seção 8 | **GPU, ~uma noite** | RQ1 |")
    L("| F | campo de suporte ausente | corpus todo | determinístico + 100 anotações | descritivo |")
    L("| G | classe VERDADEIRO ausente | coleta nova | GDELT + ~8 h GPU | RQ1, RQ4 |")
    L("| H | prompt do `ideia_central` assimétrico | prompt | edição | RQ1 |")
    L("\n**Ordem recomendada:** A → B → D (determinísticos, encadeados) → decidir E com a")
    L("seção 8 → G (coleta) → F e H em paralelo.")
    if inv.get("fila_restante"):
        L(f"\n> ⚠ Ainda faltam {mil(inv['fila_restante'])} artigos no refinamento. Rode esta"
          )
        L("> auditoria de novo ao final: a fatia do método `texto` tende a crescer, porque o")
        L("> arquivo profundo é justamente onde o template antigo não tem `<blockquote>`.")


# ==========================================================================
def main():
    print("Auditoria do corpus (somente leitura)...\n")
    print("   lendo dataset_tcc_fake_news_final.json ...")
    final = carregar(ARQ_FINAL)
    print("   lendo dataset_bruto.json ...")
    bruto = carregar(ARQ_BRUTO)

    if not final:
        print("Arquivo final vazio. Nada a auditar.")
        return

    linhas = []
    L = linhas.append
    L("# Auditoria do corpus — Fake-Collector-Refiner\n")
    L(f"Gerado em {time.strftime('%d/%m/%Y %H:%M')}. Somente leitura: nenhum arquivo de")
    L("dados foi alterado.\n")

    afetados = {}
    inv = secao_inventario(final, bruto, L)
    por_metodo = secao_proveniencia(final, L)
    ov = secao_defeito_overcapture(final, L, afetados)
    jb = secao_defeito_justificativa(final, L, afetados)
    secao_defeito_comprimento(final, por_metodo, L)
    secao_duplicatas(final, L, afetados)
    secao_vazamento(final, L, afetados)
    secao_claim(final, L)
    secao_tema(final, L)
    secao_suporte(final, bruto, L)
    secao_classe_verdadeira(final, L)
    secao_plano(L, inv, ov, jb)

    os.makedirs(PASTA_REL, exist_ok=True)
    with open(SAIDA_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")
    with open(SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump({k: sorted(set(v)) for k, v in afetados.items()}, f,
                  indent=2, ensure_ascii=False)

    # Resumo no terminal: o essencial, sem obrigar a abrir o arquivo.
    print("\n" + "=" * 74)
    print(f"  instâncias .................. {mil(inv['n'])}")
    print(f"  fila restante ............... {mil(inv['fila_restante'])}")
    for m, v in por_metodo.items():
        print(f"  viral via {m:<16} n={mil(v['n']):>7} mediana={v['mediana']:>5} "
              f"atributivo={100 * v['atributivo'] / v['n']:.1f}%")
    print(f"  defeito A (over-capture) .... {mil(ov.get('contaminados', 0))} registros")
    print(f"  defeito B (alegação na ev.) . {mil(jb.get('contaminados', 0))} registros")
    print("=" * 74)
    print(f"\nRelatório: {SAIDA_MD}")
    print(f"Afetados : {SAIDA_JSON}")


if __name__ == "__main__":
    main()
