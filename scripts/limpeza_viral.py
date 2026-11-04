# -*- coding: utf-8 -*-
"""Regras de limpeza do texto viral -- compartilhadas pelo pipeline de extração
(web_scraping_limpo.py) e pela fusão (fundir_corpus.py).

Por que um módulo só: as mesmas regras precisam valer na ORIGEM (quando o viral
é extraído) e na FUSÃO (para os registros antigos que não foram reprocessados).
Duas cópias do regex divergiriam na primeira correção.

MEDIDO no corpus (fusão de 29/09/2026), sobre as 15.846 falsas, e presente nas
TRÊS vias de extração -- não só na do LLM:

  introdução do checador ....... 2.415  ("Leia:", "Versão 1:", "Texto:", embed do Instagram...)
  "Versão N" no meio do viral .. 1.095  (várias versões do boato coladas num registro)
  lixo sem boato nenhum ........   308  ("Confira também: ...", embed sem legenda)
"""
import re

# 1. Introdução que o CHECADOR escreve antes de transcrever o boato. Nenhuma
#    notícia verdadeira começa assim, então deixá-la vira atalho ("começa com
#    Leia -> falso"). "URGENTE:", "ATENÇÃO:", "Bom dia" NÃO saem: são o estilo do
#    próprio boato.
PREFIXO_CHECADOR = re.compile(
    r"^\s*(?:"
    r"leia[^:\n]{0,80}:"                          # Leia: / Leia a mensagem: / Leia algumas das versões...:
    r"|vers(?:ão|ao)\s*\d+\s*[:.–-]"              # Versão 1:
    r"|(?:texto|mensagem|transcrição)\s*:"        # Texto: / Mensagem:
    r"|confira o desmentido (?:em|no) v[íi]deo\s*:?"
    r"|confira (?:o texto|a mensagem)[^:\n]{0,40}:"   # Confira o texto circulante:
    r"|boato\s*[–—-]"                             # linha-resumo do checador
    # Embed do Instagram: o cabeçalho sai; a legenda que vem depois, se houver, é o boato.
    r"|ver essa (?:foto|publicação) no instagram\s+uma publicação compartilhada por[^()]*\(@[\w.]+\)"
    r")\s*", re.I)

# 2. Rótulo de versão NO MEIO do texto: "... COMPARTILHE Versão 2: “Memphis...".
#    Só com ":" ou travessão depois do número -- "versão 2.0 do aplicativo" é
#    texto legítimo e não pode ser cortado.
VERSAO_NO_MEIO = re.compile(r"\s*\bvers(?:ão|ao)\s*\d+\s*[:–—-]\s", re.I)

# 3. Lixo: link interno do site ou texto curto demais para ser mensagem.
LIXO = re.compile(
    r"^\s*(?:confira também|assista também|leia mais sobre o boato"
    r"|retrospectiva \d{4} das fake news)", re.I)
MIN_PALAVRAS = 5

# 4. Narração do CHECADOR dentro da entrada: alguém falando SOBRE a mensagem
#    ("diz a mensagem", "de acordo com o texto", "a mensagem em questão tem
#    características de boatos online"). Não dá para cortar por prefixo -- a
#    narração aparece no meio, às vezes com o boato depois. MEDIDO (04/10/2026):
#    372 das 15.450 falsas (2,4%), quase todas da via LLM; 20 verdadeiras
#    (notícias sobre boatos). Achado no piloto do refinamento: com esta entrada,
#    a frase refinada saiu "A mensagem é falsa pois...". Só expressões em que o
#    texto fala de outro texto -- "rumores que circularam" sozinho é o próprio
#    boato e NÃO casa.
NARRATIVA_CHECADOR = re.compile(
    r"\b(?:a mensagem em questão|o texto em questão|o vídeo em questão|a publicação em questão"
    r"|(?:diz|afirma|alega|garante) (?:a mensagem|o texto|a publicação|o post|a corrente)"
    r"|(?:a mensagem|o texto|a publicação|a corrente|o post) (?:diz|afirma|alega|garante|sugere|insinua) que"
    r"|segundo (?:a mensagem|o texto|a publicação|a corrente)"
    r"|de acordo com (?:a mensagem|o texto|a publicação|a corrente)"
    r"|(?:a mensagem|o texto|o boato) que circula"
    r"|que circula(?:m)? (?:online|na internet|nas redes|pelo whatsapp|no whatsapp)"
    r"|como (?:vocês|você) (?:viram|viu)|boatos? online|velho roteiro dos boatos"
    r"|leia (?:o texto|a mensagem|algumas das|as mensagens)[^:]{0,60}:)", re.I)


def narrativa_checador(texto):
    return bool(NARRATIVA_CHECADOR.search(texto or ""))


# 5. Cabeçalho / assinatura do CHECADOR no lugar do boato. MEDIDO (05/10/2026,
#    auditar_vazamento.py): 873 das 12.087 falsas do pareado (7,2%) eram o TÍTULO
#    do artigo de checagem com a etiqueta de veredito do site ("Datena está sendo
#    processado... #boato Datena está sendo processado... #boato Entretenimento
#    01/12/..."), quase todas da via LLM e de artigos antigos; 197 com a
#    assinatura do autor; rodapé "sugestão de leitores". Nenhuma verdadeira tem
#    nenhum dos três. É o rótulo dentro da entrada: o texto inteiro é recusado.
CABECALHO_CHECADOR = re.compile(
    r"#boato\b|\bedgard matsuki\b|\bpor edgard\b|\bhellen bizerra\b"
    r"|sugestão de leitores|quiser sugerir um tema|esse artigo é uma sugestão"
    r"|@?boatos\.org\b", re.I)


def cabecalho_checador(texto):
    return bool(CABECALHO_CHECADOR.search(texto or ""))


def tirar_prefixos(texto):
    """Remove as introduções do checador, repetidamente -- elas se empilham
    ("Leia algumas das versões...: Versão 1: <boato>"). Devolve o texto e os
    prefixos removidos, que ficam no registro para auditoria."""
    texto = (texto or "").replace("﻿", " ").strip()
    removidos = []
    while True:
        m = PREFIXO_CHECADOR.match(texto)
        if not m or not m.group(0).strip():
            return texto, removidos
        removidos.append(m.group(0).strip())
        texto = texto[m.end():].strip()


def primeira_versao(texto):
    """Várias versões do MESMO boato coladas num texto: fica a primeira.
    Instância = alegação checada; as versões são variações dela, e gravá-las
    separadas encheria o corpus de quase-duplicatas. Devolve (texto, n_versoes)."""
    partes = [p.strip() for p in VERSAO_NO_MEIO.split(texto) if p.strip()]
    if len(partes) <= 1:
        return texto, 1
    return partes[0], len(partes)


def degenerada(texto):
    return len(re.findall(r"\w+", texto or "")) < MIN_PALAVRAS or bool(LIXO.match(texto or ""))


def limpar_viral(texto):
    """Aplica as regras. Devolve dict com o texto limpo, os prefixos removidos,
    quantas versões havia, se o que sobrou é degenerado e se é voz do checador
    (cabeçalho/assinatura ou narração) -- nos dois últimos casos o texto NÃO é o
    boato e a extração deve recusá-lo."""
    t, prefixos = tirar_prefixos(texto)
    t, versoes = primeira_versao(t)
    if versoes > 1:                      # a versão 1 pode vir com introdução própria
        t, mais = tirar_prefixos(t)
        prefixos += mais
    return {"texto": t, "prefixos": prefixos, "versoes": versoes, "degenerada": degenerada(t),
            "checador": cabecalho_checador(t) or narrativa_checador(t)}
