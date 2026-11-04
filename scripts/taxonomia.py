# -*- coding: utf-8 -*-
"""Taxonomia de duas camadas: categoria original e categoria de pareamento.

Por que duas. As camadas servem a coisas diferentes e misturá-las estraga as
duas:

  `tema` -- granular, herdado da categoria editorial do portal de checagem
  (Celebridades, Religião, Golpes e Fraudes...). É DADO DE PESQUISA: descreve
  que tipo de desinformação circula, e é contra ele que se mede a concordância
  de 48,7% com o LLM, que é resultado da RQ2. Não pode ser sobrescrito.

  `tema_pareamento` -- grosso, compartilhado pelas duas classes. É INSTRUMENTO:
  existe só para equilibrar FALSO e VERDADEIRO por assunto. Precisa ser
  comparável entre lados que vêm de fontes com taxonomias distintas -- portal de
  checagem de um lado, editoria de agência pública do outro.

REGRA QUE NÃO PODE SER QUEBRADA: nos dois lados, `tema_pareamento` deriva de um
rótulo EDITORIAL HUMANO já presente na fonte (categoria na URL do Boatos.org,
editoria na URL da agência). Se um lado fosse rotulado por LLM e o outro por
editor, o campo carregaria a assinatura do método e viraria atalho -- o
classificador separaria as classes pela forma como o tema foi atribuído, não
pelo conteúdo.

LIMITE CONHECIDO, para constar na seção de validade: agrupar não elimina o
descasamento temático, só o esconde num balde maior. Dentro de "Cultura e
Sociedade", o lado FALSO é boato de morte de celebridade e o VERDADEIRO é
política cultural. O vocabulário ainda difere, e isso é atalho residual -- tem
de ser medido junto com os outros na RQ4, não presumido resolvido.
"""

# Conjunto grosso: a interseção viável entre as categorias do Boatos.org e as
# editorias de agência pública. "Educação" existe do lado verdadeiro mas não tem
# demanda do falso (o Boatos.org não usa essa categoria) -- fica no conjunto
# porque não custa nada, e some sozinha quando a demanda for zero.
TEMAS_PAREAMENTO = [
    "Política", "Economia", "Saúde", "Educação", "Justiça e Segurança",
    "Meio Ambiente", "Ciência e Tecnologia", "Cultura e Sociedade",
    "Esportes", "Internacional", "Outros",
]

# --- lado FALSO: categoria do Boatos.org -> balde de pareamento --------------
# Golpes e Fraudes vai para Justiça e Segurança, e não para Economia, porque o
# que o boato explora ali é crime (estelionato, phishing), não conjuntura.
# Celebridades e Religião caem juntas em Cultura e Sociedade: nenhuma agência
# pública tem editoria equivalente, e é o balde mais próximo sem inventar
# correspondência.
MAPA_FALSO = {
    "Política": "Política",
    "Economia": "Economia",
    "Saúde": "Saúde",
    "Ciência e Tecnologia": "Ciência e Tecnologia",
    "Meio Ambiente": "Meio Ambiente",
    "Esportes": "Esportes",
    "Internacional": "Internacional",
    "Segurança Pública": "Justiça e Segurança",
    "Golpes e Fraudes": "Justiça e Segurança",
    "Celebridades": "Cultura e Sociedade",
    "Religião": "Cultura e Sociedade",
    "Outros": "Outros",
}

# --- lado VERDADEIRO: editoria na URL da agência -> balde de pareamento ------
# A chave é o segmento de editoria como aparece na URL, sem acento, que é como
# a agência escreve. Ex.: agenciabrasil.ebc.com.br/meio-ambiente/noticia/2019-03/...
#
# MEDIDO: a Agência Brasil NÃO tem editoria de ciência em ano nenhum. Os slugs
# de Ciência e Tecnologia abaixo ficam porque custam nada e cobrem outras
# fontes, mas na Agência Brasil os quatro candidatos voltaram zerados em
# 2021-2026. Esse balde (1.938 instâncias do lado falso) fica sem contraparte.
MAPA_EDITORIA_ABERTA = {
    "politica": "Política",
    "economia": "Economia",
    "saude": "Saúde",
    "educacao": "Educação",
    "justica": "Justiça e Segurança",
    "seguranca": "Justiça e Segurança",
    "meio-ambiente": "Meio Ambiente",
    "pesquisa-e-inovacao": "Ciência e Tecnologia",
    "pesquisaeinovacao": "Ciência e Tecnologia",     # variante sem hífen, vista no acervo
    "ciencia-e-tecnologia": "Ciência e Tecnologia",
    "ciencia": "Ciência e Tecnologia",
    "tecnologia": "Ciência e Tecnologia",
    "inovacao": "Ciência e Tecnologia",
    "cultura": "Cultura e Sociedade",
    "direitos-humanos": "Cultura e Sociedade",
    "direitoshumanos": "Cultura e Sociedade",        # idem
    "esportes": "Esportes",
    "internacional": "Internacional",
    "geral": "Outros",
}


def pareamento_do_falso(tema):
    """Balde de pareamento a partir da categoria do portal de checagem."""
    return MAPA_FALSO.get(tema, "Outros")


def pareamento_da_editoria(editoria):
    """Balde de pareamento a partir da editoria na URL da agência.

    Devolve None para editoria desconhecida em vez de chutar 'Outros': aqui o
    desconhecido deve ser DESCARTADO, não absorvido. Absorver encheria o balde
    'Outros' do lado verdadeiro com qualquer coisa que a agência publique, e
    'Outros' tem só 31 instâncias de demanda do lado falso."""
    if not editoria:
        return None
    return MAPA_EDITORIA_ABERTA.get(editoria.strip("/").lower())
