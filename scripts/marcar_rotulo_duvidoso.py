# -*- coding: utf-8 -*-
"""Marca as falsas cujo artigo de checagem tem rótulo duvidoso.

    python scripts/marcar_rotulo_duvidoso.py

O rótulo FALSO é herdado do artigo de checagem: toda mensagem extraída de um
artigo do Boatos.org recebe FALSO. Uma varredura dos títulos dos 11.961 artigos
da classe falsa (05/11/2026), filtrada por regra e revisada à mão, separou os
artigos em que esse rótulo é duvidoso, em seis categorias. A lista de títulos
abaixo é o resultado da revisão manual; o script só a aplica.

Marcar, não apagar: a lista fica em `relatorios/marcacoes/rotulo_duvidoso.json`,
como a das checagens de discurso. A decisão de retirar essas entradas da base
fica para a próxima versão publicada.

Exploratório, não pré-registrado: a varredura foi feita depois dos experimentos,
a partir de uma pergunta da revisão do TCC.
"""
import os
import json
from collections import Counter

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_CORPUS = os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl")
ARQ_SAIDA = os.path.join(_RAIZ, "relatorios", "marcacoes", "rotulo_duvidoso.json")

# categoria -> (descrição, trechos do título do artigo de checagem)
CATEGORIAS = {
    "nao_e_checagem": (
        "conteúdo patrocinado ou de serviço publicado no formato de checagem",
        ["Desvendando mitos sobre ganhos rápidos em cassinos online", "Tokens culturais são fórmulas mágicas",
         "Plataformas modulares e microserviços de jogos", "Empresário brasileiro teve uma vitória milionária em Cassino",
         "Comissão de Valores Mobiliários aprovou ETF de Bitcoin", "Steve Kerr já definiu", "Cupom da Vez"]),
    "verdadeira_ou_nao_verificada": (
        "o título indica que a mensagem é verdadeira, foi corrigida pelo checador ou ainda está em apuração",
        ["Bloquinhos de concreto são realmente", "Há, de fato, um sinal com as mãos", "[Erramos] Caso da cerveja",
         "[Atualizado] Silvio Santos foi internado", "Em apuração: Gael", "Kelvin, de dois anos, está desaparecido"]),
    "desatualizada": (
        "mensagem verdadeira quando circulou, desmentida por estar desatualizada",
        ["bebê que nasceu sem pernas e braços no Ceará já morreu", "garotas filipinas que pisotearam cão já foram identificadas",
         "Homem filmado agredindo menino de 3 anos na Malásia já está preso", "Desaparecimento de garota Kamille ocorreu em 2021",
         "Programa da Capes de intercâmbio para Moçambique já foi concluído", "Kit Kat é feito de óleo de palma é desatualizada"]),
    "meia_verdade": (
        "fato real com erro, exagero ou conclusão falsa",
        ["é real, mas não foi eleita", "É real que estudante cearense", "Electrolux vai demitir 3 mil",
         "Estudo japonês eliminou cromossomo", "Exagerado: israelenses", "Governo não recusou “toda” ajuda do Uruguai",
         "Lei prevê que voto valha como prova de vida", "Lula falou sobre PIB e feriados", "Marcola obteve alvará",
         "Mensagens antecipam escalação de Endrick", "naftalina como", "Projeto Mãos Solidárias existe",
         "Vídeo de ônibus é de saída temporária", "foi, de fato, gravado no museu de Toronto",
         "Zagallo foi internado, mas não morreu", "É fato que Hino Nacional", "É possível que Nubank tenha que trocar",
         "Caso do menino Arthur com câncer é real", "Corrente que fala de conselhos de Diretor do Hospital de Pequim",
         "Friboi já vendia osso de patinho"]),
    "artigo_misto": (
        "artigo que trata de várias alegações, verdadeiras e falsas, ou responde perguntas sobre boatos",
        ["Greve dos caminhoneiros 2015", "Anitta é bi?", "Evaristo Costa é gay?", "Marina Silva é evangélica?",
         "Michel Temer é satanista?", "Neymar é encontrado morto?", "Afinal, o que aconteceu na história de Dilma",
         "Atualizado: Dilma e o cachorro sacrificado", "Schumacher está com 44 kg"]),
    "resumo_de_desinformacao": (
        "resumo de várias desinformações de um período; as mensagens tendem a ser falsas",
        ["Debate na Band e pesquisas eleitorais ditam tom", "Primeira semana de campanha eleitoral teve ataques",
         "Nova regra do WhatsApp gera onda de desinformação", "Novamente, onda de frio vira pauta",
         "No mundo da desinformação, Brasil já está em estado de sítio"]),
}
# categorias em que o rótulo pode estar errado (as outras são decisões defensáveis)
RISCO = ("nao_e_checagem", "verdadeira_ou_nao_verificada", "artigo_misto")


def categoria(titulo):
    for cat, (_, trechos) in CATEGORIAS.items():
        if any(t in titulo for t in trechos):
            return cat
    return None


def main():
    itens, achados = [], Counter()
    for l in open(ARQ_CORPUS, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        if r["veredito"] != "FALSO":
            continue
        titulo = r.get("titulo_artigo_checagem") or ""
        cat = categoria(titulo)
        if cat:
            achados[cat] += 1
            itens.append({"id": r["id"], "categoria": cat, "risco": cat in RISCO, "fonte": r["fonte"],
                          "titulo_checagem": titulo, "split": r["split"],
                          "subconjunto_pareado": r["subconjunto_pareado"], "url": r.get("url")})
    for cat, (_, trechos) in CATEGORIAS.items():
        faltam = [t for t in trechos if not any(t in i["titulo_checagem"] for i in itens)]
        assert not faltam, f"trechos sem artigo na categoria {cat}: {faltam}"
    os.makedirs(os.path.dirname(ARQ_SAIDA), exist_ok=True)
    with open(ARQ_SAIDA, "w", encoding="utf-8") as f:
        json.dump({"marcacao": "rotulo_duvidoso",
                   "categorias": {c: d for c, (d, _) in CATEGORIAS.items()},
                   "categorias_de_risco": list(RISCO), "n": len(itens),
                   "por_categoria": dict(achados),
                   "no_pareado": sum(i["subconjunto_pareado"] for i in itens),
                   "itens": itens}, f, indent=1, ensure_ascii=False)
    print(f"{len(itens)} instâncias marcadas ({sum(i['subconjunto_pareado'] for i in itens)} no pareado) -> {ARQ_SAIDA}")
    for c, n in achados.most_common():
        print(f"  {c:30s} {n}")


if __name__ == "__main__":
    main()
