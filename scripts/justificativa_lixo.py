# -*- coding: utf-8 -*-
"""Pré-preenche a planilha de recuperação: pares com 0 ou 1 palavra em comum viram nao_relacionada.
Desvio declarado no adendo 1 de `relatorios/PREREGISTRO_RECUPERACAO.md`.

    python scripts/justificativa_lixo.py

Edita a planilha no lugar (mantém a aba de instruções e as listas de opções),
grava "auto: ≤1 palavra em comum" na observação e só mexe em linhas com o
julgamento 1 vazio. Faz uma cópia de segurança antes.
"""
import os
import re
import shutil

from openpyxl import load_workbook

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ = os.path.join(_RAIZ, "relatorios", "recuperacao", "planilha_recuperacao.xlsx")
ARQ_COPIA = ARQ.replace(".xlsx", "_antes_preenchimento.xlsx")
MARCA = "auto: ≤1 palavra em comum"

# Lista simples de stopwords (palavras que não ajudam na semântica)
stopwords = set(['que', 'os', 'as', 'de', 'do', 'da', 'em', 'um', 'uma', 'para', 'com', 'não', 'é', 'na', 'no', 'o', 'a', 'se', 'por', 'como', 'mais', 'ao', 'dos', 'das', 'sua', 'seu', 'ou', 'quando', 'muito', 'nos', 'já', 'está'])


def tem_sobreposicao(consulta, candidato):
    # Extrai palavras maiores que 1 letra e tira as stopwords
    c_words = set([w for w in re.findall(r'\b[a-zà-ÿ]+\b', str(consulta).lower()) if w not in stopwords and len(w) > 1])
    d_words = set([w for w in re.findall(r'\b[a-zà-ÿ]+\b', str(candidato).lower()) if w not in stopwords and len(w) > 1])

    # Retorna o número de palavras em comum
    return len(c_words.intersection(d_words))


def main():
    shutil.copy2(ARQ, ARQ_COPIA)
    wb = load_workbook(ARQ)
    ws = wb["anotacao"]
    col = {c.value: i for i, c in enumerate(ws[1])}
    marcadas = vazias = 0
    for linha in ws.iter_rows(min_row=2):
        if linha[0].value in (None, "") or linha[col["julgamento_1"]].value not in (None, ""):
            continue
        vazias += 1
        if tem_sobreposicao(linha[col["consulta (boato novo)"]].value, linha[col["candidato (checagem antiga)"]].value) <= 1:
            linha[col["julgamento_1"]].value = "nao_relacionada"
            linha[col["observacao_1"]].value = MARCA
            marcadas += 1
    wb.save(ARQ)
    print(f"{marcadas} de {vazias} linhas vazias pré-preenchidas como nao_relacionada; cópia em {ARQ_COPIA}")


if __name__ == "__main__":
    main()
