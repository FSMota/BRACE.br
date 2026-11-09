# -*- coding: utf-8 -*-
"""Gera o apêndice do TCC com a matriz de modelos completa, direto dos resultados.

    python scripts/tabela_matriz_apendice.py /caminho/do/TCC/Apendices/ApendiceB.tex

Lê relatorios/matriz/apuracao.json e os resumos da matriz (versão dos experimentos).
Nenhum valor é digitado à mão. Confere as faixas citadas no capítulo de resultados.
"""
import os
import sys
import json

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASTA = os.path.join(_RAIZ, "relatorios", "matriz")
FORA = ("Bc_NB", "Ol_NB")                        # falharam no controle com rótulos embaralhados
ORDEM = [("P_LR", "TF-IDF de palavras + LR"), ("P_SVM", "TF-IDF de palavras + SVM"), ("P_NB", "TF-IDF de palavras + NB"),
         ("Ch_LR", "TF-IDF de caracteres + LR"), ("Ch_SVM", "TF-IDF de caracteres + SVM"), ("Ch_NB", "TF-IDF de caracteres + NB"),
         ("Bc_LR", "BERTimbau congelado + LR"), ("Bc_SVM", "BERTimbau congelado + SVM"), ("Bc_NB", "BERTimbau congelado + NB$^{*}$"),
         ("Ol_LR", "\\texttt{bge-m3} + LR"), ("Ol_SVM", "\\texttt{bge-m3} + SVM"), ("Ol_NB", "\\texttt{bge-m3} + NB$^{*}$"),
         ("Ba", "BERTimbau com ajuste fino")]


def br(v):
    return "---" if v is None else "$" + f"{v:.2f}".replace(".", "{,}") + "$"


def main(saida):
    pm = json.load(open(os.path.join(PASTA, "apuracao.json"), encoding="utf-8"))["por_modelo"]
    resumo = {}
    for parte in ("resumo_tfidf", "resumo_densa"):
        for k, v in json.load(open(os.path.join(PASTA, parte + ".json"), encoding="utf-8")).items():
            if "|" in k:
                modelo, braco = k.split("|")
                resumo[(modelo.replace("+", "_"), braco)] = v
    assert sorted(pm) == sorted(k for k, _ in ORDEM)
    linhas = []
    for k, nome in ORDEM:
        m = pm[k]
        fb_b = resumo.get((k, "bruto"), {}).get("fakebr_fakebr")
        fb_r = resumo.get((k, "refinado"), {}).get("fakebr_fakebr")
        if k != "Ba":
            for campo, ref in (("nosso_teste", "bruto"), ("nosso_fakebr", "bruto"), ("fakebr_nosso", "bruto")):
                assert abs(resumo[(k, ref)][campo] - m[campo]) < 0.011, (k, campo)
            for campo in ("nosso_teste", "nosso_fakebr", "fakebr_nosso"):
                assert abs(resumo[(k, "refinado")][campo] - m["ref_" + campo]) < 0.011, (k, campo)
        vals = [m["nosso_teste"], m["nosso_fakebr"], fb_b, m["fakebr_nosso"],
                m["ref_nosso_teste"], m["ref_nosso_fakebr"], fb_r, m["ref_fakebr_nosso"]]
        linhas.append(f"{nome} & " + " & ".join(br(v) for v in vals) + " \\\\")
    validos = [k for k, _ in ORDEM if k not in FORA]
    a2 = sorted(round(pm[k]["nosso_teste"] - pm[k]["ref_nosso_teste"], 1) for k in validos)
    a1 = sorted(round(pm[k]["nosso_fakebr"] - pm[k]["ref_nosso_fakebr"], 1) for k in validos)
    print("perda interna (A2):", a2[0], "a", a2[-1], "| perda no Fake.br (A1):", a1[0], "a", a1[-1])
    tex = r"""\chapter{Matriz de modelos}\label{ap:matriz}

A Tabela~\ref{tab:matriz_completa} traz a acurácia balanceada de todos os modelos
da matriz, na versão da base usada nos experimentos, nas duas versões do texto e
nas quatro combinações de treino e teste. As afirmações discutidas na
Seção~\ref{sec:matriz} são conferidas sobre esses valores. Cada coluna indica a
base de treino e a de teste: BB é BRACE.br~$\rightarrow$~BRACE.br, no teste separado
por data; BF é BRACE.br~$\rightarrow$~Fake.br; FF é Fake.br~$\rightarrow$~Fake.br,
por validação cruzada em 5 dobras; e FB é Fake.br~$\rightarrow$~BRACE.br. Os valores
são gerados por \texttt{scripts/tabela\_matriz\_apendice.py}, a partir dos resultados
publicados no repositório do trabalho.

\begin{table}[H]
\centering
\caption{Acurácia balanceada (\%) dos 13 modelos da matriz. LR: regressão logística;
NB: Naive Bayes, multinomial nas representações esparsas e gaussiano nas densas.
$^{*}$ Fora da contagem das afirmações, por falhar no controle com rótulos
embaralhados. O BERTimbau com ajuste fino não tem a combinação FF, porque a validação
cruzada exigiria cinco ajustes finos.}
\label{tab:matriz_completa}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{\textbf{Texto bruto}} & \multicolumn{4}{c}{\textbf{Texto refinado}} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}
\textbf{Modelo} & \textbf{BB} & \textbf{BF} & \textbf{FF} & \textbf{FB} & \textbf{BB} & \textbf{BF} & \textbf{FF} & \textbf{FB} \\
\midrule
""" + "\n".join(linhas) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    open(saida, "w", encoding="utf-8").write(tex)
    print("->", saida)


if __name__ == "__main__":
    main(sys.argv[1])
