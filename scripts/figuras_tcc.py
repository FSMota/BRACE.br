# -*- coding: utf-8 -*-
"""Figuras do capítulo de resultados do TCC, geradas a partir dos JSONs de resultado.

    python scripts/figuras_tcc.py [--saida /mnt/c/Users/filip/TCC-FakeMachine/figuras]

Cada figura lê só arquivos já commitados em `relatorios/`; nenhum número é
digitado à mão. PDFs vetoriais, modo claro (impressão). Paleta categórica de
referência (azul, laranja, verde-água, amarelo), validada para daltonismo;
como dois tons ficam abaixo de 3:1 de contraste, todo valor é rotulado
diretamente e cada série tem também um marcador próprio (funciona em preto e
branco).
"""
import os
import sys
import json
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(_RAIZ, "relatorios")
SAIDA = sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else "/mnt/c/Users/filip/TCC-FakeMachine/figuras"

AZUL, LARANJA, AQUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CINZA, TINTA, TINTA2, GRADE = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1"
LARGURA = 6.3                                                      # polegadas, ~16 cm

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.edgecolor": CINZA, "axes.labelcolor": TINTA2,
    "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
})


def ler(*partes):
    return json.load(open(os.path.join(REL, *partes), encoding="utf-8"))


def salvar(fig, nome):
    os.makedirs(SAIDA, exist_ok=True)
    caminho = os.path.join(SAIDA, nome)
    fig.savefig(caminho, bbox_inches="tight")
    fig.savefig(caminho.replace(".pdf", ".png"), bbox_inches="tight", dpi=160)   # prévia
    plt.close(fig)
    print(f"  {nome}")


def br(x, casas=1):
    return f"{x:.{casas}f}".replace(".", ",")


# --------------------------------------------------------------------------
def fig_base(anos=None, temas=None, nome="fig_base_ano_tema.pdf"):
    """Pareado por ano e por tema, falso × verdadeiro: um gráfico em cima do outro, letra maior."""
    if anos is None:
        anos, temas = Counter(), Counter()
        for l in open(os.path.join(_RAIZ, "datasets", "corpus_fundido.jsonl"), encoding="utf-8"):
            r = json.loads(l)
            if r["subconjunto_pareado"]:
                anos[(r["data_publicacao"][:4], r["veredito"])] += 1
                temas[(r["tema_pareamento"], r["veredito"])] += 1
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 11,
                         "xtick.labelsize": 10.5, "ytick.labelsize": 11}):
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(LARGURA, 7.6), gridspec_kw={"height_ratios": [1, 1.05]})
        xs = sorted({a for a, _ in anos})
        x = np.arange(len(xs))
        for desl, cl, cor, rot in ((-0.21, "FALSO", LARANJA, "falso"), (0.21, "VERDADEIRO", AZUL, "verdadeiro")):
            a1.bar(x + desl, [anos[(a, cl)] for a in xs], 0.42, color=cor, label=rot, edgecolor="white", linewidth=0.6)
        a1.set_xticks(x, [a[2:] for a in xs])
        a1.set_xlabel("ano de publicação (20xx)")
        a1.set_ylabel("textos no pareado")
        a1.set_title("Por ano: as duas classes têm o mesmo número de textos", loc="left")
        a1.legend(loc="upper left", ncol=2)
        a1.set_ylim(0, max(anos.values()) * 1.15)
        a1.grid(axis="x", visible=False)
        ts = sorted({t for t, _ in temas}, key=lambda t: -temas[(t, "FALSO")])
        y = np.arange(len(ts))
        a2.barh(y, [temas[(t, "FALSO")] for t in ts], 0.72, color=CINZA)
        for yi, t in zip(y, ts):
            a2.text(temas[(t, "FALSO")] + 60, yi, f"{temas[(t, 'FALSO')]:,}".replace(",", "."), va="center")
        a2.set_yticks(y, ts)
        a2.invert_yaxis()
        a2.set_xlabel("pares (falso = verdadeiro)")
        a2.set_title("Por tema", loc="left")
        a2.grid(axis="y", visible=False)
        a2.set_xlim(0, max(temas[(t, "FALSO")] for t in ts) * 1.15)
        fig.tight_layout(h_pad=2.5)
        salvar(fig, nome)


def fig_forma_conteudo():
    """Halteres: interno → Fake.br, por entrada (só forma, refinado, combinado, bruto)."""
    cb = ler("resultado_comparativo_bracos.json")["bracos"]
    est = ler("resultado_modelo_estilo.json")["gradient_boosting"]
    linhas = [("só forma (17 atributos)", 100 * est["interno"], None, 100 * est["fakebr"], [100 * v for v in est["fakebr_ic95"]], AQUA),
              ("só conteúdo (refinado)", cb["refinado"]["interno"], cb["refinado"]["interno_ic95"], cb["refinado"]["fakebr"], cb["refinado"]["fakebr_ic95"], LARANJA),
              ("forma + conteúdo (bruto)", cb["bruto"]["interno"], cb["bruto"]["interno_ic95"], cb["bruto"]["fakebr"], cb["bruto"]["fakebr_ic95"], AZUL),
              ("bruto + refinado (combinado)", cb["combinado"]["interno"], cb["combinado"]["interno_ic95"], cb["combinado"]["fakebr"], cb["combinado"]["fakebr_ic95"], CINZA)]
    from matplotlib.lines import Line2D
    with plt.rc_context({"font.size": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10.5}):
        fig, ax = plt.subplots(figsize=(LARGURA, 3.6))
        for i, (rot, vi, ici, vf, icf, cor) in enumerate(linhas):
            # A distância entre os dois pontos é a queda fora da base.
            ax.plot([vf, vi], [i, i], color=cor, lw=3, alpha=0.3, solid_capstyle="round", zorder=1)
            # IC 95%: barra com remate, na cor da série e opaca, para não sumir sob a linha.
            for v, ic in ((vi, ici), (vf, icf)):
                if ic:
                    ax.errorbar(v, i, xerr=[[v - ic[0]], [ic[1] - v]], fmt="none", ecolor=cor,
                                elinewidth=1.6, capsize=5, capthick=1.6, zorder=2)
            ax.plot(vi, i, "o", color=cor, ms=10, mec="white", mew=1.5, zorder=3)
            ax.plot(vf, i, "D", color=cor, ms=9, mec="white", mew=1.5, zorder=3)
            ax.text(vi, i + 0.3, f"{br(vi)}%", color=TINTA, ha="center")
            ax.text(vf, i + 0.3, f"{br(vf)}%", color=TINTA, ha="center")
        ax.set_yticks(range(len(linhas)), [l[0] for l in linhas])
        ax.set_ylim(-0.6, len(linhas) - 0.25)
        ax.set_xlim(50, 100)
        ax.axvline(50, color=CINZA, lw=0.8)
        ax.set_xlabel("acurácia balanceada (%)")
        ax.grid(axis="y", visible=False)
        legenda = [Line2D([], [], marker="o", ls="none", color=TINTA2, ms=9, label="teste interno"),
                   Line2D([], [], marker="D", ls="none", color=TINTA2, ms=8, label="Fake.br"),
                   Line2D([], [], marker="|", ls="-", color=TINTA2, lw=1.6, ms=10, mew=1.6, label="IC 95%")]
        ax.legend(handles=legenda, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.5,
                  columnspacing=2.0)
        salvar(fig, "fig_forma_conteudo.pdf")


def fig_transferencia():
    """Matriz 2 × 2 treino × teste, bruto e refinado."""
    inv = ler("resultado_avaliacao_inversa.json")["por_braco"]
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 2.6))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("azul", ["#cde2fb", "#2a78d6", "#0d366b"])
    for ax, braco in zip(eixos, ("bruto", "refinado")):
        b = inv[braco]
        m = np.array([[b["nosso_nosso_conferencia"]["bal"], b["nosso_fakebr_conferencia"]["bal"]],
                      [b["fakebr_nosso_teste"]["bal"], b["fakebr_fakebr"]["bal"]]])
        ax.imshow(m, cmap=cmap, vmin=50, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{br(m[i, j])}%", ha="center", va="center", fontsize=10,
                        color="white" if m[i, j] > 78 else TINTA, fontweight="bold")
        ax.set_xticks([0, 1], ["BRACE.br (teste)", "Fake.br"])
        ax.set_yticks([0, 1], ["BRACE.br", "Fake.br"])
        ax.set_xlabel("teste")
        ax.set_ylabel("treino")
        ax.set_title(f"texto {braco}", loc="left")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    salvar(fig, "fig_transferencia_2x2.pdf")


NOMES = {"P_LR": "TF-IDF palavras + LR", "P_SVM": "TF-IDF palavras + SVM", "P_NB": "TF-IDF palavras + NB",
         "Ch_LR": "TF-IDF caracteres + LR", "Ch_SVM": "TF-IDF caracteres + SVM", "Ch_NB": "TF-IDF caracteres + NB",
         "Bc_LR": "BERTimbau congelado + LR", "Bc_SVM": "BERTimbau congelado + SVM", "Bc_NB": "BERTimbau congelado + NB*",
         "Ol_LR": "bge-m3 + LR", "Ol_SVM": "bge-m3 + SVM", "Ol_NB": "bge-m3 + NB*", "Ba": "BERTimbau ajustado"}


def fig_matriz():
    """Para cada modelo (texto bruto): teste interno, BRACE.br → Fake.br, Fake.br → BRACE.br."""
    pm = ler("matriz", "apuracao.json")["por_modelo"]
    ordem = sorted(pm, key=lambda k: pm[k]["nosso_teste"])
    fig, ax = plt.subplots(figsize=(LARGURA, 4.4))
    series = (("nosso_teste", "BRACE.br → teste interno", "o", AZUL),
              ("nosso_fakebr", "BRACE.br → Fake.br", "D", LARANJA),
              ("fakebr_nosso", "Fake.br → BRACE.br", "^", AQUA))
    for i, k in enumerate(ordem):
        L = pm[k]
        excluido = k in ("Bc_NB", "Ol_NB")
        ax.plot([L["nosso_fakebr"], L["nosso_teste"]], [i, i], color=CINZA, lw=1.2, alpha=0.6)
        for chave, _, mk, cor in series:
            ax.plot(L[chave], i, mk, color=cor, ms=7, mec="white", mew=1.2, alpha=0.35 if excluido else 1)
        ax.text(L["nosso_teste"] + 0.7, i, br(L["nosso_teste"]), va="center", fontsize=7.5, color=TINTA2)
        ax.text(min(L["nosso_fakebr"], L["fakebr_nosso"]) - 0.7, i,
                f"{br(L['nosso_fakebr'])} · {br(L['fakebr_nosso'])}", va="center", ha="right", fontsize=7.5, color=TINTA2)
    ax.set_yticks(range(len(ordem)), [NOMES[k] for k in ordem])
    for t, k in zip(ax.get_yticklabels(), ordem):
        if k in ("Bc_NB", "Ol_NB"):
            t.set_color(CINZA)
    ax.set_xlim(45, 103)
    ax.set_xlabel("acurácia balanceada (%), texto bruto")
    ax.grid(axis="y", visible=False)
    for _, rot, mk, cor in series:
        ax.plot([], [], mk, color=cor, ms=7, label=rot)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3)
    fig.text(0.01, -0.04, "Números à esquerda: BRACE.br → Fake.br · Fake.br → BRACE.br. À direita: teste interno.\n"
             "* NB gaussiano: falhou no controle com rótulos embaralhados e está fora da contagem (em cinza).",
             fontsize=7.5, color=TINTA2, va="top")
    salvar(fig, "fig_matriz_modelos.pdf")


def fig_rq3():
    """Palavras decisivas do modelo × controle, k = 5, justificativa principal."""
    pb = ler("resultado_rq3_alinhamento.json")["por_braco"]
    chave = "acertadas|justificativa|exata|k5"
    fig, ax = plt.subplots(figsize=(LARGURA * 0.62, 2.5))
    x = np.arange(2)
    for desl, campo, cor, rot, mk in ((-0.18, "modelo", AZUL, "palavras decisivas do modelo", "o"),
                                     (0.18, "controle", CINZA, "palavras quaisquer do texto (controle)", "s")):
        vals = [pb[b][chave][f"{campo}_pct"] for b in ("bruto", "refinado")]
        ics = [pb[b][chave][f"{campo}_ic95"] for b in ("bruto", "refinado")]
        ax.bar(x + desl, vals, 0.34, color=cor, label=rot, edgecolor="white", linewidth=0.8)
        ax.errorbar(x + desl, vals, yerr=[[v - i[0] for v, i in zip(vals, ics)], [i[1] - v for v, i in zip(vals, ics)]],
                    fmt="none", ecolor=TINTA2, elinewidth=1, capsize=2)
        for xi, v, i in zip(x + desl, vals, ics):
            ax.text(xi, i[1] + 0.6, f"{br(v)}%", ha="center", fontsize=8)
    for xi, b in zip(x, ("bruto", "refinado")):
        d = pb[b][chave]["dif_pp"]
        ax.text(xi, max(pb[b][chave]["modelo_pct"], pb[b][chave]["controle_pct"]) + 5, f"+{br(d)} pp",
                ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x, ["texto bruto", "texto refinado"])
    ax.set_ylabel("% presentes na justificativa")
    ax.set_ylim(0, 32)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1, fontsize=7.5)
    ax.grid(axis="x", visible=False)
    salvar(fig, "fig_rq3_alinhamento.pdf")


def fig_fidelidade():
    """C3 adjudicada: categorias por classe."""
    adj = ler("fidelidade", "resultado_fidelidade.json")["adjudicado"]["por_classe"]
    cats = (("fiel", AZUL), ("perdeu_central", AMARELO), ("mudou_sentido", LARANJA), ("inventou", AQUA))
    nomes = {"fiel": "fiel", "perdeu_central": "perda do foco", "mudou_sentido": "alteração de sentido", "inventou": "alucinação"}
    fig, ax = plt.subplots(figsize=(LARGURA, 1.7))
    for i, cl in enumerate(("FALSO", "VERDADEIRO")):
        esq = 0
        n = adj[cl]["n"]
        for cat, cor in cats:
            v = 100 * adj[cl]["categorias"].get(cat, 0) / n
            ax.barh(i, v, left=esq, color=cor, edgecolor="white", linewidth=2, height=0.6,
                    label=nomes[cat] if i == 0 else None)
            if v >= 5:
                ax.text(esq + v / 2, i, f"{br(v, 0)}%", ha="center", va="center", fontsize=8,
                        color="white" if cat == "fiel" else TINTA)
            esq += v
    ax.set_yticks([0, 1], ["frases das falsas", "frases das verdadeiras"])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% das 100 frases refinadas de cada classe (anotação adjudicada)")
    ax.grid(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4)
    salvar(fig, "fig_c3_fidelidade.pdf")


def fig_temporal():
    """Datado × não datado: só o ano, TF-IDF no próprio teste, no teste datado e no Fake.br."""
    c = ler("pareamento_datado", "resultado.json")["celulas"]
    grupos = [("só o ano", c["ano_D"], c["ano_N"]), ("TF-IDF,\npróprio teste", c["D_D"], c["N_N"]),
              ("TF-IDF,\nteste datado", c["D_Dteste_mesmos"], c["N_Dteste"]), ("TF-IDF,\nFake.br", c["D_fakebr"], c["N_fakebr"])]
    fig, ax = plt.subplots(figsize=(LARGURA, 2.5))
    x = np.arange(len(grupos))
    for desl, idx, cor, rot in ((-0.19, 1, AZUL, "verdadeiras datadas (D)"), (0.19, 2, LARANJA, "verdadeiras não datadas (N)")):
        vals = [g[idx]["bal"] for g in grupos]
        ics = [g[idx]["ic95"] for g in grupos]
        ax.bar(x + desl, vals, 0.36, color=cor, label=rot, edgecolor="white", linewidth=0.8)
        ax.errorbar(x + desl, vals, yerr=[[v - i[0] for v, i in zip(vals, ics)], [i[1] - v for v, i in zip(vals, ics)]],
                    fmt="none", ecolor=TINTA2, elinewidth=1, capsize=2)
        for xi, v, i in zip(x + desl, vals, ics):
            ax.text(xi, i[1] + 1.2, br(v), ha="center", fontsize=8)
    ax.axhline(50, color=CINZA, lw=0.8, ls="--")
    ax.set_xticks(x, [g[0] for g in grupos])
    ax.set_ylabel("acurácia balanceada (%)")
    ax.set_ylim(30, 104)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    ax.grid(axis="x", visible=False)
    salvar(fig, "fig_controle_temporal.pdf")


if __name__ == "__main__":
    print(f"figuras em {SAIDA}:")
    for f in (fig_base, fig_forma_conteudo, fig_transferencia, fig_matriz, fig_rq3, fig_fidelidade, fig_temporal):
        f()
