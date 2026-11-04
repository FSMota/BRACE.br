# -*- coding: utf-8 -*-
"""Devolve à fila do main_nlp() os artigos com defeito de extração.

    python scripts/marcar_reprocessamento.py              # simula, não grava
    python scripts/marcar_reprocessamento.py --aplicar    # grava (com backup)

Roda ANTES de levar os datasets para a máquina com GPU. Lá, basta o main_nlp():
ele processa exatamente os artigos marcados aqui, com a extração v2 (prompt novo
+ limpeza_viral.py nas três vias).

Critérios, medidos na fusão de 29/09/2026 (RESULTADOS_E_PLANO.md):
  1. introdução do checador no viral ("Leia:", "Versão 1:", embed do Instagram...)
  2. "Versão N" no meio do viral (versões coladas num registro)
  3. viral degenerado (lixo: "Confira também...", curto demais)
  4. o mesmo viral gravado para alegações diferentes do mesmo artigo
  7. cabeçalho do checador no viral: título com "#boato", assinatura do autor,
     rodapé do site -- acrescentado em 05/10/2026 (auditar_vazamento.py).
  6. narração do checador no viral ("diz a mensagem", "de acordo com o texto")
     -- acrescentado em 04/10/2026, depois do piloto do refinamento.
  5. construção atributiva de veredito no viral -- SÓ na via LLM: nas vias
     determinísticas o viral é o bloco literal, e reprocessar não o mudaria.

O que acontece com um artigo marcado:
  - seus registros SAEM do dataset final (o main_nlp grava os novos);
  - no bruto: processado_pela_ia = False, boatos_extraidos = 0,
    reprocessar_motivo = "extracao_v2".
Nada se perde: há backup dos dois arquivos, e o texto bruto do artigo fica.

NÃO rode com o main_nlp() em execução: os dois escrevem os mesmos arquivos.
"""
import os
import re
import sys
import json
import shutil
from datetime import datetime
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limpeza_viral import limpar_viral, narrativa_checador, cabecalho_checador   # noqa: E402

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from auditar_corpus import ATRIBUTIVOS                             # noqa: E402

ARQ_BRUTO = os.path.join(_RAIZ, "datasets", "dataset_bruto.json")
ARQ_FINAL = os.path.join(_RAIZ, "datasets", "dataset_tcc_fake_news_final.json")
MOTIVO = "extracao_v3"      # v2 em 29/09/2026; v3 em 05/10/2026 (voz do checador)
SEG_POR_ARTIGO = 8.6       # medido na RTX 4060 com qwen3.5:9b; GPU mais forte = menos


def mil(n):
    return "{:,}".format(n).replace(",", ".")


def criterios(final):
    """Devolve {criterio: set(urls)}."""
    por = defaultdict(set)
    repetidos = defaultdict(set)
    for r in final:
        url, viral = r["url_origem_fake"], r.get("texto_viral_bruto") or ""
        limp = limpar_viral(viral)
        if limp["prefixos"]:
            por["1 introdução do checador"].add(url)
        if limp["versoes"] > 1:
            por["2 'Versão N' no meio"].add(url)
        if limp["degenerada"]:
            por["3 viral degenerado"].add(url)
        repetidos[(url, re.sub(r"\s+", " ", viral.lower()).strip())].add(r.get("claim_refinada"))
        if narrativa_checador(viral):
            por["6 narração do checador"].add(url)
        if cabecalho_checador(viral):
            por["7 cabeçalho do checador (#boato, autor)"].add(url)
        if r.get("origem_viral") == "llm" and ATRIBUTIVOS.search(viral):
            por["5 veredito no viral (via LLM)"].add(url)
    for (url, _), claims in repetidos.items():
        if len(claims) > 1:
            por["4 mesmo viral p/ alegações diferentes"].add(url)
    return por


def main():
    aplicar = "--aplicar" in sys.argv
    print("=" * 72)
    print("  MARCAR PARA REPROCESSAMENTO  " + ("[APLICANDO]" if aplicar else "[DRY-RUN — nada será gravado]"))
    print("=" * 72)
    with open(ARQ_BRUTO, encoding="utf-8") as f:
        bruto = json.load(f)
    with open(ARQ_FINAL, encoding="utf-8") as f:
        final = json.load(f)
    print(f"\nentrada: {mil(len(final))} registros / {mil(len(bruto))} artigos")

    ja = sum(1 for i in bruto if i.get("reprocessar_motivo") == MOTIVO)
    if ja:
        # Não afirmar que os registros novos estão limpos: em 06/10/2026 o
        # main_nlp rodou com o código anterior e eles voltaram com o defeito.
        print(f"\n[!] {mil(ja)} artigos já estão marcados com '{MOTIVO}'. Os critérios são")
        print("    recalculados sobre os registros ATUAIS: só volta à fila o que ainda tem")
        print("    defeito. Se voltar muita coisa, confira se o main_nlp rodou com o código")
        print("    certo (VERSAO_EXTRACAO). Seguindo.")

    por = criterios(final)
    urls = set().union(*por.values()) if por else set()
    print("\nartigos por critério (um artigo pode cair em vários):")
    for k in sorted(por):
        regs = sum(1 for r in final if r["url_origem_fake"] in por[k])
        print(f"  {k:40s} {mil(len(por[k])):>6} artigos  {mil(regs):>6} registros")

    saem = [r for r in final if r["url_origem_fake"] in urls]
    ficam = [r for r in final if r["url_origem_fake"] not in urls]
    no_bruto = [i for i in bruto if i.get("url_origem") in urls]
    fila_hoje = sum(1 for i in bruto if not i.get("processado_pela_ia"))
    print(f"\nTOTAL: {mil(len(urls))} artigos | {mil(len(saem))} registros saem do final "
          f"({mil(len(ficam))} ficam)")
    print(f"  encontrados no bruto: {mil(len(no_bruto))} | já na fila antes: {fila_hoje}")
    print(f"  origem dos registros que saem: {dict(Counter(r.get('origem_viral') for r in saem))}")
    fila = len(no_bruto) + fila_hoje
    print(f"  fila do main_nlp depois: {mil(fila)} artigos  ->  ~{fila * SEG_POR_ARTIGO / 3600:.1f} h "
          f"a {SEG_POR_ARTIGO} s/artigo (RTX 4060)")

    if len(no_bruto) != len(urls):
        print(f"\n[!] {len(urls) - len(no_bruto)} URLs do final não estão no bruto -- elas SAIRIAM do")
        print("    final sem voltar à fila. Abortando para não perder registros.")
        sys.exit(1)

    if not aplicar:
        print("\n" + "-" * 72)
        print("DRY-RUN: nada foi gravado. Rode com --aplicar para efetivar.")
        return

    carimbo = datetime.now().strftime("%Y%m%d_%H%M")
    for caminho in (ARQ_BRUTO, ARQ_FINAL):
        destino = caminho.replace(".json", f"_BACKUP_{carimbo}.json")
        shutil.copy2(caminho, destino)
        print(f"\nbackup: {os.path.basename(destino)}")
    for item in no_bruto:
        item["processado_pela_ia"] = False
        item["boatos_extraidos"] = 0
        item["reprocessar_motivo"] = MOTIVO
    for caminho, dados in ((ARQ_FINAL, ficam), (ARQ_BRUTO, bruto)):
        tmp = caminho + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(dados, f, indent=4, ensure_ascii=False)
        os.replace(tmp, caminho)
    print(f"\nGravado. {mil(len(ficam))} registros no final; {mil(fila)} artigos na fila do main_nlp().")


if __name__ == "__main__":
    main()
