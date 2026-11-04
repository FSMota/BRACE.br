# -*- coding: utf-8 -*-
"""Extrai os vetores densos da matriz de modelos (Bc, Ol) e do protótipo de recuperação (S3, S4).
Roda no PC com GPU. Pré-registros: `relatorios/PREREGISTRO_MATRIZ_MODELOS.md` (§1.2)
e `relatorios/PREREGISTRO_RECUPERACAO.md`.

    python scripts/extrair_vetores.py --modelo bertimbau     # precisa de torch + transformers (GPU)
    python scripts/extrair_vetores.py --modelo bge-m3        # precisa do Ollama com `ollama pull bge-m3`

Entrada: `datasets/braco_refinado.jsonl` (31.149 registros: corpus pareado + Fake.br).
Saída: `datasets/vetores/vetores_<modelo>.npz` com `ids`, `texto_entrada`,
`ideia_central` (uma linha por registro, na ordem de `ids`) e o sha da entrada.

Parâmetros fixados no pré-registro:
  Bc: neuralmind/bert-base-portuguese-cased, até 128 tokens, média dos vetores
      da última camada sem os de preenchimento.
  Ol: bge-m3 via API do Ollama, sem normalização adicional.

Retomável: grava pedaços em `datasets/vetores/partes_<modelo>/` e pula o que já existe.
"""
import os
import sys
import json
import hashlib

import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
PASTA = os.path.join(_RAIZ, "datasets", "vetores")
CAMPOS = ("texto_entrada", "ideia_central")
LOTE = 64
PEDACO = 2048                      # registros por arquivo parcial
OLLAMA = "http://localhost:11434/api/embed"


def sha_arquivo(caminho):
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()[:16]


def codificador_bertimbau():
    import torch
    from transformers import AutoTokenizer, AutoModel
    nome = "neuralmind/bert-base-portuguese-cased"
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"BERTimbau em {disp}")
    tok = AutoTokenizer.from_pretrained(nome)
    mod = AutoModel.from_pretrained(nome).to(disp).eval()

    @torch.no_grad()
    def codificar(textos):
        enc = tok(textos, truncation=True, max_length=128, padding=True, return_tensors="pt").to(disp)
        with torch.autocast(device_type=disp, enabled=(disp == "cuda")):
            h = mod(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
        return ((h * m).sum(1) / m.sum(1)).float().cpu().numpy()
    return codificar


def codificador_bge():
    import requests

    def codificar(textos):
        r = requests.post(OLLAMA, json={"model": "bge-m3", "input": textos}, timeout=600)
        r.raise_for_status()
        return np.array(r.json()["embeddings"], dtype=np.float32)
    return codificar


def main():
    modelo = sys.argv[sys.argv.index("--modelo") + 1] if "--modelo" in sys.argv else None
    if modelo not in ("bertimbau", "bge-m3"):
        sys.exit("use --modelo bertimbau ou --modelo bge-m3")
    regs = [json.loads(l) for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()]
    sha = sha_arquivo(ARQ_DADOS)
    print(f"{len(regs)} registros | sha da entrada {sha}")
    codificar = codificador_bertimbau() if modelo == "bertimbau" else codificador_bge()
    partes = os.path.join(PASTA, f"partes_{modelo}")
    os.makedirs(partes, exist_ok=True)
    n_pedacos = (len(regs) + PEDACO - 1) // PEDACO
    for p in range(n_pedacos):
        arq = os.path.join(partes, f"{p:03d}.npz")
        if os.path.exists(arq):
            continue
        bloco = regs[p * PEDACO:(p + 1) * PEDACO]
        saida = {"ids": np.array([r["id"] for r in bloco])}
        for campo in CAMPOS:
            textos = [(r.get(campo) or " ") for r in bloco]
            saida[campo] = np.vstack([codificar(textos[i:i + LOTE]) for i in range(0, len(textos), LOTE)])
        np.savez(arq, **saida)
        print(f"  pedaço {p + 1}/{n_pedacos} gravado")
    juntos = [np.load(os.path.join(partes, f"{p:03d}.npz")) for p in range(n_pedacos)]
    final = {"ids": np.concatenate([j["ids"] for j in juntos]), "sha_entrada": np.array(sha)}
    for campo in CAMPOS:
        final[campo] = np.concatenate([j[campo] for j in juntos])
    assert list(final["ids"]) == [r["id"] for r in regs]
    destino = os.path.join(PASTA, f"vetores_{modelo}.npz")
    np.savez_compressed(destino, **final)
    print(f"salvo em {destino}: {final['texto_entrada'].shape}")


if __name__ == "__main__":
    main()
