# -*- coding: utf-8 -*-
"""BERTimbau ajustado (Ba) da matriz de modelos. Roda no PC com GPU.
Pré-registro: `relatorios/PREREGISTRO_MATRIZ_MODELOS.md` (§1.3).

    python scripts/ajustar_bertimbau.py --fumaca     # 200 textos, 1 época; confere o código, sem métricas
    python scripts/ajustar_bertimbau.py              # 2 braços × 2 direções (horas)

Parâmetros fixados: neuralmind/bert-base-portuguese-cased, cabeça de
classificação padrão, até 128 tokens, taxa de aprendizado 2e-5, lote 32,
3 épocas, semente 0; a época é escolhida pela acurácia balanceada na validação.
  nossa direção: treino → validação (escolha) → nosso teste e Fake.br
  direção Fake.br: dobras 1–4 → dobra 0 (escolha) → nosso teste
                   (sem a célula Fake.br → Fake.br: exceção declarada no §1.3)
As dobras por par são as da avaliação inversa (semente 0), reimplementadas
aqui para o script não depender do resto do repositório.

Saída: `relatorios/matriz/pred_Ba_<braço>.npz`, no formato das outras partes.
"""
import os
import sys
import json
import random

import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARQ_DADOS = os.path.join(_RAIZ, "datasets", "braco_refinado.jsonl")
PASTA = os.path.join(_RAIZ, "relatorios", "matriz")
BRACOS = {"bruto": "texto_entrada", "refinado": "ideia_central"}
NOME = "neuralmind/bert-base-portuguese-cased"
LR, LOTE, EPOCAS, MAXLEN, SEED, N_DOBRAS = 2e-5, 32, 3, 128, 0, 5


def bal(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return 0.5 * ((p[y == 1] == 1).mean() + (p[y == 0] == 0).mean())


def rotulos(regs):
    return np.array([1 if r["veredito"] == "FALSO" else 0 for r in regs])


def dobras_por_par(fakebr):
    """Igual a `avaliacao_inversa.dobras_por_par` (semente 0)."""
    pares = sorted({int(r["id"].rsplit("-", 1)[1]) for r in fakebr})
    random.Random(SEED).shuffle(pares)
    dobra_de = {p: i % N_DOBRAS for i, p in enumerate(pares)}
    return [dobra_de[int(r["id"].rsplit("-", 1)[1])] for r in fakebr]


def treinar_e_prever(treino, validacao, testes, campo, epocas):
    """Ajusta, escolhe a época pela validação e devolve as predições da época escolhida."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    disp = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(NOME)
    mod = AutoModelForSequenceClassification.from_pretrained(NOME, num_labels=2).to(disp)
    opt = torch.optim.AdamW(mod.parameters(), lr=LR)

    def lotes(regs, embaralhar):
        idx = list(range(len(regs)))
        return DataLoader(idx, batch_size=LOTE, shuffle=embaralhar,
                          generator=torch.Generator().manual_seed(SEED))

    def codificar(regs, idx):
        enc = tok([regs[i][campo] or " " for i in idx], truncation=True, max_length=MAXLEN,
                  padding=True, return_tensors="pt")
        return {k: v.to(disp) for k, v in enc.items()}

    @torch.no_grad()
    def prever(regs):
        mod.eval()
        saida = []
        for idx in lotes(regs, False):
            with torch.autocast(device_type=disp, dtype=torch.bfloat16, enabled=(disp == "cuda")):
                saida.append(mod(**codificar(regs, idx.tolist())).logits.argmax(-1).cpu().numpy())
        return np.concatenate(saida)

    y_tr = torch.tensor(rotulos(treino))
    melhor = None
    for ep in range(epocas):
        mod.train()
        for idx in lotes(treino, True):
            idx = idx.tolist()
            with torch.autocast(device_type=disp, dtype=torch.bfloat16, enabled=(disp == "cuda")):
                perda = mod(**codificar(treino, idx), labels=y_tr[idx].to(disp)).loss
            opt.zero_grad()
            perda.backward()
            opt.step()
        b = bal(rotulos(validacao), prever(validacao))
        print(f"    época {ep + 1}: validação {100 * b:.2f}%" if "--fumaca" not in sys.argv else f"    época {ep + 1}: ok")
        if melhor is None or b > melhor[0]:
            melhor = (b, ep + 1, {k: prever(v) for k, v in testes.items()})
    return melhor[1], melhor[2]


def main():
    fumaca = "--fumaca" in sys.argv
    regs = [json.loads(l) for l in open(ARQ_DADOS, encoding="utf-8") if l.strip()]
    por = {s: [r for r in regs if r["split"] == s] for s in ("treino", "validacao", "teste", "fakebr")}
    if fumaca:
        rnd = random.Random(1)
        por = {s: rnd.sample(v, min(len(v), 200)) for s, v in por.items()}
    dobras = dobras_por_par(por["fakebr"])
    fb_treino = [r for r, d in zip(por["fakebr"], dobras) if d != 0]
    fb_valid = [r for r, d in zip(por["fakebr"], dobras) if d == 0]
    epocas = 1 if fumaca else EPOCAS
    os.makedirs(PASTA, exist_ok=True)
    for braco, campo in BRACOS.items():
        print(f"{braco}: nossa direção")
        ep_n, p_n = treinar_e_prever(por["treino"], por["validacao"],
                                     {"nosso_teste": por["teste"], "nosso_fakebr": por["fakebr"]}, campo, epocas)
        print(f"{braco}: direção Fake.br")
        ep_f, p_f = treinar_e_prever(fb_treino, fb_valid, {"fakebr_nosso": por["teste"]}, campo, epocas)
        if fumaca:
            print(f"  (fumaça) {braco} rodou")
            continue
        y_te, y_fb = rotulos(por["teste"]), rotulos(por["fakebr"])
        print(f"  {braco}: época escolhida {ep_n} / {ep_f} | nosso→teste {100 * bal(y_te, p_n['nosso_teste']):.2f} | "
              f"nosso→Fake.br {100 * bal(y_fb, p_n['nosso_fakebr']):.2f} | "
              f"Fake.br→nosso {100 * bal(y_te, p_f['fakebr_nosso']):.2f}")
        np.savez_compressed(os.path.join(PASTA, f"pred_Ba_{braco}.npz"),
                            teste_ids=np.array([r["id"] for r in por["teste"]]),
                            fakebr_ids=np.array([r["id"] for r in por["fakebr"]]),
                            nosso_teste=p_n["nosso_teste"], nosso_fakebr=p_n["nosso_fakebr"],
                            fakebr_nosso=p_f["fakebr_nosso"], epoca_nossa=ep_n, epoca_fakebr=ep_f)


if __name__ == "__main__":
    main()
