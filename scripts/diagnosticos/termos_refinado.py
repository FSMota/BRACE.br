import json, os, re, unicodedata
from collections import defaultdict
from openpyxl import load_workbook
R = os.path.expanduser("~/projetos/Fake-Collector-Refiner/")

def norm(s):
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in s if not unicodedata.combining(c))

regs = [json.loads(l) for l in open(R + "datasets/braco_refinado.jsonl", encoding="utf-8") if l.strip()]
from collections import Counter; print(Counter(r["origem"] for r in regs)); regs = [r for r in regs if "fake" not in str(r["origem"]).lower()]; print(len(regs))
for termo, classe in [("globo", "FALSO"), ("ivermectina", "FALSO"), ("video", "FALSO"),
                      ("senado", "VERDADEIRO"), ("camara", "VERDADEIRO"), ("casos", "VERDADEIRO"), ("texto", "FALSO")]:
    pad = re.compile(r"\b" + termo + r"\b")
    com = [r for r in regs if r["veredito"] == classe and pad.search(norm(r.get("ideia_central")))]
    orig = sum(bool(pad.search(norm(r.get("texto_entrada")))) for r in com)
    print(f"{termo:12s} {classe:10s} frases com o termo: {len(com):5d} | termo também no original: {orig:5d} ({100*orig/max(1,len(com)):.1f}%)")

ws = load_workbook(R + "relatorios/recuperacao/planilha_recuperacao.xlsx", read_only=True)["anotacao"]
rows = list(ws.iter_rows(values_only=True))
cab = rows[0]; i_c = cab.index("consulta_n"); i_j = cab.index("consenso_j1")
por = defaultdict(set)
for r in rows[1:]:
    if r[0]:
        por[r[i_c]].add(r[i_j])
print("consultas", len(por), "com >=1 relacionada ou mesma:", sum(1 for s in por.values() if s & {"relacionada", "mesma_alegacao"}),
      "| com >=1 mesma:", sum(1 for s in por.values() if "mesma_alegacao" in s))
