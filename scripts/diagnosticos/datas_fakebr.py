# -*- coding: utf-8 -*-
"""Período coberto pelo Fake.br, pelas datas dos metadados (exploratório)."""
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from avaliacao_cruzada import carregar_fakebr   # noqa: E402

pares, _ = carregar_fakebr()
for classe in ("fake", "true"):
    datas = [p[classe]["data"] for p in pares]
    anos = Counter(m.group(0) if (m := re.search(r"20\d\d", d)) else "sem" for d in datas)
    print(classe, sorted(anos.items()), "exemplos:", datas[:3])
