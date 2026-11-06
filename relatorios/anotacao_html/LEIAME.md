# Auditoria dirigida da via html (2016 e 2026)

Exploratória, posterior aos experimentos. Pergunta: a taxa de entradas que não são
a mensagem que circulou difere entre 2016 e 2026, os anos com mais entradas da via
html? Hipótese do pesquisador: o padrão do `<blockquote>` é mais estável nos artigos
recentes, e 2026 terá menos erros.

- Sorteio: `scripts/anotacao_html_epoca.py`, semente 20261106, 50 entradas por ano,
  sem as lidas na anotação das 200 falsas.
- `amostra.json` guarda a chave dos anos e não deve ser aberta antes do fim da leitura.
- **Desvio declarado:** o sorteio foi registrado no Git em 06/11/2026, depois de o
  pesquisador ter começado a leitura. A planilha foi gerada uma única vez, e o
  arquivo não mudou desde então (sha256 de `amostra.json`:
  43fd80f17b66b9f497c20056425c0a14a38eaa402035119c426ef8ec36ce2baa).
- A leitura não é totalmente cega: a planilha não mostra ano nem id, mas o link
  leva à página, onde a data aparece.
