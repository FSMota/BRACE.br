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

## Resultado (06/11/2026)

- Não é a mensagem: 2016, 5 de 50 (10%, IC 95% de 4,3 a 21,4); 2026, 1 de 50 (2%, de
  0,4 a 10,5). Fisher, p = 0,20: direção da hipótese, sem significância.
- Os 6 erros estão em artigos com vários registros: 6 de 26 (23%, de 11 a 42),
  contra 0 de 74 nos demais. Esses artigos são 23 das 50 entradas de 2016 e 3 das
  50 de 2026, o que pode explicar parte da diferença entre os anos.
- 18 linhas têm comentário sobre trecho faltante: 16 estão em outra entrada do mesmo
  artigo, comparadas por palavras (2 só em parte, linhas 15 e 59); 2 não entraram
  na base (linhas 25 e 74).
- Efeito: `scripts/sensibilidade_html_multiplos.py`. Sem as entradas html de artigos
  com vários registros, no máximo 0,77 ponto, sem mudar conclusão.
