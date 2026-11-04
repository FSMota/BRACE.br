# Pré-registro — RQ2, experimento pareado (estrutura × LLM no mesmo artigo)

> Escrito em 08/10/2026, **antes** de a via do LLM rodar nos artigos sorteados.
> Commitar antes de rodar `scripts/experimento_rq2.py --rodar`. Nada aqui é editado
> depois da rodada; correções vão num adendo datado, no fim.

**Pergunta (RQ2):** para recortar a mensagem falsa de dentro do artigo de
checagem, regras sobre a estrutura da página funcionam melhor do que pedir a um
LLM?

**Por que um experimento novo:** os números de hoje (literalidade html 99,4% ×
llm 87,1%) vêm de conjuntos disjuntos. O LLM só rodou onde a página não tinha
marcação, possivelmente os artigos mais difíceis. Aqui as duas vias são
comparadas no mesmo artigo.

---

## 1. Desenho fixado

| item | valor |
|---|---|
| população | falsas do subconjunto pareado com `origem_viral` = `html` (1.275) ou `texto` (1.010), um bloco por artigo, uma entrada pareada por URL |
| amostra | 200 por via = 400 artigos, semente 20261008 (`--preparar`) |
| fila | `datasets/rq2_fila.jsonl`, sha256 `7a93e9d826835814120b2fce19c3616991861505e58c7dabafb0f6be4a793182` |
| via do LLM | a mesma função do pipeline, `extrair_boatos_do_artigo` (qwen3.5:9b, extração `v3-2026-10`), com o artigo **sem** o recorte estrutural; sha do código gravado em cada saída e conferido na primeira linha da execução |
| referência | o `texto_viral_bruto` determinístico já gravado no corpus |
| resposta bruta | gravada antes de `limpar_viral()`, para medir o que o modelo faz sem a rede de segurança |

**Métricas** (IC95% de Wilson; por via e no total):

- **M1 · rendimento:** % dos artigos em que o LLM devolve ao menos uma mensagem
  depois da limpeza do pipeline. A via determinística rende 100% por construção
  da amostra (declarado).
- **M2 · literalidade:** % das mensagens do LLM encontradas no artigo
  (`classificar_literalidade`), comparada com a da via determinística nos mesmos
  artigos. Teste: McNemar exato, pareado.
- **M3 · concordância com o recorte determinístico** (`SequenceMatcher` sobre o
  texto normalizado): igual ≥ 0,90; parcial 0,50–0,90; diferente < 0,50; sem
  mensagem.
- **M4 · contaminação na resposta bruta:** % com introdução do checador
  ("Leia:", "Versão 1:"), com voz do checador, degenerada, com palavra de
  veredito, e % descartados pela limpeza.
- **M5 · título copiado:** % das mensagens do LLM com similaridade ≥ 0,80 com o
  título do artigo de checagem.
- **Sensibilidade:** M1 e M3 só nos artigos cuja mensagem começa dentro dos 3.800
  caracteres que a via do LLM lê.

**Limites declarados agora:**
- A referência não é verdade absoluta: o recorte determinístico também erra. M3
  mede concordância, não acerto.
- Um modelo e um prompt.
- O corte de 3.800 caracteres é parte da via do LLM como ela rodou no pipeline;
  por isso a sensibilidade.

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| M1 | rendimento do LLM | **80% – 95%** |
| M2 | literalidade do LLM / da via determinística nos mesmos artigos | **85% – 93%** / **≥ 98%**; McNemar a favor da determinística (p < 0,05) |
| M3 | igual (≥ 0,90) | **40% – 65%**; diferente + sem mensagem ≤ 30% |
| M4 | introdução do checador na resposta bruta / voz do checador / veredito | **10% – 30%** / **5% – 15%** / **≤ 5%** |
| M5 | título copiado | **≤ 5%** |
| V | via `texto` (arquivo antigo) mais difícil para o LLM que a `html` | M1 e M3 ao menos **5 pp** menores |

## 3. Leituras fixadas antes do resultado

| resultado | leitura |
|---|---|
| literalidade determinística − LLM **≥ 5 pp**, McNemar significativo | **confirma:** no mesmo artigo, a estrutura supera o LLM; a vantagem medida antes não era só efeito de seleção |
| diferença **< 3 pp** e M3 igual **≥ 80%** | **refuta:** quando a mensagem está no texto, o LLM recorta quase como a regra; a diferença anterior vinha dos artigos mais difíceis. A RQ2 passa a ser sobre custo e cobertura, não sobre fidelidade |
| entre os dois | **parcial:** a estrutura é mais fiel, mas parte da diferença anterior era seleção |

Em qualquer caso, M4 mede quanto a rede de segurança (`limpar_viral`) é
necessária na via do LLM.

## 4. Previsões do pesquisador

> Preencher antes de rodar, em uma linha cada.

Registradas em __/__/2026, antes da rodada.

- **M1** (rendimento do LLM): ______
- **M2** (literalidade do LLM; a determinística ganha?): ______
- **M3** (% igual ao recorte determinístico): ______
- **M4** (% com introdução do checador na resposta bruta): ______
- **V** (qual via é mais difícil para o LLM?): ______

---

## Adendos

### Adendo 1 — 09/10/2026, resultado da rodada

**Execução.** Pré-registro e script commitados em 08/10/2026, 23:18 (10a9866);
a rodada terminou em 09/10, 00:44. Fila conferida (sha256 pré-registrado), 400
de 400 artigos com saída, código sha `8cece01cb14c380a` em todas, extração
`v3-2026-10`, uma tentativa por artigo, 12,2 s por artigo em média. Saída
`datasets/rq2_saida.jsonl`, sha256
`3223c5d698df752837ce713f2392648b030e0c20545e87740837b60e0afe40b2`; resultado
`relatorios/resultado_rq2_pareado.json`.

| métrica | todas | html | texto | previsão (§2) |
|---|---:|---:|---:|---|
| M1 rendimento do LLM | **97,5%** [95,5; 98,6] | 95,5% | 99,5% | 80–95% ✗ (acima) |
| M2 literalidade LLM / estrutura (mesmos artigos) | **97,2%** / **100%** | 96,9 / 100% | 97,5 / 100% | LLM 85–93% ✗; estrutura ≥ 98% ✓ |
| M2 McNemar (só estrutura × só LLM) | 11 × 0, p = 0,001 | 6 × 0, p = 0,031 | 5 × 0, p = 0,063 | a favor da estrutura ✓ |
| M3 igual / parcial / diferente / sem mensagem | **62,5** / 12,5 / 22,5 / 2,5% | 46,0 / 17,0 / 32,5 / 4,5% | 79,0 / 8,0 / 12,5 / 0,5% | igual 40–65% ✓; diferente + sem ≤ 30% ✓ |
| M4 introdução do checador / voz do checador / veredito (resposta bruta) | 0,5 / 1,2 / 2,5% | 1,0 / 2,0 / 3,5% | 0 / 0,5 / 1,5% | 10–30% ✗; 5–15% ✗; ≤ 5% ✓ |
| M4 descartados pela limpeza | 1,0% | 1,5% | 0,5% | — |
| M5 título copiado | 1,8% | 3,1% | 0,5% | ≤ 5% ✓ |
| V via mais difícil | — | **html** | — | `texto` ✗ (o contrário) |

Sensibilidade (mensagem dentro dos 3.800 caracteres lidos, 392 de 400): M1
97,7%, M3 igual 63,8%; nada muda.

**Leitura fixada aplicável (§3): parcial.** A literalidade da estrutura supera
a do LLM no mesmo artigo, com significância, mas por 2,8 pp (abaixo dos 5 pp de
"confirma"); a concordância de recorte, 62,5%, fica abaixo dos 80% de "refuta".
A diferença medida antes (99,4% × 87,1%) era em boa parte efeito de seleção: no
mesmo artigo, o LLM chega a 97,2%. A contaminação pela voz do checador, que
motivou as extrações v2 e v3, quase não aparece com o prompt v3.

**Previsões do pesquisador:** a §4 ficou em branco; não há previsões do
pesquisador registradas para este experimento. Declarado como desvio.

**Análise EXPLORATÓRIA (decidida depois de ver o resultado; não
pré-registrada).** Para interpretar os 90 recortes "diferentes", 20 foram
sorteados (14 html, 6 texto; semente 20261009) e classificados pelo Claude, sem
cegamento: `relatorios/rq2_inspecao_exploratoria.json`.

| classificação | html (14) | texto (6) | total |
|---|---:|---:|---:|
| estrutura errou, LLM acertou | 8 | 2 | **10** |
| os dois erraram | 1 | 0 | 1 |
| LLM errou, estrutura acertou | 2 | 1 | 3 |
| LLM pegou só parte de uma mensagem válida | 0 | 3 | 3 |
| os dois válidos (partes diferentes da mesma mensagem) | 3 | 0 | 3 |

Os erros da estrutura foram: tweet citado pelo checador como evidência (inclusive
um do próprio Doria), assinatura de embed, link promocional do site ("20 fake
news..."), perguntas e respostas do checador num template recente ("A Embraer
fez um filme? Não."), e duas versões juntas.

**Consequência para o corpus, a confirmar:** esses recortes são entradas de
falsas no subconjunto pareado. Ordem de grandeza (n = 20, só para orientar):
cerca de 1 em cada 5 entradas da via html (65/200 × 9/14) e cerca de 1 em 25
da via texto (25/200 × 2/6) seriam outra coisa que não a mensagem viral. Isso
não é estimativa para citar. É o motivo para priorizar a anotação manual
(`relatorios/anotacao/`), que mede essa taxa às cegas, e para revisar a extração
estrutural (embeds de evidência e o template de perguntas e respostas).

**Leitura para a RQ2, a partir deste adendo:** quando a mensagem é localizável,
o LLM copia quase tão fielmente quanto a regra. A vantagem da estrutura é custo
(praticamente zero contra 12,2 s por artigo), determinismo e uma diferença
pequena, mas significativa, de fidelidade. A via estrutural também erra, e às
vezes mais que o LLM, na escolha do trecho: embeds e citações de evidência dentro
da marcação. "Não peça ao modelo o que a fonte informa" continua válido, com a
ressalva, agora medida, de que a fonte também precisa ser auditada.
