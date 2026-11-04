# Pré-registro — atraso entre a mensagem viral e a checagem (exploratório)

> Escrito em 19/10/2026, **antes** de calcular qualquer atraso. Só foi contado
> quantos artigos têm data de post incorporado (258) e quantos têm data de
> checagem no corpus (248). Commitar este arquivo e o script antes de rodar.
> Correções vão num adendo datado, no fim.

**Origem.** Defesa do uso da data do artigo de checagem (e não a da mensagem)
como `data_publicacao` (conversa de 18–19/10/2026). O argumento: a mensagem não
tem data de origem verificável e volta a circular (15,8% dos artigos do
Boatos.org mencionam recirculação), e a checagem sai perto da viralização. A
literatura mede atrasos de horas a dias (Shao et al., 2016: 10 a 20 horas no
Twitter; Wack, Duskin e Hodel, 2024: mediana de 4 dias), mas em outro país e
outras redes. Esta medida estima o atraso nos nossos próprios dados.

**Pergunta:** quantos dias separam o post viral incorporado na checagem da data
de publicação da checagem?

---

## 1. Desenho fixado

| item | valor |
|---|---|
| população | artigos do Boatos.org em `datasets/dataset_bruto.json` cujo bloco da mensagem viral (`texto_viral_estrutural`) traz a assinatura de um post incorporado do X/Twitter, no padrão `(@perfil) Month D, YYYY` |
| data do post | a **mais antiga** entre as datas de post no bloco da mensagem viral (a mais próxima da origem) |
| data da checagem | a `data_publicacao` do corpus para a URL do artigo; artigos sem registro no corpus ficam de fora (relatado) |
| atraso | data da checagem − data do post, em dias. Como o embed só traz o dia, o atraso tem resolução de 1 dia |
| medidas | mediana e intervalo interquartil; % com atraso ≤ 7, ≤ 30 e ≤ 365 dias; % negativo (post posterior à checagem: o embed não é a origem, por exemplo uma republicação ou uma atualização do artigo) |
| validação humana | 30 artigos sorteados (semente 0) entre os medidos: o pesquisador lê o bloco da mensagem viral, **sem ver datas nem o atraso**, e marca se o post incorporado é **a própria mensagem checada**, **outro post** (evidência, desmentido, reação) ou **não dá para saber**. O atraso é então relatado também só nos validados como "a própria mensagem" |

**Limites declarados agora:**
- é um subconjunto (cerca de 2% dos artigos): o dos boatos que circularam no
  X/Twitter e foram incorporados; boatos de WhatsApp, a maioria, não têm data
  pública, e é exatamente por isso que a data da checagem é usada;
- a data do post é a de um post, não a do primeiro aparecimento da alegação,
  que pode ser anterior em outra rede;
- a validação de 30 dá um IC largo (±15 pp perto de 70%).

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| T1 | mediana do atraso | **≤ 7 dias** |
| T2 | atraso ≤ 30 dias | **≥ 75%** dos artigos |
| T3 | atraso ≤ 365 dias | **≥ 90%** |
| T4 | atraso negativo | **≤ 10%** |
| T5 | validação: o post é a própria mensagem checada | **≥ 60%** dos 30 |

## 3. Leituras fixadas antes do resultado

| resultado | leitura |
|---|---|
| mediana ≤ 30 dias **e** ≥ 90% em até 365 dias | a data da checagem fica próxima da circulação **na escala do pareamento (ano)**: usá-la no lugar da data da mensagem não troca, em geral, o ano da instância |
| mediana ≤ 30 dias, mas < 90% em até 365 | perto na maioria, mas com uma cauda de boatos antigos checados muito depois (recirculação): declarar a cauda |
| mediana > 30 dias | a data da checagem não representa bem a circulação nem neste subconjunto: a limitação passa a ser discutida como tal |

Se a validação (T5) ficar abaixo de 50%, a leitura acima é declarada fraca: o
embed seria mais evidência do que a mensagem, e o atraso, uma medida imprecisa.

## 4. Previsões do pesquisador

> Preencher antes de rodar, em uma linha cada, com a data.

Registradas em 19/10/2026, antes da rodada (data preenchida pelo assistente no commit).

- **T1** (em quantos dias, na mediana, a checagem sai depois do post?): ___4___
- **T2** (que % das checagens sai em até 30 dias?): ___>=90%___
- **T5** (em quantos dos 30 o post incorporado é a própria mensagem?): ___50% dos casos___

---

## Referências

- Shao, C.; Ciampaglia, G. L.; Flammini, A.; Menczer, F. Hoaxy: A Platform for
  Tracking Online Misinformation. WWW 2016 Companion. arXiv:1603.01511.
- Wack, M.; Duskin, K.; Hodel, D. Political Fact-Checking Efforts are
  Constrained by Deficiencies in Coverage, Speed, and Reach. arXiv:2412.13280,
  2024.

## Adendos

### Adendo 1 — 19/10/2026, resultado da medida (validação humana pendente)

**Execução.** Pré-registro e script commitados antes (3257991). 248 artigos
medidos; 10 com embed sem registro no corpus ficaram de fora. Saída:
`relatorios/atraso_checagem/resultado_atraso.json`. Planilha de validação (30
artigos, sem datas) gerada e commitada em branco, antes da leitura.

**Resultado.**

| medida | valor |
|---|---|
| mediana do atraso | **3 dias** (intervalo interquartil de 1 a 11) |
| no mesmo dia ou no seguinte | 31,9% |
| até 7 dias | 70,6% |
| até 30 dias | **81,5%** |
| até 365 dias | **93,1%** |
| mais de um ano | 6,9% (17 artigos) |
| negativo (post depois da checagem) | 0% |

**Leitura fixada aplicável (§3): mediana ≤ 30 dias e ≥ 90% em até 365 dias →
"a data da checagem fica próxima da circulação na escala do pareamento (ano)".**
A cauda de 6,9% acima de um ano é coerente com a recirculação já documentada
(boatos antigos checados de novo quando voltam). A força da leitura depende da
validação (T5), pendente.

**Confronto com as previsões (parcial):**

| | resultado | §2 | pesquisador |
|---|---|---|---|
| T1 mediana | 3 dias | ≤ 7 ✓ | 4 dias (a 1 dia do resultado) |
| T2 até 30 dias | 81,5% | ≥ 75% ✓ | ≥ 90% ✗ |
| T3 até 365 dias | 93,1% | ≥ 90% ✓ | — |
| T4 negativo | 0% | ≤ 10% ✓ | — |
| T5 validação | pendente | ≥ 60% | 50% |

**Desvio declarado:** a data das previsões do pesquisador ficou em branco e
foi preenchida pelo assistente (19/10/2026) no commit do pré-registro.

### Adendo 2 — 26/10/2026, validação humana (30 artigos)

**Execução.** Planilha em branco commitada antes da leitura (d944d7f). Anotação
pelo pesquisador, sem ver as datas; dúvidas de casos individuais discutidas com
o assistente durante a anotação, que buscou o título e a justificativa do
checador na base, sem mostrar datas. Saída:
`relatorios/atraso_checagem/resultado_validacao.json`.

**Resultado da validação (T5):** o post incorporado é a própria mensagem
checada em **23 de 30 (76,7%)**; em 7 (23,3%) é outro post. Nenhum caso ficou
como "não dá para saber". Os outros posts foram, nas conversas durante a
anotação, de dois tipos: **evidência da origem** (o vídeo ou a foto original,
anterior ao boato, como a tradição de Páscoa em Florença apresentada como a
reabertura de Notre-Dame, ou o tweet do diretor-geral da OMS distorcido pelo
boato) e **desmentido** (posterior ao boato, inclusive da própria pessoa
citada, como Deltan Dallagnol e Marcelo Queiroga).

**Atraso nos 23 em que o post é a mensagem:**

| versão | mediana | até 7 dias | até 30 dias | até 365 dias |
|---|---:|---:|---:|---:|
| pré-registrada (data mais antiga do bloco) | 5 dias | — | 65,2% | — |
| corrigida (data do post que é a mensagem) | **2 dias** | 69,6% | 78,3% | 95,7% |

**Desvio declarado — versão corrigida.** Durante a anotação, um caso (o de
Notre-Dame) mostrou que, em blocos com mais de um post, a data mais antiga pode
ser a de uma **evidência anterior ao boato**: o atraso desse caso era de 424
dias pela regra pré-registrada e de 1 dia pela data do post que é a mensagem. A
regra foi mudada **antes de qualquer resultado da validação**: o pesquisador
passou a indicar, na observação, qual post do bloco é a mensagem. Dos 6 blocos
com dois posts marcados como a própria mensagem, 2 têm essa indicação (um como
"mensagem = post 1", outro como "apenas a primeira parte é o conteúdo falso",
lido como post 1); nos outros 4, assumiu-se o post 1, o boato em todos os casos
de dois posts discutidos durante a anotação. As duas versões são relatadas; a
corrigida é a que mede o que o pré-registro pretendia medir. A apuração da versão corrigida foi acrescentada ao
`scripts/atraso_checagem.py` e rodada antes de ser commitada; a regra que ela
implementa foi fixada durante a anotação, antes da apuração.

**Leitura (§3).** A validação passa de 60% (T5), o que torna a leitura do
adendo 1 forte: **a data da checagem fica próxima da circulação na escala do
pareamento (ano)**. Nos artigos em que o post é a mensagem, a checagem sai, na
mediana, 2 dias depois, e 95,7% em até um ano. A cauda de atrasos longos da
medida do adendo 1 vem, em parte, de evidências antigas incorporadas aos
artigos, e não de checagens tardias.

**Confronto com as previsões:** T5, ≥ 60% (§2) ✓; o pesquisador previu 50% ✗
(o resultado foi maior).

**Achado lateral.** Num dos artigos (Deltan Dallagnol), a instância falsa da
base é o próprio desmentido do citado, e não o boato: um erro da extração
estrutural, do tipo apontado pela RQ2 e que a anotação manual das 200 falsas
vai medir.
