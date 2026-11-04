# Pré-registro — avaliação inversa: treinar no Fake.br, testar na nossa base

> Escrito em 09/10/2026, **antes** de qualquer modelo ser treinado no Fake.br.
> Commitar este arquivo e o script antes de rodar. Nada aqui é editado depois;
> correções vão num adendo datado, no fim.

**Pedido da orientação (09/10/2026):** (1) revisar como os trabalhos de
avaliação entre bases montam treino e teste, e se o nosso desenho segue a
prática; (2) treinar no Fake.br e testar na nossa base.

---

## 0. Como a literatura faz, e onde estamos

| trabalho | treino (origem) | teste (destino) | referência intra-base |
|---|---|---|---|
| Janicka et al. (2019), 4 bases em inglês | a base de origem **inteira** | a base de destino **inteira** | divisão aleatória 80/20 da origem |
| Guedes et al. (PROPOR 2026), Fake.br, Fakepedia, FakeRecogna, FakeTrueBR | 80% da origem (15% validação, 5% teste) | a base de destino **inteira** ("desconhecida pelo modelo") | os 5% de teste da origem |
| nosso adendo 5 / braço refinado | `treino` da nossa base (divisão temporal), C na `validação` | Fake.br **inteiro** (7.200), nada ajustado nele | `teste` da nossa base |

**Leitura:** testar na base de destino inteira é a prática, e é o que fizemos.
Como o modelo nunca viu o destino, não há motivo para recortá-lo; usar tudo dá
mais itens e um IC menor. A condição é não ajustar nada no destino (nem C, nem
limiar, nem vocabulário), o que já cumprimos.

**Fraqueza comum, inclusive nossa:** a queda é medida entre conjuntos de teste
diferentes. Os 91% foram medidos no nosso teste e os 74% no Fake.br, e nenhum
dos dois trabalhos acima relata o desempenho intra-base **nos mesmos itens** do
destino. Sem esse número, "queda" mistura duas coisas: a perda por mudar de
base e a dificuldade própria do destino. Este pré-registro fecha a matriz 2 × 2:

| treino \ teste | nosso `teste` (5.162) | Fake.br (7.200) |
|---|---|---|
| **nossa base** | feito: 91,19% bruto, 81,73% refinado | feito: 73,51%, 66,31% |
| **Fake.br** | **novo** (este pré-registro) | **novo**: validação cruzada, fora da dobra |

**Diferenças nossas em relação a Guedes et al. que precisam ir para o texto:**
cortamos as duas classes no mesmo tamanho (cerca de 300 caracteres). Lá, o texto
verdadeiro não foi truncado e é, em média, 1,4 a 15 vezes maior que o falso
(6 vezes no Fake.br), o que os próprios autores listam como limitação; parte do
0,99 intra-base deles pode ser tamanho. Nossa divisão é temporal, não aleatória.
Eles também mediram a sobreposição de textos entre as bases (22% do FakeTrueBR
está no Fakepedia), e nós ainda não medimos a nossa com o Fake.br.

## 1. Desenho fixado

| item | valor |
|---|---|
| dados | `datasets/braco_refinado.jsonl`: os mesmos registros do comparativo dos braços |
| braços | **bruto** (`texto_entrada`, cerca de 300 caracteres nas duas bases) e **refinado** (`ideia_central`, prompt `cego-v2`, o mesmo nas duas bases) |
| modelo | o do comparativo: TF-IDF (1–2 gramas, `min_df=3`, sublinear) + regressão logística, C em {0,1; 1; 10} |
| dobras do Fake.br | 5 dobras **por par** (o falso `n` e o verdadeiro `n` ficam na mesma dobra, porque o par é alinhado por assunto), sorteio com semente 0 |
| C do modelo Fake.br | o de maior acurácia balanceada média nas 5 dobras. **Nenhum dado da nossa base é usado para escolher nada** |
| célula Fake.br → Fake.br | predições fora da dobra para os 7.200, com o C escolhido |
| célula Fake.br → nosso | modelo treinado nos 7.200, testado no `teste` da nossa base (5.162), o mesmo conjunto dos 91,19% |
| controle de tamanho | a nossa base tem 16.707 textos de treino, e o Fake.br, 7.200. Sorteio de 3.600 + 3.600 do nosso `treino` (semente 0), C na nossa `validação`, testado no Fake.br. Separa "a nossa base é maior" de "a nossa base é diferente" |
| sobreposição | textos normalizados (minúsculas, só letras e espaços) idênticos entre as bases, e os que coincidem nos 100 primeiros caracteres. Relatado antes do resultado |
| métricas | acurácia balanceada com IC95% por bootstrap (2.000, semente 0); revocação de cada classe; refinado × bruto por McNemar exato, nos mesmos itens |
| sensibilidade | Fake.br → **todo** o nosso pareado presente no arquivo do braço (23.949: treino + validação + teste), para ter mais itens |
| código | `scripts/avaliacao_inversa.py`. Só o modo `--fumaca` rodou, em 150 pares do Fake.br e 300 textos do nosso teste, sem mostrar métricas nem a sobreposição (50 s) |
| "não é boato" | a anotação manual ainda não foi apurada: roda sem exclusão, como o comparativo dos braços, e se declara |

**Limite declarado agora:** as duas bases usam definições diferentes de falso.
No Fake.br, falso é matéria de site de desinformação, em gênero de notícia. Na
nossa base, é mensagem viral desmentida por checador. Uma queda nesta direção é
compatível com "o modelo do Fake.br não generaliza" **e** com "as bases medem
coisas diferentes". As épocas também diferem: o Fake.br vai de 2016 a 2018, e o
nosso teste começa em 06/08/2024.

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| I1 | Fake.br → nosso, bruto | **55% a 68%** |
| I2 | revocação de FALSO **menor** que a de VERDADEIRO, I1 | diferença **≥ 15 pp**: a matéria da Agência Brasil se parece com a do G1 e da Folha; a corrente não se parece com a matéria de blog |
| I3 | Fake.br → Fake.br (fora da dobra), bruto | **80% a 92%** (abaixo do 0,99 de Guedes et al. com texto inteiro, porque o corte em 300 caracteres tira o atalho de tamanho) |
| I4 | Fake.br → nosso, refinado − bruto | **−3 a +5 pp** |
| I5 | controle de tamanho: nosso (7.200) → Fake.br | a **até 3 pp** dos 73,51%: o tamanho não explica a transferência |
| I6 | sobreposição entre as bases | **< 0,5%** dos itens |

**Por que:** o Fake.br separa blog partidário de grande imprensa; a nossa base
separa corrente de agência. Do lado verdadeiro, os dois são notícia de imprensa,
e isso deve transferir. Do lado falso, os gêneros são diferentes, e isso não
deve transferir. Em Guedes et al., Fake.br → FakeTrueBR (Boatos.org × G1 e
Folha, a base mais parecida com a nossa) deu 0,62 de acurácia com BERTimbau.

## 3. Leituras fixadas antes do resultado

**Fake.br → nosso (bruto, principal):**

| resultado | leitura |
|---|---|
| **≥ 80%** | o que o Fake.br ensina cobre a mensagem viral: há um sinal comum de desinformação entre as bases |
| **65% a 80%** | transferência parcial |
| **55% a 65%** | transferência fraca: as bases medem gêneros de falso diferentes |
| **45% a 55%** | nada transfere |
| **< 45%** | inversão: o modelo do Fake.br chama a corrente de verdadeira com frequência |

**Assimetria** (com o controle de tamanho): compara Fake.br → nosso com nosso
(7.200) → Fake.br, os dois treinados com 7.200 textos.
- Se a nossa direção ficar **≥ 5 pp** acima: a nossa base ensina algo que vale
  no Fake.br mais do que o contrário.
- Se a diferença ficar **dentro de ±5 pp**: transferência simétrica.
- Se a direção do Fake.br ficar **≥ 5 pp** acima: o contrário.

**Queda nos mesmos itens:** (Fake.br → Fake.br) − (nosso → Fake.br), nos 7.200.
Esta é a queda "limpa" da nossa base fora dela, que substitui a comparação
91% × 74% entre conjuntos diferentes.

## 4. Previsões do pesquisador

> Preencher antes de rodar, em uma linha cada.

Registradas em 09/10/2026, antes da rodada (data preenchida pelo assistente no commit).

- **I1** (Fake.br → nosso, bruto: acurácia balanceada): ___48% - 56%___
- **I2** (qual classe o modelo do Fake.br acerta mais na nossa base?): ___Política___
- **I3** (Fake.br → Fake.br: acurácia balanceada): ___70% - 80%___
- **I4** (na nossa base, o refinado vai melhor ou pior que o bruto?): ___vai pior___
- **Assimetria** (qual direção transfere melhor, com o mesmo tamanho de treino?): ___treinar no nosso para testar no fake, a nossa base representa uma melhora para treino___

---

## Referências

- Janicka, M.; Pszona, M.; Wawer, A. Cross-Domain Failures of Fake News
  Detection. *Computación y Sistemas*, 23(3), 2019, p. 1089–1097.
  doi:10.13053/CyS-23-3-3281.
- Guedes, R.; Barros, B.; do Nascimento, H. Identificação de notícias falsas em
  português: um olhar sobre a generalização de modelos. *PROPOR 2026*, v. 1,
  p. 717–726. https://aclanthology.org/2026.propor-1.71/
- Monteiro, R. A. et al. Contributions to the Study of Fake News in Portuguese:
  New Corpus and Automatic Detection Results. *PROPOR 2018*, LNCS 11122,
  p. 324–334.

## Adendos

### Adendo 1 — 09/10/2026, resultado da rodada

**Execução.** Pré-registro e script commitados antes da rodada; rodada única
(3 min 50 s). Saída: `relatorios/resultado_avaliacao_inversa.json`.
Conferências: nosso → nosso 91,19% e 81,73% (iguais às referências); nosso →
Fake.br 73,53% e 66,31% (referências 73,51% e 66,31%; a diferença de 0,02 pp no
bruto vem da implementação do comparativo, que este script reaproveita).

**Sobreposição (I6).** Nenhum texto do Fake.br é idêntico a um da nossa base; 4
de 7.200 coincidem nos 100 primeiros caracteres. Não há vazamento entre as bases.

**Matriz 2 × 2** (acurácia balanceada, IC95%):

| braço | treino \ teste | nosso teste (5.162) | Fake.br (7.200) |
|---|---|---|---|
| bruto | nossa base | 91,19% | 73,53% [72,5; 74,5] |
| bruto | Fake.br | **78,07%** [76,9; 79,2] | 89,57% [88,8; 90,3] (fora da dobra) |
| refinado | nossa base | 81,73% | 66,31% |
| refinado | Fake.br | **69,20%** [68,0; 70,5] | 74,24% [73,2; 75,2] (fora da dobra) |

Revocações, Fake.br → nosso, bruto: FALSO 74,2%, VERDADEIRO 82,0%. Refinado:
59,7% e 78,7%. Sensibilidade em todo o pareado (23.949): 78,06% (bruto) e
70,20% (refinado), a mesma conclusão. C escolhido no Fake.br: 10 (bruto) e 1
(refinado).

**Controle de tamanho (I5).** Nosso treino reduzido a 7.200 → Fake.br: 72,62%,
contra 73,53% com o treino inteiro (−0,9 pp). O tamanho da nossa base quase não
muda o que transfere.

**Queda nos mesmos itens.** No Fake.br: (Fake.br → Fake.br) − (nosso → Fake.br)
= **16,0 pp** [14,9; 17,1] no bruto e 7,9 pp no refinado. Esta é a queda "limpa"
do nosso modelo fora da base. Substitui a comparação 91,19% × 73,51% (17,7 pp),
feita entre conjuntos diferentes, e fica perto dela.

**Leituras fixadas aplicáveis (§3):**
- Fake.br → nosso, bruto, 78,07%: **"transferência parcial"** (65% a 80%).
- Assimetria com o mesmo tamanho de treino, bruto: 72,62 − 78,07 = **−5,45 pp →
  "o Fake.br transfere melhor"**. A diferença passa o limiar de 5 pp por pouco.
  No refinado, −3,94 pp: "simétrica".

**Confronto com as previsões:**

| | resultado | §2 | pesquisador |
|---|---|---|---|
| I1 Fake.br → nosso, bruto | 78,07% | 55–68 ✗ (acima) | 48–56 ✗ (acima) |
| I2 revocação VERDADEIRO − FALSO | +7,8 pp | ≥ 15 ✗ (a direção acerta, o tamanho não) | "Política": é um tema, não uma classe; não avaliável como escrito |
| I3 Fake.br → Fake.br | 89,57% | 80–92 ✓ | 70–80 ✗ (acima) |
| I4 refinado − bruto, Fake.br → nosso | −8,87 pp (McNemar 544 × 1.002, p < 0,001) | −3 a +5 ✗ | "vai pior" ✓ |
| I5 controle de tamanho | −0,9 pp | ±3 ✓ | — |
| I6 sobreposição | 0% | < 0,5% ✓ | — |
| assimetria | o Fake.br transfere melhor (−5,45 pp) | — | "treinar no nosso transfere melhor" ✗ |

**Leitura.** As duas bases compartilham um sinal que transfere nos dois
sentidos: registro informal e sensacionalista contra registro jornalístico. O
modelo do Fake.br aprende "blog de desinformação × grande imprensa" e acerta 78%
de "corrente × Agência Brasil". Ele acerta mais do que o nosso modelo acerta no
Fake.br, provavelmente porque o contraste de estilo na nossa base é ainda mais
marcado (só a forma dá 87% internamente; RQ4). A nossa base é um teste mais fácil
para um detector de estilo. Isso não mostra que ela ensina menos: com o mesmo
tamanho, as duas direções ficam a 5,5 pp uma da outra, e as quedas nos mesmos
itens são parecidas (16,0 pp no Fake.br; 13,1 pp na nossa base, 91,19 − 78,07,
comparação não pré-registrada). O refinamento piora nas quatro células,
inclusive dentro do Fake.br (89,57% → 74,24%), o que reforça a RQ1: tirar a
forma tira o sinal que mais transfere, nas duas bases.

**Desvios e limites declarados:** (1) a data das previsões do pesquisador ficou
em branco e foi preenchida pelo assistente (09/10/2026) no commit do
pré-registro, como no pré-registro da RQ3; (2) a previsão I2 do pesquisador cita
um tema ("Política") onde se pedia uma classe; não foi avaliada, e a acurácia
por tema não foi calculada nesta rodada; (3) a leitura de assimetria compara
destinos diferentes, de dificuldade diferente, e passa o limiar por 0,45 pp, o
que limita o alcance de "o Fake.br transfere melhor"; (4) a anotação manual
ainda não foi apurada, então o teste inclui entradas que talvez não sejam o
boato, como no comparativo dos braços.
