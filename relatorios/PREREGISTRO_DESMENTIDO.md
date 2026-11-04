# Pré-registro — desmentido no lugar do boato (regra de marcação e categoria na anotação)

> Escrito em 25/10/2026, **antes** de rodar a marcação no corpus inteiro e
> antes de qualquer anotação das 200 falsas (planilha em branco, conferido).
> Commitar este arquivo, os scripts, a planilha e o meta antes de anotar.
> Correções vão num adendo datado, no fim.

**Origem.** Revisando a Metodologia (24/10/2026), o pesquisador perguntou por
que 13.140 artigos geram 14.856 falsas. A conferência mostrou que a extração
gera **um registro por bloco de mensagem**: 10.331 artigos geraram 1 registro,
1.633 geraram de 2 a 13, e 1.176 não geraram nenhum. Na maioria dos casos, os
blocos de um mesmo artigo são versões do mesmo boato. Em alguns, o bloco
traz o **desmentido**: a nota de uma empresa ou órgão ("a CET informa que não
haverá…"), a fala de quem foi citado ("fizeram uma montagem falando que eu tô
com câncer") ou o texto do checador em artigos de lista ("É falso que…",
"7 fake news sobre…"). Esses registros entraram como FALSO.

Uma medida exploratória (24/10/2026) mostrou que, no teste, a revocação de
FALSO dos modelos treinados na nossa base é a mesma nos artigos com 1 registro e
nos com vários (P + LR: 92,7% contra 90,7% e 92,4%). O modelo do Fake.br
acerta menos nos blocos seguintes ao primeiro (69,2% contra 74,9%). Nenhum
sinal de uma fatia grande de rótulos errados, mas a taxa real não foi medida.

**Decisões do pesquisador (24–25/10/2026):** (1) registrar na anotação das
200 falsas quantos registros o artigo gerou, sem mostrar isso ao anotador;
(2) escrever uma regra de marcação do desmentido **sem abrir a amostra**; (3)
acrescentar a categoria `desmentido` à planilha antes de anotar.

---

## 1. Desenho fixado

| item | valor |
|---|---|
| registros por artigo | `anotacao_manual.py --registros` grava `registros_no_artigo` em `relatorios/anotacao/amostra_meta.json` (137 itens de artigo com 1 registro, 63 com 2 ou mais). A planilha não muda |
| regra | `scripts/marcar_desmentido.py`: três padrões (`nota_oficial`, `nega_autoria`, `texto_checador`), que exigem a **negação junto da voz oficial ou do citado**, porque o próprio boato usa "nota oficial" e "é falso". Grava `relatorios/marcacoes/desmentido.json` e o campo `flag_desmentido` no meta (escondido do anotador) |
| desenvolvimento da regra | só em artigos **fora** da amostra: todo artigo com algum registro sorteado ficou de fora, e nenhum texto desses artigos foi lido. Na última versão, a regra marcou 55 registros fora da amostra (36 `texto_checador`, 10 `nega_autoria`, 9 `nota_oficial`). Na leitura do assistente, que **não é cega** e é declarada como tal, 53 deles não são o boato (precisão de desenvolvimento ≈ 96%). A cobertura é baixa de propósito: a regra é estreita para errar pouco |
| categoria nova | `desmentido` na planilha (instruções e lista suspensa), via `anotacao_manual.py --atualizar-categorias`, que se recusa a rodar com qualquer linha anotada. Os textos, ids e links não mudam (conferido célula a célula). Na apuração, `desmentido` entra no grupo "não é o boato", junto de `misto`, `checador` e `lixo` |
| apuração | `anotacao_manual.py --apurar`: taxa de "não é o boato" geral, por via de extração e por registros do artigo (1 contra 2+), com IC de Wilson; precisão e cobertura de cada marcação automática, agora também `flag_desmentido`, contra a leitura humana |

## 2. Previsões

| # | previsão | faixa |
|---|---|---|
| R1 | itens anotados como `desmentido` nas 200 | **1% a 5%** (2 a 10 itens) |
| R2 | taxa de "não é o boato" nos itens de artigo com 2+ registros, comparada à dos itens de artigo com 1 | **maior**, em pelo menos 5 pp |
| R3 | itens da amostra marcados pela regra | **0 a 4** (a regra marcou ~0,4% fora da amostra) |
| R4 | cobertura da regra: dos anotados como `desmentido` ou `checador`, quantos ela marca | **≤ 50%** |

**Por que:** a regra marca cerca de 0,4% das falsas, e a leitura de 25
artigos com vários registros, sorteados fora da amostra, achou quase só versões
do mesmo boato. O desmentido deve ser raro e concentrado nos artigos com
vários blocos. Com tão poucos itens marcados na amostra, a **precisão** da regra
não poderá ser estimada nela; a anotação mede sobretudo a **cobertura**.

## 3. Leituras fixadas antes do resultado

| resultado | leitura |
|---|---|
| `desmentido` + `checador` ≤ 5% das 200 | o ruído de rótulo desse tipo é pequeno; ele entra como limitação medida, com a taxa e o IC, e os números do TCC não mudam |
| `desmentido` + `checador` > 5% | além da limitação, os números principais (modelo de referência, teste interno e Fake.br → nossa base) são recalculados **sem os registros marcados pela regra**, e a diferença é relatada |
| R2 confirmada | a versão publicada da base traz `registros_no_artigo` como atributo, para quem quiser filtrar |

## 4. Previsões do pesquisador

> Preencher antes de anotar, em uma linha cada, com a data.

Registradas em ___/___/2026, antes da anotação.

- **R1** (quantas das 200 você acha que vão ser desmentido?): ______
- **R2** (os artigos com vários registros erram mais?): ______

---

## Adendos

### Adendo 1 — 03/11/2026, resultado e desvios

**Execução.** A anotação das 200 falsas foi feita em outro computador, numa
cópia do repositório anterior ao commit deste pré-registro (24e8ff4). Por isso
a planilha anotada tem as cinco categorias antigas, sem `desmentido`. Na junção
dos dois ramos (f18bcd4), a planilha anotada substituiu a versão com a
categoria nova. O arquivo escondido do anotador (`amostra_meta.json`), com
`registros_no_artigo` e `flag_desmentido`, foi preservado e continuou fora da
vista do anotador. A apuração foi refeita com o script atual
(`anotacao_manual.py --apurar`), sem alterar nenhuma anotação. Saída:
`relatorios/anotacao/resultado_anotacao.json`.

**Desvios declarados.**

1. **Sem a categoria `desmentido` na anotação.** Os desmentidos encontrados
   foram anotados como `lixo`.
2. **Previsões do pesquisador não registradas.** A §4 ficou em branco antes da
   anotação; R1 e R2 do pesquisador não podem ser confrontadas.
3. **R1 estimada depois da anotação, sem cegamento.** Durante a anotação, dois
   itens foram identificados, nas conversas com o assistente, como desmentidos:
   n = 80 (post de terceiro dizendo que a doença atribuída a uma apresentadora
   foi inventada) e n = 88 (alerta da própria empresa vítima de um golpe). Os
   dois estão como `lixo` na planilha, que não foi alterada. A contagem de
   `desmentido` abaixo usa essa reclassificação e vale como estimativa.

**Resultado.**

| medida | valor |
|---|---|
| não é só o boato (todas as categorias de erro) | 11 de 200 = 5,5% [3,1; 9,6] |
| `desmentido` (estimativa, desvio 3) | 2 de 200 = 1,0% |
| `checador` | 4 de 200 = 2,0% |
| `desmentido` + `checador` | 6 de 200 = 3,0% [1,4; 6,4] |
| erro, artigos com 1 registro | 2,9% [1,1; 7,3] (n = 137) |
| erro, artigos com 2 ou mais registros | 11,1% [5,5; 21,2] (n = 63) |
| itens marcados pela regra | 2 (n = 148 e n = 175, ambos `texto_checador`) |
| precisão da regra na amostra | 2 de 2 (os dois anotados como `checador`) |
| cobertura da regra (`desmentido` + `checador`) | 2 de 6 = 33% |

Os dois desmentidos (n = 80 e n = 88) vêm de artigos com 1 registro e não
foram marcados pela regra, que foi escrita para nota oficial, negação de
autoria e texto do checador, e não para posts de terceiros incorporados ao
artigo.

**Confronto com as previsões (§2).**

| | resultado | previsão |
|---|---|---|
| R1 `desmentido` | 1,0% (estimativa) | 1% a 5% ✓ (no limite inferior) |
| R2 2+ registros contra 1 | 11,1% contra 2,9%: +8,2 pp | maior em ≥ 5 pp ✓ |
| R3 itens marcados pela regra | 2 | 0 a 4 ✓ |
| R4 cobertura da regra | 33% | ≤ 50% ✓ |

**Leituras fixadas (§3).**

- `desmentido` + `checador` = 3,0%, abaixo de 5% → **o ruído de rótulo desse
  tipo é pequeno**; entra como limitação medida, com a taxa e o IC, e os
  números do TCC não mudam.
- R2 confirmada → **a versão publicada da base traz `registros_no_artigo`**
  como atributo, para quem quiser filtrar os artigos com várias mensagens.

**Nota.** Um rascunho deste adendo foi escrito no outro computador e não foi
enviado ao repositório. Este é o adendo válido; o rascunho deve ser
descartado.
