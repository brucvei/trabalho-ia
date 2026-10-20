# Roteiro do vídeo — Predição de Atropelamentos de Fauna Silvestre com Redes Neurais

**Duração alvo:** 9 min 30 s (limite: 10 min)
**Apresentadores:** Bruna Caetano, Inácio Boemo e Lucas Aued
**Ritmo de referência:** ~130 palavras por minuto. As falas abaixo somam cerca de 1.200 palavras.

| Bloco | Tema | Tempo | Quem |
|---|---|---|---|
| 1 | Abertura e problema | 0:00 – 1:15 | Bruna |
| 2 | Dados e análise exploratória | 1:15 – 2:45 | Bruna |
| 3 | Metodologia: atributos, pré-processamento e divisão | 2:45 – 4:30 | Inácio |
| 4 | Classificação binária e multiclasse | 4:30 – 6:15 | Inácio |
| 5 | Regressão | 6:15 – 7:00 | Lucas |
| 6 | Optuna e regularização | 7:00 – 8:15 | Lucas |
| 7 | Hotspots e conclusões | 8:15 – 9:30 | Lucas |

---

## Bloco 1 — Abertura e problema (0:00 – 1:15)

**Na tela:** slide de título com o nome do trabalho, a disciplina e os integrantes. Depois, uma foto ou ícone de uma rodovia cortando o cerrado.

**Fala:**

> Olá! Somos Bruna, Inácio e Lucas, e este é o nosso trabalho de Inteligência Artificial sobre a predição de atropelamentos de fauna silvestre com redes neurais.
>
> O atropelamento é uma das maiores causas de morte de animais silvestres no Brasil. Medidas como passagens de fauna, cercas-guia e redutores de velocidade funcionam, mas custam caro, então não dá para instalar em toda a rodovia. A pergunta é: **onde e quando** elas devem ser instaladas?
>
> Para responder, usamos os dados do Projeto RodoFauna, do IBRAM do Distrito Federal, e montamos três problemas de aprendizado supervisionado:
> um **binário**, que prevê se um trecho de estrada vai ter atropelamento em um mês;
> um **multiclasse**, que prevê o grupo do animal atropelado: ave, mamífero, réptil ou anfíbio;
> e uma **regressão**, que estima quantos atropelamentos cada trecho terá.

---

## Bloco 2 — Dados e análise exploratória (1:15 – 2:45)

**Na tela:** tabela-resumo dos dados e depois as figuras `figuras/fig03_mapa.png`, `figuras/fig04_temporal.png` e `figuras/fig02_classes_especies.png`.

**Fala:**

> A base cobre de abril de 2010 a março de 2015, com percursos sistemáticos em quatro trechos de rodovias perto de três unidades de conservação: ESECAE, Parque Nacional de Brasília e JBB/FAL/IBGE.
>
> O primeiro achado foi que **cada registro aparecia duas vezes** no arquivo. Das 10.328 linhas, sobraram 5.164; ficando só a fauna silvestre, ficamos com **4.422 registros**.
>
> *(mostrar o mapa)* No mapa dá para ver que os atropelamentos não são uniformes: há trechos com mais de 100 registros e outros quase vazios. Ou seja, existem hotspots.
>
> *(mostrar o gráfico temporal)* Existe também uma sazonalidade forte: os meses de chuva, de outubro a março, concentram a maior parte dos casos, e répteis e anfíbios acompanham a chuva ainda mais de perto.
>
> *(mostrar as classes)* As classes são muito desbalanceadas: as aves são 68 % dos registros e os anfíbios só 6 %. Entre as espécies é ainda pior: são 180 espécies, mas só 23 têm pelo menos 30 registros. Por isso fizemos a classificação multiclasse **por grupo taxonômico**, e não por espécie.
>
> Por fim, os campos sexo, posição e hora estavam quase todos como "não identificado" e, como manda o enunciado, não foram usados.

---

## Bloco 3 — Metodologia (2:45 – 4:30)

**Na tela:** diagrama simples "registros → células de 500 m × mês → atributos → MLP". Depois, a tabela de atributos e a tabela da divisão temporal.

**Fala:**

> A base só registra **presenças**, ou seja, animais encontrados. Para um problema binário precisamos também de exemplos negativos, e a simples ausência de ponto não prova ausência de risco.
>
> Nossa solução foi dividir a rota em **células de 500 por 500 metros** e usar só as células que comprovadamente fazem parte da rota monitorada, ou seja, que tiveram algum registro em algum momento dos cinco anos. Como a mesma rota era percorrida umas oito vezes por mês, uma dessas células sem registro em um mês foi percorrida e não teve atropelamento. Isso é uma **ausência observada**. No fim temos 299 células, 48 meses e 14.352 linhas, com 17,5 % de positivos.
>
> Os atributos usam **só informação do passado**, para não vazar o alvo: atropelamentos nos últimos meses, a média histórica da célula, a densidade nas células vizinhas, o mês codificado em seno e cosseno, a estação chuvosa, o esforço de amostragem e a unidade de conservação.
>
> No pré-processamento, as contagens passaram por log1p e padronização, as variáveis numéricas foram padronizadas e as categóricas viraram one-hot. Tudo foi ajustado **só no treino**. A padronização é essencial para a MLP: sem ela, a coordenada y, que fica na casa dos 8 milhões, dominaria a rede.
>
> A divisão principal é **temporal**: treino de 2011 a 2013, validação no ano seguinte e teste no último ano. É assim que o modelo seria usado na prática, prevendo o próximo ano. O teste nunca foi usado para escolher hiperparâmetro, limiar ou época.

---

## Bloco 4 — Classificação (4:30 – 6:15)

**Na tela:** a tabela de resultados do binário, depois `figuras/fig07_curvas_mlp_binario.png`, `figuras/fig10_importancia_binario.png` e, por fim, a tabela do multiclasse.

**Fala:**

> A MLP base tem duas camadas ocultas, de 64 e 32 neurônios, com ReLU, otimizador Adam e early stopping.
>
> No problema binário, a MLP chegou a **ROC-AUC de 0,775** no teste, praticamente empatada com o Random Forest, que teve 0,773, e com a Regressão Logística, que teve 0,768.
>
> Aqui aparece uma lição importante: **a accuracy engana**. Um modelo que sempre diz "não tem atropelamento" acerta 85 % das vezes e não serve para nada. A própria MLP com o limiar padrão de 0,5 só encontrava 17 % dos atropelamentos. Quando escolhemos o limiar na validação, em 0,28, o recall subiu para 53 % e o F1 foi de 0,24 para 0,43.
>
> *(mostrar as curvas)* As curvas mostram que a rede entra em overfitting muito rápido: a melhor época foi a sexta, e o early stopping segurou isso.
>
> *(mostrar a importância)* O atributo mais importante, de longe, é o **histórico da própria célula**. Em seguida vem a sazonalidade. Quem já foi hotspot tende a continuar sendo.
>
> Também comparamos três divisões dos dados. A aleatória dá os números mais altos, mas é otimista, porque registros vizinhos no espaço e no tempo vazam entre treino e teste. Nas divisões espacial e temporal a AUC cai só de 1 a 2 centésimos, então o modelo generaliza bem.
>
> No multiclasse a tarefa é bem mais difícil: no mesmo trecho e no mesmo dia morrem aves, répteis e mamíferos. O Random Forest foi o melhor, com F1 macro de 0,43, contra 0,32 da MLP. Os pesos balanceados ajudaram a MLP a encontrar as minorias: o recall dos anfíbios subiu de 16 % para 56 %, à custa das aves.

---

## Bloco 5 — Regressão (6:15 – 7:00)

**Na tela:** a tabela da regressão e `figuras/fig14_regressao_real_predito.png`.

**Fala:**

> Na regressão prevemos o número de atropelamentos por célula e por mês. Na escala mensal, o R² é baixo para todos os modelos, no máximo 0,12. Isso é esperado: a média é 0,25 atropelamento por célula-mês, e prever exatamente qual mês terá um evento raro é basicamente ruído.
>
> Quando somamos as previsões do ano por célula, o R² sobe para até 0,54, e o gráfico real versus predito mostra que os modelos acertam bem **onde** os atropelamentos se concentram.
>
> Curiosamente, o melhor modelo foi a **Regressão Linear**. Depois do log e da padronização, a relação ficou quase linear, e a capacidade extra da MLP só trouxe variância.

---

## Bloco 6 — Optuna e regularização (7:00 – 8:15)

**Na tela:** `figuras/fig15_optuna.png`, a tabela base × otimizada e depois `figuras/fig17_regularizacao.png`.

**Fala:**

> Com o Optuna, fizemos 40 trials buscando a melhor ROC-AUC de validação, variando o número de camadas, os neurônios, a ativação, o otimizador, a taxa de aprendizado, a regularização L2 e o tamanho do batch. O melhor modelo tinha três camadas e ativação logística, e a AUC de validação subiu de 0,758 para 0,766.
>
> Mas esse ganho **não se transferiu para o teste**: lá a AUC ficou em 0,772, um pouco abaixo da base. A conclusão é que o gargalo é a informação disponível nos atributos, e não a arquitetura. Com 40 tentativas no mesmo conjunto de validação, parte do ganho foi ajuste ao ruído.
>
> *(mostrar a regularização)* Para estudar overfitting, treinamos uma rede grande, de 256-256-128 neurônios, sem regularização. Ela decorou o treino, com AUC de 0,9995, mas no teste caiu para 0,60. Com early stopping, a AUC de teste voltou para 0,775. O melhor equilíbrio foi juntar L2 e early stopping: diferença de só 0,026 entre treino e validação, e treino quase dez vezes mais rápido. E uma rede pequena, de 16 neurônios, ficou quase tão boa quanto a melhor.

---

## Bloco 7 — Hotspots e conclusões (8:15 – 9:30)

**Na tela:** `figuras/fig18_hotspots.png` e depois um slide de conclusões em tópicos.

**Fala:**

> Por fim, a aplicação prática. Aplicamos a MLP a todas as células no ano de teste e ordenamos as células pelo risco previsto. A correlação com o que foi de fato observado foi de **0,78**. As **10 % de células** com maior risco previsto concentraram **28 % de todos os atropelamentos** do ano. Ou seja, o modelo permite concentrar o orçamento de mitigação em poucos trechos, e a previsão mensal ainda ajuda a reforçar ações na época das chuvas.
>
> As conclusões principais são estas:
> primeiro, o histórico do trecho e a sazonalidade são o que mais explica os atropelamentos;
> segundo, em dados tabulares pequenos a MLP **não superou** os métodos clássicos, e mais complexidade não trouxe resultado melhor;
> terceiro, com classes desbalanceadas a accuracy engana, e ajustar o limiar faz muita diferença.
>
> Como limitações, a base só tem presenças, não tem a geometria das rodovias e cobre poucos trechos. Como trabalho futuro, incluir chuva diária, tráfego, velocidade e uso do solo permitiria aplicar o modelo até em rodovias que nunca foram monitoradas.
>
> Obrigado!

---

## Dicas de gravação

- Ensaiem com cronômetro. Se passar de 9:45, cortem primeiro as comparações de divisões do multiclasse (Bloco 4) e os detalhes do espaço de busca do Optuna (Bloco 6).
- Deixem cada figura na tela pelo menos 5 segundos e destaquem com o cursor o ponto de que estão falando, por exemplo a linha do early stopping nas curvas.
- Os números citados vêm de `relatorio.md` e `resultados/`. Se rodarem `trabalho.py` de novo, confiram se não mudaram.
