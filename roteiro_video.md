# Roteiro do vídeo — Predição de Atropelamentos de Fauna Silvestre com Redes Neurais

**Duração alvo:** 9 min 40 s (limite: 10 min)
**Apresentadores:** Bruna Caetano, Inácio Boemo e Lucas Aued
**Formato:** gravação de tela com o `trabalho.py` aberto no editor. Cada bloco mostra o trecho de código correspondente e, em seguida, a saída (`resultados/saida.txt` ou CSV) e a figura gerada por ele.
**Ritmo de referência:** ~130 palavras por minuto. As falas abaixo somam cerca de 1.250 palavras.

| Bloco | Tema | Tempo | Quem | Código mostrado |
|---|---|---|---|---|
| 1 | Abertura e visão geral do pipeline | 0:00 – 0:50 | Bruna | `main()` |
| 2 | Carga dos dados e análise exploratória | 0:50 – 2:10 | Bruna | `carregarDados()`, `analisarDados()` |
| 3 | Painel célula × mês, pré-processamento e divisão | 2:10 – 4:10 | Inácio | `montarPainel()`, `criarPreprocessador()`, `dividirTemporal()` |
| 4 | Classificação binária e multiclasse | 4:10 – 6:20 | Inácio | `criarMlpBinario()`, `treinarMlp()`, `escolherLimiar()`, `classificarMulticlasse()` |
| 5 | Regressão | 6:20 – 7:05 | Lucas | `regredir()` |
| 6 | Optuna e regularização | 7:05 – 8:30 | Lucas | `otimizarOptuna()`, `analisarRegularizacao()` |
| 7 | Hotspots e conclusões | 8:30 – 9:40 | Lucas | `mapearHotspots()` |

Os números de linha abaixo se referem à versão atual de `trabalho.py`. No VS Code, `Ctrl+G` leva direto à linha.

---

## Bloco 1 — Abertura e visão geral do pipeline (0:00 – 0:50)

**Na tela:** slide de título por 5 segundos e depois o editor com `main()` (`trabalho.py:75-91`).

**Fala:**

> Olá! Somos Bruna, Inácio e Lucas, e este é o nosso trabalho de Inteligência Artificial sobre a predição de atropelamentos de fauna silvestre com redes neurais, usando os dados do Projeto RodoFauna, do IBRAM do Distrito Federal.
>
> Todo o trabalho está em um único script, o `trabalho.py`. *(apontar `main`)* A função `main` mostra o pipeline inteiro na ordem do enunciado: carregar e analisar os dados, montar o painel de células, e depois os três problemas, que são o **binário**, se um trecho terá atropelamento no mês; o **multiclasse**, qual grupo de animal; e a **regressão**, quantos atropelamentos. No fim vêm o Optuna, a regularização e o mapa de hotspots. Rodando o script, ele gera todas as figuras e tabelas do relatório.

---

## Bloco 2 — Carga dos dados e análise exploratória (0:50 – 2:10)

**Na tela:**
1. `carregarDados()` (`trabalho.py:94-117`), destacando a linha 98 (`drop_duplicates`) e as linhas 105-116 (atributos derivados).
2. A linha 80 de `main()` (filtro de silvestres).
3. `analisarDados()` nas linhas 128-131 (cálculo de "nao_identificado").
4. As figuras `fig03_mapa.png`, `fig04_temporal.png` e `fig02_classes_especies.png`, geradas nas linhas 163-189.

**Fala:**

> *(linha 98)* O primeiro achado foi que **cada registro aparecia duas vezes** no arquivo. Removendo o `OBJECTID` e as duplicatas, as 10.328 linhas viram 5.164. *(linha 80)* Ficando só com a fauna silvestre, temos **4.422 registros**, de abril de 2010 a março de 2015.
>
> *(linhas 105-116)* Aqui já derivamos os atributos da data: o mês codificado em **seno e cosseno**, para que dezembro fique perto de janeiro, a estação chuvosa e o ano de monitoramento, que vai de abril a março. As linhas 105 e 106 já encaixam cada ponto em uma célula de 500 metros.
>
> *(linhas 128-131)* Esta parte mede a porcentagem de "não identificado" por coluna. Sexo, posição e hora estão quase todos assim e, como manda o enunciado, não entram no modelo.
>
> *(mapa)* O mapa mostra que os atropelamentos não são uniformes: há trechos com mais de 100 registros e outros quase vazios. *(gráfico temporal)* E há uma sazonalidade forte: os meses de chuva concentram a maior parte dos casos. *(classes)* As classes são muito desbalanceadas: aves são 68 % e anfíbios só 6 %. Das 180 espécies, só 23 têm 30 registros ou mais, por isso o multiclasse é **por grupo taxonômico**.

---

## Bloco 3 — Painel célula × mês, pré-processamento e divisão (2:10 – 4:10)

**Na tela:**
1. `montarPainel()` (`trabalho.py:192-230`), destacando em ordem: linha 194 (células da rota), linhas 198-200 (matriz de contagem), linhas 202-205 (vizinhos), linhas 208-219 (laço que cria os lags) e linha 224 (alvo).
2. As constantes de atributos nas linhas 63-65.
3. `criarPreprocessador()` (`trabalho.py:637-644`).
4. `dividirTemporal()` (`trabalho.py:647-648`) e a linha 274, onde o pré-processador é ajustado só no treino.

**Fala:**

> A base só registra **presenças**. Para o binário precisamos de exemplos negativos, e a simples ausência de ponto não prova ausência de risco.
>
> *(linha 194)* Nossa solução: as células de 500 metros que entram no painel são só as que tiveram **algum registro, inclusive de animal doméstico**, em algum momento dos cinco anos. Isso prova que a célula está na rota monitorada. Como a rota era percorrida umas oito vezes por mês, uma dessas células sem registro em um mês foi percorrida e não teve atropelamento: é uma **ausência observada**.
>
> *(linhas 198-200)* Aqui montamos uma matriz célula por mês com as contagens. *(linhas 202-205)* A matriz de vizinhança marca as oito células ao redor, e o produto `vizinhos @ m` dá a contagem dos vizinhos em uma única operação.
>
> *(linhas 208-219)* Este laço é o coração da engenharia de atributos. Para cada mês `t`, só usamos colunas **anteriores a `t`**: os lags de 1, 2, 3 e 12 meses, a média histórica e a densidade dos vizinhos. A contagem do próprio mês `t` só aparece na linha 218, e ela é o alvo. Assim não há vazamento. *(linha 224)* O alvo binário é ter pelo menos um atropelamento. No fim são 299 células, 48 meses e 14.352 linhas, com 17,5 % de positivos.
>
> *(`criarPreprocessador`)* O pré-processamento é um `ColumnTransformer`: as contagens passam por `log1p` e padronização, as numéricas por `StandardScaler`, e a unidade de conservação por one-hot. Sem a padronização, a coordenada y, na casa dos 8 milhões, dominaria a MLP.
>
> *(`dividirTemporal` e linha 274)* A divisão principal é **temporal**: treino até março de 2013, validação no ano seguinte e teste no último ano. Repare que o `fit` do pré-processador é feito **só no treino**. O teste nunca foi usado para escolher hiperparâmetro, limiar ou época.

---

## Bloco 4 — Classificação binária e multiclasse (4:10 – 6:20)

**Na tela:**
1. `criarMlpBinario()` (`trabalho.py:603-604`).
2. `treinarMlp()` (`trabalho.py:561-586`), destacando o `partial_fit` (linhas 567-570) e o early stopping (linhas 575-582).
3. `escolherLimiar()` (`trabalho.py:665-667`) e a linha 294, onde ele é chamado com os dados de validação.
4. A tabela `[binario_temporal.csv]` em `resultados/saida.txt` e a figura `fig07_curvas_mlp_binario.png`.
5. A importância por permutação (linhas 320-324) e `fig10_importancia_binario.png`.
6. `dividirEspacial()` (`trabalho.py:657-662`) e a tabela `binario_divisoes.csv`.
7. `classificarMulticlasse()` nas linhas 352-354 e a tabela `multiclasse_aleatorio.csv`.

**Fala:**

> *(`criarMlpBinario`)* A MLP base tem duas camadas ocultas, de 64 e 32 neurônios, ReLU, Adam com taxa de aprendizado 0,001 e batch de 128.
>
> *(`treinarMlp`)* Para ter as curvas de treino e validação por época, escrevemos nosso próprio laço: cada `partial_fit` é uma época, e depois medimos loss e métrica nos dois conjuntos. *(linhas 575-582)* Esse é o early stopping: guardamos uma cópia do melhor modelo e paramos depois de 20 épocas sem melhora na validação.
>
> *(tabela)* No teste, a MLP chegou a **ROC-AUC de 0,775**, empatada com o Random Forest, com 0,773, e a Regressão Logística, com 0,768.
>
> Aqui aparece uma lição importante: **a accuracy engana**. O Dummy, que sempre diz "não tem atropelamento", acerta 85 %. A MLP com o limiar padrão de 0,5 só encontrava 17 % dos atropelamentos. *(`escolherLimiar`)* Esta função testa limiares de 0,05 a 0,95 e escolhe o de maior F1 **na validação**. Com o limiar em 0,28, o recall subiu para 53 % e o F1 foi de 0,24 para 0,43.
>
> *(curvas)* A melhor época foi a sexta: a rede entra em overfitting rápido e o early stopping segura isso. *(importância)* Com a importância por permutação, o atributo mais importante, de longe, é o **histórico da própria célula**, seguido da sazonalidade.
>
> *(`dividirEspacial`)* Também comparamos três divisões. A espacial usa `GroupShuffleSplit` com blocos de 3 km, para que pontos vizinhos não caiam em treino e teste ao mesmo tempo. A aleatória é otimista, mas nas divisões espacial e temporal a AUC cai só 1 a 2 centésimos.
>
> *(linhas 352-354)* No multiclasse, treinamos a MLP com e sem `compute_sample_weight("balanced")`. A tarefa é difícil: o Random Forest foi o melhor, com F1 macro de 0,43, contra 0,32 da MLP. Os pesos fizeram o recall dos anfíbios subir de 16 % para 56 %, à custa das aves.

---

## Bloco 5 — Regressão (6:20 – 7:05)

**Na tela:** `regredir()` (`trabalho.py:381-417`), destacando as linhas 398-399 (clip e baseline ingênuo) e 402-404 (agregação por célula no ano). Depois a tabela `regressao_temporal.csv` e `fig14_regressao_real_predito.png`.

**Fala:**

> A regressão usa o mesmo painel, só que o alvo é a contagem. *(linha 398)* Cortamos previsões negativas em zero e, *(linha 399)* como referência, usamos a média histórica da célula.
>
> Na escala mensal, o R² é baixo para todos, no máximo 0,12. Isso é esperado: a média é 0,25 atropelamento por célula-mês, e acertar o mês exato de um evento raro é quase ruído. *(linhas 402-404)* Por isso também somamos as previsões do ano por célula: aí o R² sobe para até 0,54, e o gráfico mostra que os modelos acertam **onde** os atropelamentos se concentram. O melhor foi a **Regressão Linear**: depois do log e da padronização a relação ficou quase linear.

---

## Bloco 6 — Optuna e regularização (7:05 – 8:30)

**Na tela:**
1. A função `objetivo` dentro de `otimizarOptuna()` (`trabalho.py:425-439`) e a linha 441 (`create_study` com `TPESampler`).
2. `fig15_optuna.png` e a tabela `optuna_comparacao.csv`.
3. A lista `configs` em `analisarRegularizacao()` (`trabalho.py:482-489`).
4. `fig17_regularizacao.png` e a tabela `regularizacao.csv`.

**Fala:**

> *(função `objetivo`)* Este é o espaço de busca do Optuna: de uma a três camadas, de 8 a 256 neurônios em escala log, ativação, Adam ou SGD com momentum, taxa de aprendizado, L2, batch e épocas. *(linha 439)* O objetivo é a melhor ROC-AUC **de validação**; o teste não aparece aqui. Foram 40 trials.
>
> *(figura e tabela)* O melhor modelo tinha três camadas e ativação logística, e a AUC de validação subiu de 0,758 para 0,766. Mas o ganho **não se transferiu para o teste**: lá ficou em 0,772, um pouco abaixo da base. O gargalo é a informação nos atributos, não a arquitetura.
>
> *(lista `configs`)* Para estudar overfitting, treinamos estas cinco configurações. A primeira é uma rede grande, 256-256-128, sem nenhuma regularização. *(figura)* Ela decorou o treino, com AUC de 0,9995, e caiu para 0,60 no teste. Com early stopping, voltou para 0,775. O melhor equilíbrio foi L2 mais early stopping: diferença de só 0,026 entre treino e validação e treino quase dez vezes mais rápido. E a rede reduzida, de 16 neurônios, ficou quase tão boa quanto a melhor.

---

## Bloco 7 — Hotspots e conclusões (8:30 – 9:40)

**Na tela:** `mapearHotspots()` (`trabalho.py:507-530`), destacando as linhas 511-518. Depois `fig18_hotspots.png` e um slide de conclusões em tópicos.

**Fala:**

> *(linhas 511-518)* Por fim, a aplicação prática. Aplicamos a MLP otimizada a todas as células no ano de teste, tiramos a probabilidade média de cada célula e comparamos com o que foi observado. *(figura)* A correlação de Spearman foi de **0,78**, e as **10 % de células** com maior risco previsto concentraram **28 % de todos os atropelamentos** do ano. Ou seja, o modelo permite concentrar o orçamento de mitigação em poucos trechos.
>
> *(slide)* As conclusões principais: primeiro, o histórico do trecho e a sazonalidade são o que mais explica os atropelamentos; segundo, em dados tabulares pequenos a MLP **não superou** os métodos clássicos, e mais complexidade não trouxe resultado melhor; terceiro, com classes desbalanceadas a accuracy engana, e ajustar o limiar faz muita diferença.
>
> Como limitações, a base só tem presenças, não tem a geometria das rodovias e cobre poucos trechos. Incluir chuva, tráfego, velocidade e uso do solo permitiria aplicar o modelo até em rodovias nunca monitoradas.
>
> Obrigado!

---

## Dicas de gravação

- **Não rodem o script ao vivo.** O Optuna e os experimentos de divisão levam vários minutos. Mostrem a saída já pronta em `resultados/saida.txt` e os CSVs de `resultados/`.
- Aumentem a fonte do editor (`Ctrl` + `+`) para pelo menos 18 pt e usem um tema claro ou de alto contraste; código pequeno fica ilegível no vídeo.
- Dobrem as funções que não estão sendo explicadas (`Ctrl+K Ctrl+0` no VS Code) e abram só a do bloco atual. Selecionem as linhas citadas enquanto falam delas.
- Deixem cada trecho de código e cada figura na tela pelo menos 5 segundos.
- Ensaiem com cronômetro. Se passar de 9:45, cortem primeiro a comparação de divisões (Bloco 4) e a lista detalhada do espaço de busca do Optuna (Bloco 6).
- Os números citados vêm de `relatorio.md` e `resultados/`. Se mexerem em `trabalho.py`, confiram os números e os números de linha deste roteiro.
