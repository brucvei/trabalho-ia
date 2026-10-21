# Explicação do código — `trabalho.py`

Este guia explica, função por função, o que o `trabalho.py` faz e por quê. No fim há perguntas que os professores podem fazer, com respostas sugeridas. Os números de linha se referem à versão atual do script.

Para rodar: `pip install -r requirements.txt` e depois `python trabalho.py`. O script lê `ProgramaRodofaunaAtropelada.xlsx`, grava as figuras em `figuras/` e as tabelas em `resultados/`, e imprime tudo no terminal (a saída da última execução está em `resultados/saida.txt`).

---

## 1. Configuração inicial (linhas 1-72)

- **Imports:** pandas e numpy para os dados, matplotlib para as figuras, scikit-learn para os modelos e métricas, e Optuna para a busca de hiperparâmetros.
- `matplotlib.use("Agg")` faz o matplotlib salvar as figuras em arquivo sem abrir janelas.
- **Constantes:**
  - `SEMENTE = 42` deixa os resultados reprodutíveis.
  - `TAM_CELULA = 500` é o tamanho, em metros, da célula espacial usada no painel.
  - `TAM_BLOCO = 3000` é o tamanho dos blocos usados na divisão espacial.
  - `MESES_CHUVOSOS` vai de outubro a abril, o período de chuva no cerrado do DF.
  - `TRIALS_OPTUNA = 40` é o número de tentativas do Optuna.
- **Listas de atributos:**
  - `CONT_PAINEL` são as contagens do passado: lags, média histórica e vizinhos.
  - `NUM_PAINEL` são as numéricas: coordenadas, mês em seno e cosseno, estação chuvosa e esforço.
  - `CAT_PAINEL` é a categórica `uc`.
  - `NUM_REGISTRO` e `CAT_REGISTRO` são os atributos do multiclasse, que trabalha registro a registro.
- Os `warnings.filterwarnings` silenciam avisos esperados, como o de convergência do `partial_fit`, que é chamado uma época por vez.

## 2. `main()` (linhas 75-91)

É o roteiro do pipeline, na ordem das etapas do enunciado:

1. Carrega os dados e filtra só os silvestres.
2. Faz a análise exploratória.
3. Monta o painel célula × mês e analisa esse painel.
4. Roda a classificação binária, a multiclasse e a regressão.
5. Roda o Optuna, a análise de regularização e o mapa de hotspots.

O dicionário devolvido por `classificarBinario` (dados divididos, pré-processador, MLP base) é reaproveitado pelo Optuna, pela regularização e pelos hotspots. Assim todos usam exatamente a mesma divisão.

## 3. `carregarDados()` (linhas 94-117)

- Lê a planilha e remove `OBJECTID` e as **linhas duplicadas**. Cada registro aparecia duas vezes, e as 10.328 linhas viram 5.164.
- Corrige inconsistências de texto: `"eg_"` vira `"seg"`, e as variações de "não identificado" no sexo são padronizadas.
- Converte `data_dt` em data de verdade.
- Cria os atributos derivados:
  - `cx` e `cy`: o centro da célula de 500 m em que o ponto cai. A divisão inteira `//` arredonda a coordenada para baixo.
  - `bloco`: identificador do bloco de 3 km, usado na divisão espacial.
  - `anoMonit`: ano de monitoramento, que vai de abril a março, como as campanhas do projeto.
  - `chuvoso`: 1 se o mês é de chuva.
  - `mesSin`/`mesCos` e `diaSin`/`diaCos`: codificação cíclica do mês e do dia do ano.
  - `diaSemana`: dia da semana recalculado a partir da data.

## 4. `analisarDados()` (linhas 120-189)

Faz a Etapa 1 (análise exploratória):

- Imprime a contagem de silvestres e domésticos, o período, os tipos dos atributos e os valores ausentes.
- Calcula a **% de "nao_identificado"** por coluna. Sexo, posição e hora são quase 100 % não identificados.
- Mostra as estatísticas descritivas e a correlação entre `ano`, `campanha`, `periodo`, `x` e `y`. Ano, campanha e período são redundantes entre si.
- Confere se o `dia_da_sem` original bate com o dia calculado pela data, e mostra que não há registros em finais de semana.
- Mostra a distribuição das classes e das espécies: 180 espécies, só 23 com 30 registros ou mais.
- Gera as figuras:
  - `fig01`: % de não identificado.
  - `fig02`: classes e espécies.
  - `fig03`: mapa de pontos e densidade.
  - `fig04`: série mensal, sazonalidade, dia da semana e ano.

## 5. `montarPainel()` (linhas 192-230) — a parte mais importante

Transforma registros de presença em uma tabela **célula × mês** com exemplos positivos e negativos.

- **Linha 194, células da rota:** pega todas as células com **qualquer registro, inclusive doméstico**, em qualquer momento. Isso garante que a célula está no trajeto monitorado. Resultado: 299 células.
- **Linha 196, esforço:** número de dias de percurso em cada mês, contados a partir das datas distintas.
- **Linhas 198-200, matriz `m`:** uma matriz de 299 células × 60 meses com o número de atropelamentos silvestres. Os meses sem registro ficam com 0.
- **Linhas 202-205, vizinhança:** monta uma matriz 299 × 299 que vale 1 quando duas células são vizinhas, ou seja, quando a distância é de até 500 m em x e em y, sem contar a própria célula. O produto `vizinhos @ m` soma, para cada célula e cada mês, os atropelamentos das células vizinhas.
- **Linhas 208-219, laço por mês:** para cada mês `t`, a partir do 13º, cria uma linha por célula. Os atributos usam só os meses **antes de `t`**:
  - `lag1`, `lag2`, `lag3` e `lag12`: contagem de 1, 2, 3 e 12 meses atrás.
  - `mediaHist`: média de todos os meses anteriores.
  - `vizUlt3` e `vizHist`: atropelamentos nos vizinhos nos últimos 3 meses e a média histórica deles.
  - `cont = m[:, t]`: a contagem do próprio mês. Ela é o alvo e nunca entra como atributo.

  Os 12 primeiros meses servem só de histórico, para que exista `lag12`.
- **Linha 224, alvo binário:** `alvo = cont > 0`.

Resultado: 14.352 linhas (299 × 48), com 17,5 % de positivos.

## 6. `analisarPainel()` (linhas 233-265)

- Mostra o balanceamento do alvo binário e a distribuição da contagem: a maioria das células-mês tem 0 atropelamentos.
- Detecta células **outliers** pela regra do IQR (Q3 + 1,5 × IQR) e gera a `fig05`.
- Lista os pares de atributos com correlação acima de 0,8 e gera a matriz de correlação (`fig06`).

## 7. `classificarBinario()` (linhas 268-337)

1. Faz a divisão temporal e ajusta o pré-processador **só no treino** (linha 274). Depois aplica o `transform` em treino, validação e teste.
2. Treina a MLP base com `treinarMlp`, usando no máximo 200 épocas e paciência de 20, e gera as curvas (`fig07`).
3. Treina os modelos clássicos sobre os mesmos dados: Regressão Logística, Random Forest, Gradient Boosting e o Dummy como referência.
4. Para cada modelo, escolhe o limiar na validação (`escolherLimiar`) e avalia no teste. Para a MLP também registra o resultado com o limiar fixo de 0,5, para mostrar a diferença.
5. Gera as matrizes de confusão (`fig08`) e as curvas ROC (`fig09`).
6. Calcula a **importância por permutação**: embaralha um atributo de cada vez no teste e mede quanto a ROC-AUC cai (`fig10`). O pré-processador entra junto no pipeline, então a importância é medida nos atributos originais.
7. Chama `compararDivisoes` para o experimento aleatória × espacial × temporal.

## 8. `classificarMulticlasse()` (linhas 340-378)

- Usa os registros individuais (não o painel) e prevê o grupo: aves, mamíferos, répteis ou anfíbios. Remove os registros com classe ou UC não identificada.
- Os atributos são o que se sabe **antes** de achar o animal: local, data, dia da semana e UC. Espécie, ordem e família ficam de fora, porque revelariam a resposta.
- A divisão é aleatória estratificada 60/20/20, que mantém a proporção das classes em cada parte.
- Treina duas MLPs, uma com `compute_sample_weight("balanced")` (peso maior para as classes raras) e outra sem pesos. Compara com Regressão Logística, Random Forest e Gradient Boosting, todos com `class_weight` balanceado.
- Gera as métricas por classe (`classification_report`), as matrizes de confusão (`fig12`) e o experimento de divisões.

## 9. `regredir()` (linhas 381-417)

- Usa o mesmo painel e a mesma divisão temporal do binário, mas o alvo é `cont`, o número de atropelamentos.
- Compara a MLP (`MLPRegressor`) com Regressão Linear, Random Forest e Gradient Boosting com perda **Poisson**, adequada para contagens.
- Linha 398: corta as previsões negativas em 0, porque não existe contagem negativa.
- Linha 399: usa a média histórica da célula como **baseline ingênuo**.
- Linhas 402-404: além do R² mensal, soma as previsões do ano por célula e calcula o R² nessa escala (`R2_celula_ano`).
- `fig14`: real × predito, resíduos e total por célula.

## 10. `otimizarOptuna()` (linhas 420-475)

- A função `objetivo` sorteia os hiperparâmetros:
  - de 1 a 3 camadas, cada uma com 8 a 256 neurônios em escala log;
  - ativação `relu`, `tanh` ou `logistic`;
  - otimizador `adam` ou `sgd`, este último com momentum;
  - taxa de aprendizado de 1e-4 a 1e-1 e L2 (`alpha`) de 1e-6 a 1e-1;
  - batch de 32 a 512 e de 30 a 200 épocas.
- A função treina a MLP com early stopping e devolve a **melhor ROC-AUC de validação**. O teste não aparece em nenhum momento.
- O `TPESampler` usa os trials anteriores para propor os próximos, o que é mais eficiente que busca aleatória.
- Salva todos os trials e calcula a importância dos hiperparâmetros (fANOVA). A `fig15` mostra o histórico e as importâncias.
- Retreina a melhor configuração e compara base × otimizada no teste (`optuna_comparacao.csv` e `fig16`).

## 11. `analisarRegularizacao()` (linhas 478-504)

Treina cinco configurações por até 150 épocas e compara:

| Configuração | O que testa |
|---|---|
| Grande sem regularização (256-256-128) | provocar overfitting |
| Grande + L2 (`alpha=0,01`) | penalização de pesos |
| Grande + early stopping | parada antecipada |
| Grande + L2 + early stopping | as duas juntas |
| Reduzida (16 neurônios) | redução da complexidade |

Para cada uma, registra a loss e a AUC de treino, validação e teste, o **gap** treino − validação, o F1 e o tempo (`regularizacao.csv` e `fig17`).

## 12. `mapearHotspots()` (linhas 507-530)

- Aplica a MLP otimizada a todas as células no ano de teste e calcula a probabilidade média por célula.
- Compara com o total observado:
  - quantas células do top 10 % previsto estão também no top 10 % real;
  - quantos atropelamentos o top 10 % previsto capturou (28 %);
  - a correlação de Spearman (0,78).
- Salva o ranking em `hotspots_previstos.csv` e os mapas lado a lado em `fig18`.

## 13. Funções auxiliares (linhas 533-729)

- **`compararDivisoes`:** treina a MLP e o Random Forest nas três divisões e salva uma tabela comparativa.
- **`treinarMlp`:** laço de treino próprio.
  - Cada `partial_fit` é uma época. Depois dela a função mede a loss e a métrica em treino e validação, o que gera as curvas.
  - Se houver `paciencia`, guarda uma cópia (`deepcopy`) do melhor modelo e para quando a validação não melhora por `paciencia` épocas seguidas. Esse é o **early stopping**.
  - Devolve o melhor modelo e o histórico.
- **`medirBinario`, `medirMulticlasse`, `medirRegressao`:** devolvem a loss e a métrica principal de cada problema (ROC-AUC, F1 macro e R²).
- **`criarMlp*`:** as arquiteturas base, todas 64-32 com ReLU e Adam, e a MLP montada com os parâmetros do Optuna.
- **`criarClassificadores`:** os modelos clássicos do binário.
- **`criarPreprocessador`:** um `ColumnTransformer` com três ramos:
  - contagens: `log1p` seguido de `StandardScaler`;
  - numéricas: `StandardScaler`;
  - categóricas: `OneHotEncoder`, com `handle_unknown="ignore"`.
- **As três divisões:**
  - `dividirTemporal`: treino nos anos de monitoramento até 2012, validação em 2013 e teste em 2014.
  - `dividirAleatorio`: 60/20/20 estratificado.
  - `dividirEspacial`: usa `GroupShuffleSplit` por bloco de 3 km, de modo que um bloco inteiro fica só em treino, só em validação ou só em teste.
- **`escolherLimiar`:** testa 91 limiares entre 0,05 e 0,95 e devolve o de maior F1.
- **`avaliar*`:** calculam as métricas pedidas no enunciado.
- **`plotarCurvas`, `salvarTabela`, `salvarFigura`, `titulo`:** cuidam da saída (gráficos, CSV e terminal).

---

## Perguntas e respostas prováveis

### Dados e problema

**1. Como vocês obtiveram exemplos negativos, se a base só tem presenças?**
Dividimos o espaço em células de 500 m e usamos só as células com algum registro, inclusive doméstico, em algum momento. Isso prova que a célula faz parte da rota percorrida. Como a rota era percorrida cerca de oito vezes por mês, uma célula dessas sem registro num mês foi observada e não teve atropelamento. É uma ausência observada, e não a mera falta de ponto.

**2. Por que células de 500 m e escala mensal?**
O próprio enunciado sugere segmentos de 500 m. Células menores ficariam quase sempre vazias e muito sensíveis ao erro do GPS, e células maiores misturariam trechos com risco diferente. O mês é a menor escala em que o esforço é estável (cerca de 8 percursos) e ainda captura a sazonalidade. Em escala diária, praticamente tudo seria zero.

**3. E as células da rota que nunca tiveram nenhum registro?**
Ficaram de fora, porque não temos como provar que foram percorridas. Isso é uma limitação: o modelo só prevê para trechos já monitorados. Com a geometria das rodovias seria possível incluir esses trechos.

**4. Por que remover as duplicatas?**
Cada `id` aparecia exatamente duas vezes, com os mesmos dados. Manter as duas cópias dobraria as contagens e, no multiclasse, poderia colocar a mesma linha no treino e no teste, inflando o resultado.

**5. Por que multiclasse por grupo e não por espécie?**
São 180 espécies e só 23 têm 30 registros ou mais. Muitas aparecem uma única vez. O enunciado recomenda agrupar nesse caso.

**6. Por que não usaram sexo, posição e hora?**
Estão quase 100 % como "não identificado" e o enunciado proíbe usá-los. Ordem, família e nome popular também revelam a classe, o que seria vazamento do alvo.

### Engenharia de atributos e pré-processamento

**7. Como garantem que não há vazamento de informação?**
- No painel, todo atributo de contagem usa só meses anteriores ao mês previsto (`m[:, :t]`, `t-1` etc.). A contagem do mês `t` só vira alvo.
- O pré-processador é ajustado só no treino.
- O limiar, o early stopping e o Optuna usam só a validação.

**8. Por que seno e cosseno no mês?**
Com o mês como número de 1 a 12, dezembro e janeiro ficam distantes, mas na verdade são vizinhos. Seno e cosseno colocam os meses num círculo.

**9. Por que `log1p` nas contagens?**
As contagens são muito assimétricas: muitos zeros e poucas células com valores altos. O log reduz o peso dos extremos, e o `+1` lida com o zero.

**10. Por que padronizar? Que diferença faz para a MLP?**
A MLP é treinada por gradiente. Com escalas muito diferentes (y ≈ 8 milhões, contagens de 0 a 10), os pesos de alguns atributos dominariam e o treino ficaria instável. Árvores, como o Random Forest, não dependem de escala, e essa diferença responde à pergunta do enunciado sobre sensibilidade ao pré-processamento.

**11. Por que one-hot na `uc` e não label encoding?**
A UC é nominal. Com label encoding, ESECAE = 0 e JBB = 2 criariam uma ordem que não existe.

**12. Os atributos `lag` e `mediaHist` não são muito correlacionados?**
Sim, e isso aparece na `fig06`. Para a MLP e as árvores isso não é um problema grave de desempenho. Só atrapalha a interpretação da importância, porque permutar um atributo deixa a informação ainda disponível no outro.

### Divisão e avaliação

**13. Por que a divisão principal é temporal?**
Simula o uso real: treinar com o passado e prever o ano seguinte. Uma divisão aleatória colocaria meses do futuro no treino.

**14. A divisão aleatória é otimista? Quanto?**
Sim. Pontos vizinhos no espaço e no tempo caem em treino e teste. No binário, as divisões espacial e temporal tiveram AUC só de 1 a 2 centésimos menor que a aleatória, então o modelo generaliza razoavelmente.

**15. Como foi feita a divisão espacial?**
Com `GroupShuffleSplit`, usando como grupo o bloco de 3 km. Cada bloco inteiro fica só em treino, só em validação ou só em teste, o que evita que células vizinhas vazem entre os conjuntos.

**16. Por que a accuracy engana aqui?**
Só 15 % das células-mês do teste têm atropelamento. O Dummy, que sempre diz "não", tem 85 % de accuracy e recall zero. Por isso usamos ROC-AUC, F1 e recall.

**17. Por que mudar o limiar de 0,5? Isso não é trapaça?**
Com classes desbalanceadas, as probabilidades da rede raramente passam de 0,5, e com esse limiar a MLP só encontrava 17 % dos atropelamentos. O limiar foi escolhido na validação, maximizando o F1, e aplicado no teste sem ajuste. Assim não há trapaça. Com o limiar em 0,28, o recall foi para 53 % e o F1 de 0,24 para 0,43.

**18. Qual atributo é mais importante?**
O histórico da própria célula (`mediaHist` e os lags), seguido da sazonalidade. A importância foi medida por permutação: embaralhamos o atributo e medimos quanto a AUC cai no teste.

### MLP e treinamento

**19. Por que a arquitetura 64-32 com ReLU e Adam?**
É uma rede pequena para um conjunto de 7.176 exemplos de treino e 16 atributos depois do pré-processamento. Duas camadas são suficientes para relações não lineares sem excesso de parâmetros. ReLU evita a saturação do gradiente, e o Adam ajusta a taxa por peso e funciona bem com o padrão de 0,001. O batch de 128 equilibra o ruído do gradiente e a velocidade.

**20. Por que vocês escreveram o laço de treino em vez de usar o `early_stopping=True` do scikit-learn?**
- O `early_stopping` nativo separa sozinho 10 % do treino como validação, aleatoriamente. Nós queríamos usar a nossa validação temporal.
- Ele monitora só a accuracy, e nós queríamos a ROC-AUC.
- Precisávamos guardar a loss e a métrica de treino e validação em cada época para gerar as curvas.

**21. A MLP foi melhor que os modelos clássicos?**
Não de forma significativa. No binário, a MLP ficou com AUC de 0,775, contra 0,773 do Random Forest e 0,768 da Regressão Logística. No multiclasse, o Random Forest foi bem melhor (F1 macro de 0,43 contra 0,32), e na regressão a Regressão Linear foi a melhor. Em dados tabulares pequenos isso é comum.

**22. Por que a MLP perdeu no multiclasse?**
A informação disponível é fraca. No mesmo trecho e no mesmo dia morrem animais de vários grupos. As árvores lidam melhor com interações locais entre coordenada e data, e a classe majoritária (aves, 68 %) puxa a rede.

**23. O que os pesos balanceados fizeram?**
Aumentaram o peso das classes raras na loss. O recall dos anfíbios subiu de 16 % para 56 %, em troca de perder recall nas aves. O F1 macro valoriza esse equilíbrio.

### Regressão

**24. Por que o R² é tão baixo?**
A média é de 0,25 atropelamento por célula-mês, quase tudo zero ou um. Acertar o mês exato de um evento raro é basicamente ruído (o processo é parecido com um Poisson). Agregando o ano por célula, o R² sobe para cerca de 0,54. O modelo acerta onde os atropelamentos se concentram, mas não exatamente quando.

**25. Por que a Regressão Linear ganhou?**
Depois do `log1p` e da padronização, a relação entre o histórico e a contagem futura ficou quase linear. A capacidade extra da MLP só aumentou a variância.

**26. Por que Gradient Boosting com perda Poisson?**
Porque o alvo é uma contagem não negativa, que é o caso típico de uma distribuição de Poisson.

### Optuna e regularização

**27. Como o Optuna funciona e qual foi a função objetivo?**
Ele usa o TPE, que modela quais regiões do espaço de busca deram bons resultados e propõe novos pontos nelas. A função objetivo foi a máxima ROC-AUC de validação. Foram 40 trials, e o teste nunca foi usado.

**28. Por que o modelo otimizado não melhorou no teste?**
A AUC de validação subiu de 0,758 para 0,766, mas a de teste ficou em 0,772 contra 0,775 da base. Com 40 tentativas no mesmo conjunto de validação, parte do ganho é ajuste ao ruído da validação. O limite está nos atributos, não na arquitetura.

**29. Por que os limites do espaço de busca são esses?**
- De 1 a 3 camadas e de 8 a 256 neurônios cobre desde uma rede menor que a base até uma bem maior.
- A taxa de aprendizado e o L2 variam várias ordens de grandeza, por isso são buscados em escala log.
- Os batches de 32 a 512 cobrem do treino ruidoso ao estável.

**30. A rede teve overfitting?**
Sim. A rede grande sem regularização chegou a AUC de 0,9995 no treino e só 0,60 no teste, ou seja, decorou o treino. No fim do treino, a loss de treino era 0,03 e a de validação 1,96. Nas curvas, a loss de validação sobe enquanto a de treino continua caindo.

**31. Qual regularização funcionou melhor?**
L2 junto com early stopping: o gap entre treino e validação ficou em 0,026 e o treino foi cerca de dez vezes mais rápido. O early stopping sozinho já trouxe a AUC de teste de volta para 0,775. A rede de 16 neurônios ficou quase tão boa quanto a melhor, o que mostra que complexidade não era o que faltava.

**32. Por que não usaram dropout?**
O `MLPClassifier` do scikit-learn não tem dropout. O enunciado pede pelo menos uma técnica, e usamos três: L2, early stopping e redução da complexidade. Para usar dropout seria preciso Keras ou PyTorch.

### Aplicação e limitações

**33. Como o modelo seria usado na prática?**
Ordenando as células pelo risco previsto para o próximo ano. As 10 % mais arriscadas concentraram 28 % dos atropelamentos, quase três vezes o esperado ao acaso. O órgão ambiental pode priorizar passagens de fauna e redutores nesses trechos e reforçar ações nos meses de chuva.

**34. Quais as principais limitações?**
- A base só tem presenças.
- Não há geometria das rodovias, então trechos nunca registrados ficam de fora.
- São poucos trechos, todos perto de três UCs.
- Não há chuva diária, tráfego nem velocidade.
- O esforço de amostragem varia.
- Carcaças podem ser removidas por carniceiros antes do percurso.

**35. O que melhoraria a predição?**
Precipitação diária, volume de tráfego, velocidade máxima, uso e cobertura do solo e distância a corpos d'água e à vegetação. Com esses dados o modelo também poderia ser aplicado a rodovias nunca monitoradas, sem depender do histórico.

**36. Os resultados são reprodutíveis?**
Sim. Todas as sementes estão fixadas em `SEMENTE = 42`, inclusive o sampler do Optuna, e o script gera todas as figuras e tabelas do zero. Podem existir variações pequenas entre versões de bibliotecas, como scikit-learn.
