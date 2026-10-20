"""!
@file trabalho.py
@brief Pipeline de aprendizado supervisionado para predição de atropelamentos de fauna
silvestre (Projeto RodoFauna - IBRAM/DF): análise exploratória, pré-processamento,
classificação binária e multiclasse, regressão, MLP, Optuna e regularização.
@author Bruna Caetano
"""

import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

PASTA = Path(__file__).parent
FIG = PASTA / "figuras"
RES = PASTA / "resultados"
SEMENTE = 42
TAM_CELULA = 500
TAM_BLOCO = 3000
MESES_CHUVOSOS = {10, 11, 12, 1, 2, 3, 4}
TRIALS_OPTUNA = 40
CLASSES_PT = {"birds": "aves", "mammals": "mamíferos", "reptiles": "répteis", "amphibians": "anfíbios"}

CONT_PAINEL = ["lag1", "lag2", "lag3", "lag12", "mediaHist", "vizUlt3", "vizHist"]
NUM_PAINEL = ["x", "y", "mesSin", "mesCos", "chuvoso", "esforco"]
CAT_PAINEL = ["uc"]
NUM_REGISTRO = ["x", "y", "mesSin", "mesCos", "diaSin", "diaCos", "chuvoso"]
CAT_REGISTRO = ["uc", "diaSemana"]

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)
sys.stdout.reconfigure(encoding="utf-8")


def main():
    FIG.mkdir(exist_ok=True)
    RES.mkdir(exist_ok=True)

    bruto = carregarDados()
    silv = bruto[bruto["silvestre_"] == "silvestre"].copy()
    analisarDados(bruto, silv)

    painel = montarPainel(bruto, silv)
    analisarPainel(painel)

    binario = classificarBinario(painel)
    classificarMulticlasse(silv)
    regredir(painel)
    melhor = otimizarOptuna(binario)
    analisarRegularizacao(binario)
    mapearHotspots(binario, melhor)


def carregarDados():
    titulo("ETAPA 1 - ANÁLISE EXPLORATÓRIA")
    df = pd.read_excel(PASTA / "ProgramaRodofaunaAtropelada.xlsx")
    print(f"Linhas no arquivo: {len(df)} | colunas: {df.shape[1]}")
    df = df.drop(columns="OBJECTID").drop_duplicates()
    print(f"Linhas após remover registros duplicados (cada id aparecia 2x): {len(df)}")

    df["dia_da_sem"] = df["dia_da_sem"].replace({"eg_": "seg"})
    df["sexo"] = df["sexo"].str.replace(r"^n.o_identificado$", "nao_identificado", regex=True)
    df["data"] = pd.to_datetime(df["data_dt"], format="%d/%m/%y")

    df["cx"] = df["x"] // TAM_CELULA * TAM_CELULA + TAM_CELULA // 2
    df["cy"] = df["y"] // TAM_CELULA * TAM_CELULA + TAM_CELULA // 2
    df["bloco"] = (df["x"] // TAM_BLOCO).astype(str) + "_" + (df["y"] // TAM_BLOCO).astype(str)
    df["mesNum"] = df["data"].dt.month
    df["anoMonit"] = df["data"].dt.year - (df["mesNum"] < 4)
    df["chuvoso"] = df["mesNum"].isin(MESES_CHUVOSOS).astype(int)
    df["mesSin"] = np.sin(2 * np.pi * df["mesNum"] / 12)
    df["mesCos"] = np.cos(2 * np.pi * df["mesNum"] / 12)
    df["diaAno"] = df["data"].dt.dayofyear
    df["diaSin"] = np.sin(2 * np.pi * df["diaAno"] / 365.25)
    df["diaCos"] = np.cos(2 * np.pi * df["diaAno"] / 365.25)
    df["diaSemana"] = df["data"].dt.dayofweek.map(dict(enumerate(["seg", "ter", "qua", "qui", "sex", "sab", "dom"])))
    return df


def analisarDados(bruto, silv):
    print(f"Registros silvestres: {len(silv)} | domésticos: {(bruto['silvestre_'] == 'domestico').sum()}")
    print(f"Período: {silv['data'].min():%d/%m/%Y} a {silv['data'].max():%d/%m/%Y} | datas de percurso: {bruto['data'].nunique()}")
    originais = list(bruto.columns[: bruto.columns.get_loc("data")])
    print(f"\nAtributos originais ({len(originais)}) e tipos:")
    print(bruto[originais].dtypes.to_string())
    print(f"\nValores ausentes (NaN) por coluna: {int(bruto[originais].isna().sum().sum())} no total")

    naoIdent = (silv[originais] == "nao_identificado").mean().mul(100).round(1)
    naoIdent = naoIdent[naoIdent > 0].sort_values(ascending=False)
    print("\n% de 'nao_identificado' (registros silvestres):")
    print(naoIdent.to_string())

    print("\nEstatísticas descritivas (numéricas, silvestres):")
    print(silv[["ano", "campanha", "periodo", "x", "y"]].describe().round(1).to_string())
    print("\nCorrelação entre atributos numéricos originais (redundância):")
    print(silv[["ano", "campanha", "periodo", "x", "y"]].corr().round(2).to_string())

    concordancia = (silv["dia_da_sem"] == silv["diaSemana"]).mean()
    print(f"\nConcordância dia_da_sem x dia derivado da data: {concordancia:.1%}")
    print("Dias da semana com registros:", sorted(silv["diaSemana"].unique()), "(sem finais de semana)")

    print("\nDistribuição de classe (silvestres):")
    print(silv["classe"].value_counts().to_string())
    especies = silv["especie_nm"].value_counts()
    print(f"\nEspécies distintas: {len(especies)} | com >= 30 registros: {(especies >= 30).sum()} | com 1 registro: {(especies == 1).sum()}")
    print("Registros por UC:", silv["uc"].value_counts().to_dict())
    print("Registros por ano de monitoramento (abr-mar):", silv.groupby("anoMonit").size().to_dict())

    fig, ax = plt.subplots(figsize=(7, 4))
    naoIdent.plot.barh(ax=ax, color="#888")
    ax.set_xlabel("% de registros 'nao_identificado'")
    ax.invert_yaxis()
    salvarFigura(fig, "fig01_nao_identificado.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    silv["classe"].value_counts().plot.bar(ax=axs[0], color="#4c72b0")
    axs[0].set_title("Registros por classe")
    especies.drop("nao_identificado").head(15).plot.barh(ax=axs[1], color="#55a868")
    axs[1].invert_yaxis()
    axs[1].set_title("15 espécies mais atropeladas")
    salvarFigura(fig, "fig02_classes_especies.png")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for classe in CLASSES_PT:
        grupo = silv[silv["classe"] == classe]
        axs[0].scatter(grupo["x"], grupo["y"], s=4, alpha=0.5, label=CLASSES_PT[classe])
    axs[0].legend(markerscale=4)
    axs[0].set_title("Ocorrências por classe (UTM 23S)")
    hb = axs[1].hexbin(silv["x"], silv["y"], gridsize=60, bins="log", cmap="inferno", mincnt=1)
    fig.colorbar(hb, ax=axs[1], label="registros (log)")
    axs[1].set_title("Densidade espacial de atropelamentos")
    for ax in axs:
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    salvarFigura(fig, "fig03_mapa.png")

    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    mensal = silv.groupby([silv["data"].dt.to_period("M"), "classe"]).size().unstack(fill_value=0)
    mensal.index = mensal.index.to_timestamp()
    mensal.drop(columns="nao_identificado").rename(columns=CLASSES_PT).plot(ax=axs[0, 0])
    axs[0, 0].set_title("Série mensal por classe")
    silv.groupby("mesNum").size().div(5).plot.bar(ax=axs[0, 1], color="#c44e52")
    axs[0, 1].set_title("Média de registros por mês do ano")
    silv["diaSemana"].value_counts().reindex(["seg", "ter", "qua", "qui", "sex"]).plot.bar(ax=axs[1, 0], color="#8172b2")
    axs[1, 0].set_title("Registros por dia da semana")
    pd.crosstab(silv["anoMonit"], silv["classe"]).drop(columns="nao_identificado").rename(columns=CLASSES_PT).plot.bar(ax=axs[1, 1], stacked=True)
    axs[1, 1].set_title("Registros por ano de monitoramento (abr-mar)")
    salvarFigura(fig, "fig04_temporal.png")


def montarPainel(bruto, silv):
    titulo("ENGENHARIA DE ATRIBUTOS - PAINEL CÉLULA x MÊS")
    celulas = bruto.groupby(["cx", "cy"]).agg(uc=("uc", lambda s: s.mode()[0]), bloco=("bloco", "first")).reset_index()
    meses = pd.period_range("2010-04", "2015-03", freq="M")
    esforco = bruto.groupby(bruto["data"].dt.to_period("M"))["data"].nunique().reindex(meses, fill_value=0).to_numpy()

    contagem = silv.groupby(["cx", "cy", silv["data"].dt.to_period("M")]).size().unstack(fill_value=0)
    contagem = contagem.reindex(index=pd.MultiIndex.from_frame(celulas[["cx", "cy"]]), columns=meses, fill_value=0)
    m = contagem.to_numpy()

    dx = np.abs(celulas["cx"].to_numpy()[:, None] - celulas["cx"].to_numpy()[None, :])
    dy = np.abs(celulas["cy"].to_numpy()[:, None] - celulas["cy"].to_numpy()[None, :])
    vizinhos = ((dx <= TAM_CELULA) & (dy <= TAM_CELULA) & ~np.eye(len(celulas), dtype=bool)).astype(int)
    v = vizinhos @ m

    partes = []
    for t in range(12, len(meses)):
        parte = celulas.rename(columns={"cx": "x", "cy": "y"}).copy()
        parte["mes"] = meses[t].to_timestamp()
        parte["mesNum"] = meses[t].month
        parte["anoMonit"] = meses[t].year - (meses[t].month < 4)
        parte["esforco"] = esforco[t]
        parte["lag1"], parte["lag2"], parte["lag3"], parte["lag12"] = m[:, t - 1], m[:, t - 2], m[:, t - 3], m[:, t - 12]
        parte["mediaHist"] = m[:, :t].mean(axis=1)
        parte["vizUlt3"] = v[:, t - 3 : t].sum(axis=1)
        parte["vizHist"] = v[:, :t].mean(axis=1)
        parte["cont"] = m[:, t]
        partes.append(parte)
    painel = pd.concat(partes, ignore_index=True)
    painel["chuvoso"] = painel["mesNum"].isin(MESES_CHUVOSOS).astype(int)
    painel["mesSin"] = np.sin(2 * np.pi * painel["mesNum"] / 12)
    painel["mesCos"] = np.cos(2 * np.pi * painel["mesNum"] / 12)
    painel["alvo"] = (painel["cont"] > 0).astype(int)

    print(f"Células de {TAM_CELULA} m na rota monitorada (com qualquer registro, inclusive doméstico): {len(celulas)}")
    print(f"Meses: {len(meses)} (abr/2010-mar/2015); primeiros 12 usados apenas como histórico")
    print(f"Linhas do painel: {len(painel)} | colunas: {painel.shape[1]}")
    print(f"Esforço (dias de percurso/mês): mín {esforco.min()}, média {esforco.mean():.1f}, máx {esforco.max()}")
    return painel


def analisarPainel(painel):
    print(f"\nClasse binária (>=1 atropelamento silvestre na célula no mês): {painel['alvo'].mean():.1%} positivos")
    print(painel["alvo"].value_counts().to_string())
    print("\nDistribuição da contagem por célula-mês:")
    print(painel["cont"].describe().round(3).to_string())
    totalCelula = painel.groupby(["x", "y"])["cont"].sum()
    q1, q3 = totalCelula.quantile([0.25, 0.75])
    print(f"Células outliers (total > Q3 + 1,5 IQR = {q3 + 1.5 * (q3 - q1):.0f}): {(totalCelula > q3 + 1.5 * (q3 - q1)).sum()}")

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    painel["cont"].clip(upper=6).value_counts().sort_index().plot.bar(ax=axs[0], color="#4c72b0")
    axs[0].set_title("Atropelamentos por célula-mês (6 = 6 ou mais)")
    axs[1].boxplot(totalCelula, orientation="horizontal")
    axs[1].set_title("Total por célula (abr/2011-mar/2015)")
    salvarFigura(fig, "fig05_distribuicao_contagem.png")

    colunas = CONT_PAINEL + NUM_PAINEL + ["cont"]
    corr = painel[colunas].corr()
    print("\nPares de atributos com |correlação| > 0,8:")
    for i, a in enumerate(colunas):
        for b in colunas[i + 1 :]:
            if abs(corr.loc[a, b]) > 0.8:
                print(f"  {a} x {b}: {corr.loc[a, b]:.2f}")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(colunas)), colunas, rotation=90)
    ax.set_yticks(range(len(colunas)), colunas)
    for i in range(len(colunas)):
        for j in range(len(colunas)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    ax.set_title("Correlação entre atributos do painel")
    salvarFigura(fig, "fig06_correlacao.png")


def classificarBinario(painel):
    titulo("ETAPA 2/3 - CLASSIFICAÇÃO BINÁRIA (OCORRÊNCIA NA CÉLULA-MÊS)")
    colunas = CONT_PAINEL + NUM_PAINEL + CAT_PAINEL
    tr, va, te = dividirTemporal(painel)
    print(f"Divisão temporal: treino {len(tr)} (abr/2011-mar/2013), validação {len(va)} (abr/2013-mar/2014), teste {len(te)} (abr/2014-mar/2015)")
    print(f"% positivos: treino {tr['alvo'].mean():.1%}, validação {va['alvo'].mean():.1%}, teste {te['alvo'].mean():.1%}")
    prep = criarPreprocessador(CONT_PAINEL, NUM_PAINEL, CAT_PAINEL).fit(tr[colunas])
    xTr, xVa, xTe = (prep.transform(d[colunas]) for d in (tr, va, te))
    yTr, yVa, yTe = tr["alvo"].to_numpy(), va["alvo"].to_numpy(), te["alvo"].to_numpy()
    print(f"Atributos após pré-processamento: {xTr.shape[1]}")

    mlp, hist = treinarMlp(criarMlpBinario(), xTr, yTr, xVa, yVa, medirBinario, epocas=200, paciencia=20)
    print(f"MLP base: {hist['epocas']} épocas (melhor na {hist['melhorEpoca']}), {hist['tempo']:.1f} s")
    plotarCurvas([("MLP base", hist)], "ROC-AUC", "fig07_curvas_mlp_binario.png")

    modelos = {"MLP": mlp}
    tempos = {"MLP": hist["tempo"]}
    for nome, modelo in criarClassificadores().items():
        inicio = time.time()
        modelos[nome] = modelo.fit(xTr, yTr)
        tempos[nome] = time.time() - inicio

    linhas = []
    probs = {}
    for nome, modelo in modelos.items():
        probVa, probTe = modelo.predict_proba(xVa)[:, 1], modelo.predict_proba(xTe)[:, 1]
        limiar = 0.5 if nome.startswith("Dummy") else escolherLimiar(yVa, probVa)
        probs[nome] = (probTe, limiar)
        linhas.append(avaliarBinario(nome, yTe, probTe, limiar) | {"tempo_s": tempos[nome]})
        if nome == "MLP":
            linhas.append(avaliarBinario("MLP (limiar 0,5)", yTe, probTe, 0.5) | {"tempo_s": tempos[nome]})
    tabela = salvarTabela(linhas, "binario_temporal.csv")

    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    for ax, nome in zip(axs, ["MLP", "Regressão Logística", "Random Forest", "Gradient Boosting"]):
        prob, limiar = probs[nome]
        ConfusionMatrixDisplay.from_predictions(yTe, (prob >= limiar).astype(int), ax=ax, colorbar=False)
        ax.set_title(f"{nome} (limiar {limiar:.2f})")
    salvarFigura(fig, "fig08_confusao_binario.png")

    fig, ax = plt.subplots(figsize=(6, 6))
    for nome, (prob, _) in probs.items():
        if nome.startswith("Dummy"):
            continue
        fpr, tpr, _ = roc_curve(yTe, prob)
        ax.plot(fpr, tpr, label=f"{nome} (AUC {roc_auc_score(yTe, prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos")
    ax.legend()
    salvarFigura(fig, "fig09_roc_binario.png")

    importancia = {}
    for nome in ["MLP", "Random Forest"]:
        pipe = make_pipeline(prep, modelos[nome])
        r = permutation_importance(pipe, te[colunas], yTe, scoring="roc_auc", n_repeats=10, random_state=SEMENTE)
        importancia[nome] = pd.Series(r.importances_mean, index=colunas)
    importancia = pd.DataFrame(importancia).sort_values("Random Forest")
    importancia.round(4).to_csv(RES / "importancia_binario.csv")
    print("\nImportância por permutação (queda de ROC-AUC no teste):")
    print(importancia.sort_values("Random Forest", ascending=False).round(4).to_string())
    fig, ax = plt.subplots(figsize=(8, 6))
    importancia.plot.barh(ax=ax)
    ax.set_xlabel("Queda média de ROC-AUC ao permutar o atributo")
    salvarFigura(fig, "fig10_importancia_binario.png")

    titulo("EXPERIMENTO DE DIVISÃO: ALEATÓRIA x ESPACIAL x TEMPORAL (BINÁRIO)")
    compararDivisoes(painel, colunas, (CONT_PAINEL, NUM_PAINEL, CAT_PAINEL), "alvo", "binario")

    return {"tr": tr, "va": va, "te": te, "prep": prep, "colunas": colunas, "x": (xTr, xVa, xTe), "y": (yTr, yVa, yTe), "mlp": mlp, "hist": hist, "tabela": tabela, "rf": modelos["Random Forest"]}


def classificarMulticlasse(silv):
    titulo("ETAPA 3 - CLASSIFICAÇÃO MULTICLASSE (GRUPO TAXONÔMICO)")
    df = silv[(silv["classe"] != "nao_identificado") & (silv["uc"] != "nao_identificado")].copy()
    df["alvo"] = df["classe"].map(CLASSES_PT)
    colunas = NUM_REGISTRO + CAT_REGISTRO
    print(f"Registros: {len(df)} | distribuição: {df['alvo'].value_counts().to_dict()}")
    tr, va, te = dividirAleatorio(df, "alvo")
    print(f"Divisão aleatória estratificada 60/20/20: treino {len(tr)}, validação {len(va)}, teste {len(te)}")
    prep = criarPreprocessador([], NUM_REGISTRO, CAT_REGISTRO).fit(tr[colunas])
    xTr, xVa, xTe = (prep.transform(d[colunas]) for d in (tr, va, te))
    yTr, yVa, yTe = tr["alvo"].to_numpy(), va["alvo"].to_numpy(), te["alvo"].to_numpy()

    pesos = compute_sample_weight("balanced", yTr)
    mlp, hist = treinarMlp(criarMlpMulticlasse(), xTr, yTr, xVa, yVa, medirMulticlasse, epocas=300, paciencia=30, pesos=pesos)
    mlpSemPeso, histSemPeso = treinarMlp(criarMlpMulticlasse(), xTr, yTr, xVa, yVa, medirMulticlasse, epocas=300, paciencia=30)
    print(f"MLP (pesos balanceados): {hist['epocas']} épocas, {hist['tempo']:.1f} s")
    plotarCurvas([("MLP pesos balanceados", hist), ("MLP sem pesos", histSemPeso)], "F1 macro", "fig11_curvas_mlp_multiclasse.png")

    modelos = {"MLP (pesos balanceados)": mlp, "MLP (sem pesos)": mlpSemPeso}
    modelos["Regressão Logística"] = LogisticRegression(max_iter=3000, class_weight="balanced").fit(xTr, yTr)
    modelos["Random Forest"] = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEMENTE).fit(xTr, yTr)
    modelos["Gradient Boosting"] = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEMENTE).fit(xTr, yTr)
    modelos["Dummy (mais frequente)"] = DummyClassifier().fit(xTr, yTr)

    linhas = [avaliarMulticlasse(nome, yTe, m.predict(xTe)) for nome, m in modelos.items()]
    salvarTabela(linhas, "multiclasse_aleatorio.csv")
    for nome in ["MLP (pesos balanceados)", "MLP (sem pesos)", "Random Forest"]:
        print(f"\nMétricas por classe - {nome}:")
        print(classification_report(yTe, modelos[nome].predict(xTe), digits=3, zero_division=0))

    ordem = ["aves", "mamíferos", "répteis", "anfíbios"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    for ax, nome in zip(axs, ["MLP (pesos balanceados)", "MLP (sem pesos)", "Random Forest"]):
        ConfusionMatrixDisplay.from_predictions(yTe, modelos[nome].predict(xTe), labels=ordem, ax=ax, colorbar=False)
        ax.set_title(nome)
    salvarFigura(fig, "fig12_confusao_multiclasse.png")

    titulo("EXPERIMENTO DE DIVISÃO: ALEATÓRIA x ESPACIAL x TEMPORAL (MULTICLASSE)")
    compararDivisoes(df, colunas, ([], NUM_REGISTRO, CAT_REGISTRO), "alvo", "multiclasse")


def regredir(painel):
    titulo("ETAPA 4 - REGRESSÃO (Nº DE ATROPELAMENTOS POR CÉLULA-MÊS)")
    colunas = CONT_PAINEL + NUM_PAINEL + CAT_PAINEL
    tr, va, te = dividirTemporal(painel)
    prep = criarPreprocessador(CONT_PAINEL, NUM_PAINEL, CAT_PAINEL).fit(tr[colunas])
    xTr, xVa, xTe = (prep.transform(d[colunas]) for d in (tr, va, te))
    yTr, yVa, yTe = (d["cont"].to_numpy().astype(float) for d in (tr, va, te))

    mlp, hist = treinarMlp(criarMlpRegressao(), xTr, yTr, xVa, yVa, medirRegressao, epocas=300, paciencia=30)
    print(f"MLP: {hist['epocas']} épocas (melhor na {hist['melhorEpoca']}), {hist['tempo']:.1f} s")
    plotarCurvas([("MLP regressão", hist)], "R²", "fig13_curvas_mlp_regressao.png", "MSE")

    modelos = {"MLP": mlp}
    modelos["Regressão Linear"] = LinearRegression().fit(xTr, yTr)
    modelos["Random Forest"] = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEMENTE).fit(xTr, yTr)
    modelos["Gradient Boosting (Poisson)"] = HistGradientBoostingRegressor(loss="poisson", random_state=SEMENTE).fit(xTr, yTr)

    preds = {nome: np.clip(m.predict(xTe), 0, None) for nome, m in modelos.items()}
    preds["Ingênuo (média histórica)"] = te["mediaHist"].to_numpy()
    linhas = [avaliarRegressao(nome, yTe, p) for nome, p in preds.items()]

    teCelula = te.assign(**{f"p_{k}": v for k, v in preds.items()}).groupby(["x", "y"]).sum(numeric_only=True)
    for linha in linhas:
        linha["R2_celula_ano"] = r2_score(teCelula["cont"], teCelula[f"p_{linha['modelo']}"])
    salvarTabela(linhas, "regressao_temporal.csv")

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    axs[0].scatter(yTe + np.random.default_rng(SEMENTE).uniform(-0.2, 0.2, len(yTe)), preds["MLP"], s=4, alpha=0.3)
    axs[0].plot([0, yTe.max()], [0, yTe.max()], "r--")
    axs[0].set(xlabel="real (com ruído visual)", ylabel="predito", title="MLP: real x predito (célula-mês)")
    axs[1].scatter(preds["MLP"], yTe - preds["MLP"], s=4, alpha=0.3)
    axs[1].axhline(0, color="r", ls="--")
    axs[1].set(xlabel="predito", ylabel="resíduo", title="MLP: resíduos")
    axs[2].scatter(teCelula["cont"], teCelula["p_MLP"], s=10)
    axs[2].plot([0, teCelula["cont"].max()], [0, teCelula["cont"].max()], "r--")
    axs[2].set(xlabel="real", ylabel="predito", title="MLP: total por célula no ano de teste")
    salvarFigura(fig, "fig14_regressao_real_predito.png")


def otimizarOptuna(binario):
    titulo("ETAPA 5 - OTIMIZAÇÃO DE HIPERPARÂMETROS COM OPTUNA (MLP BINÁRIA)")
    xTr, xVa, xTe = binario["x"]
    yTr, yVa, yTe = binario["y"]

    def objetivo(trial):
        p = {
            "nCamadas": trial.suggest_int("nCamadas", 1, 3),
            "ativacao": trial.suggest_categorical("ativacao", ["relu", "tanh", "logistic"]),
            "otimizador": trial.suggest_categorical("otimizador", ["adam", "sgd"]),
            "lr": trial.suggest_float("lr", 1e-4, 1e-1, log=True),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
            "batch": trial.suggest_categorical("batch", [32, 64, 128, 256, 512]),
            "epocas": trial.suggest_int("epocas", 30, 200, step=10),
        }
        p["neuronios"] = [trial.suggest_int(f"neuronios{i}", 8, 256, log=True) for i in range(p["nCamadas"])]
        if p["otimizador"] == "sgd":
            p["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)
        _, hist = treinarMlp(criarMlpOptuna(p), xTr, yTr, xVa, yVa, medirBinario, epocas=p["epocas"], paciencia=15)
        return max(hist["metVal"])

    estudo = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEMENTE))
    inicio = time.time()
    estudo.optimize(objetivo, n_trials=TRIALS_OPTUNA)
    print(f"Trials: {TRIALS_OPTUNA} | tempo total: {time.time() - inicio:.0f} s")
    print(f"Melhor ROC-AUC de validação: {estudo.best_value:.4f}")
    print(f"Melhores hiperparâmetros: {estudo.best_params}")
    estudo.trials_dataframe().to_csv(RES / "optuna_trials.csv", index=False)

    importancias = optuna.importance.get_param_importances(estudo)
    print("Importância dos hiperparâmetros (fANOVA):", {k: round(v, 3) for k, v in importancias.items()})
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    valores = [t.value for t in estudo.trials]
    axs[0].plot(valores, "o", alpha=0.6, label="trial")
    axs[0].plot(np.maximum.accumulate(valores), "r-", label="melhor até o trial")
    axs[0].set(xlabel="trial", ylabel="ROC-AUC (validação)", title="Histórico da otimização")
    axs[0].legend()
    pd.Series(importancias).sort_values().plot.barh(ax=axs[1], color="#55a868")
    axs[1].set_title("Importância dos hiperparâmetros")
    salvarFigura(fig, "fig15_optuna.png")

    p = dict(estudo.best_params)
    p["neuronios"] = [p[f"neuronios{i}"] for i in range(p["nCamadas"])]
    otimizada, hist = treinarMlp(criarMlpOptuna(p), xTr, yTr, xVa, yVa, medirBinario, epocas=p["epocas"], paciencia=15)
    plotarCurvas([("MLP base", binario["hist"]), ("MLP otimizada", hist)], "ROC-AUC", "fig16_curvas_base_otimizada.png")

    linhas = []
    for nome, modelo, h in [("MLP base", binario["mlp"], binario["hist"]), ("MLP otimizada (Optuna)", otimizada, hist)]:
        limiar = escolherLimiar(yVa, modelo.predict_proba(xVa)[:, 1])
        linha = avaliarBinario(nome, yTe, modelo.predict_proba(xTe)[:, 1], limiar)
        linha["aucValidacao"] = max(h["metVal"])
        linha["epocas"] = h["epocas"]
        linha["tempo_s"] = h["tempo"]
        linhas.append(linha)
    salvarTabela(linhas, "optuna_comparacao.csv")
    return otimizada


def analisarRegularizacao(binario):
    titulo("ETAPA 6 - REGULARIZAÇÃO E OVERFITTING (MLP BINÁRIA)")
    xTr, xVa, xTe = binario["x"]
    yTr, yVa, yTe = binario["y"]
    grande = (256, 256, 128)
    configs = [
        ("Grande sem regularização", grande, 0.0, None),
        ("Grande + L2 (alpha=0,01)", grande, 1e-2, None),
        ("Grande + early stopping", grande, 0.0, 15),
        ("Grande + L2 + early stopping", grande, 1e-2, 15),
        ("Reduzida (16)", (16,), 1e-4, None),
    ]
    resultados = []
    linhas = []
    for nome, camadas, alpha, paciencia in configs:
        mlp = MLPClassifier(hidden_layer_sizes=camadas, alpha=alpha, learning_rate_init=1e-3, batch_size=64, random_state=SEMENTE)
        modelo, hist = treinarMlp(mlp, xTr, yTr, xVa, yVa, medirBinario, epocas=150, paciencia=paciencia)
        resultados.append((nome, hist))
        aucTr = roc_auc_score(yTr, modelo.predict_proba(xTr)[:, 1])
        aucVa = roc_auc_score(yVa, modelo.predict_proba(xVa)[:, 1])
        aucTe = roc_auc_score(yTe, modelo.predict_proba(xTe)[:, 1])
        limiar = escolherLimiar(yVa, modelo.predict_proba(xVa)[:, 1])
        f1Te = f1_score(yTe, (modelo.predict_proba(xTe)[:, 1] >= limiar).astype(int))
        lossTr, lossVa = log_loss(yTr, modelo.predict_proba(xTr)), log_loss(yVa, modelo.predict_proba(xVa))
        linhas.append({"config": nome, "epocas": hist["epocas"], "lossTreino": lossTr, "lossVal": lossVa, "aucTreino": aucTr, "aucVal": aucVa, "aucTeste": aucTe, "gapAuc": aucTr - aucVa, "f1Teste": f1Te, "tempo_s": hist["tempo"]})
    salvarTabela(linhas, "regularizacao.csv")
    plotarCurvas(resultados, "ROC-AUC", "fig17_regularizacao.png")


def mapearHotspots(binario, otimizada):
    titulo("ETAPA 7 - IDENTIFICAÇÃO DE HOTSPOTS NO ANO DE TESTE")
    te = binario["te"].copy()
    _, _, xTe = binario["x"]
    te["prob"] = otimizada.predict_proba(xTe)[:, 1]
    celula = te.groupby(["x", "y"]).agg(prob=("prob", "mean"), real=("cont", "sum")).reset_index()
    n = int(np.ceil(0.1 * len(celula)))
    topPrevisto = set(celula.nlargest(n, "prob").index)
    topReal = set(celula.nlargest(n, "real").index)
    print(f"Células no top 10% previsto que também estão no top 10% observado: {len(topPrevisto & topReal)}/{n}")
    print(f"Atropelamentos capturados pelo top 10% previsto: {celula.loc[list(topPrevisto), 'real'].sum()} de {celula['real'].sum()} ({celula.loc[list(topPrevisto), 'real'].sum() / celula['real'].sum():.1%})")
    print(f"Correlação de Spearman (prob. média x total observado por célula): {celula['prob'].corr(celula['real'], method='spearman'):.3f}")
    celula.sort_values("prob", ascending=False).round(4).to_csv(RES / "hotspots_previstos.csv", index=False)

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    sc = axs[0].scatter(celula["x"], celula["y"], c=celula["prob"], cmap="inferno_r", s=18)
    fig.colorbar(sc, ax=axs[0], label="probabilidade média prevista")
    axs[0].set_title("Risco previsto (MLP otimizada) abr/2014-mar/2015")
    sc = axs[1].scatter(celula["x"], celula["y"], c=celula["real"], cmap="inferno_r", s=18)
    fig.colorbar(sc, ax=axs[1], label="atropelamentos observados")
    axs[1].set_title("Atropelamentos observados abr/2014-mar/2015")
    for ax in axs:
        ax.set_aspect("equal")
    salvarFigura(fig, "fig18_hotspots.png")


def compararDivisoes(df, colunas, grupos, alvo, tipo):
    classificacao = tipo == "binario"
    divisoes = {
        "Aleatória": dividirAleatorio(df, alvo),
        "Espacial (blocos 3 km)": dividirEspacial(df),
        "Temporal (teste abr/2014-mar/2015)": dividirTemporal(df),
    }
    linhas = []
    for nomeDivisao, (tr, va, te) in divisoes.items():
        prep = criarPreprocessador(*grupos).fit(tr[colunas])
        xTr, xVa, xTe = (prep.transform(d[colunas]) for d in (tr, va, te))
        yTr, yVa, yTe = tr[alvo].to_numpy(), va[alvo].to_numpy(), te[alvo].to_numpy()
        if classificacao:
            mlp, _ = treinarMlp(criarMlpBinario(), xTr, yTr, xVa, yVa, medirBinario, epocas=200, paciencia=20)
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEMENTE).fit(xTr, yTr)
        else:
            mlp, _ = treinarMlp(criarMlpMulticlasse(), xTr, yTr, xVa, yVa, medirMulticlasse, epocas=300, paciencia=30, pesos=compute_sample_weight("balanced", yTr))
            rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=SEMENTE).fit(xTr, yTr)
        for nome, modelo in [("MLP", mlp), ("Random Forest", rf)]:
            if classificacao:
                limiar = escolherLimiar(yVa, modelo.predict_proba(xVa)[:, 1])
                linha = avaliarBinario(nome, yTe, modelo.predict_proba(xTe)[:, 1], limiar)
            else:
                linha = avaliarMulticlasse(nome, yTe, modelo.predict(xTe))
            linhas.append({"divisao": nomeDivisao, "nTreino": len(tr), "nTeste": len(te)} | linha)
    salvarTabela(linhas, f"{tipo}_divisoes.csv")


def treinarMlp(modelo, xTr, yTr, xVa, yVa, medir, epocas, paciencia=None, pesos=None):
    classes = np.unique(yTr) if isinstance(modelo, MLPClassifier) else None
    hist = {"lossTreino": [], "lossVal": [], "metTreino": [], "metVal": []}
    melhor, melhorMet, semMelhora, melhorEpoca = None, -np.inf, 0, epocas
    inicio = time.time()
    for epoca in range(1, epocas + 1):
        if classes is None:
            modelo.partial_fit(xTr, yTr)
        else:
            modelo.partial_fit(xTr, yTr, sample_weight=pesos, classes=classes)
        for parte, x, y in (("Treino", xTr, yTr), ("Val", xVa, yVa)):
            perda, metrica = medir(modelo, x, y)
            hist["loss" + parte].append(perda)
            hist["met" + parte].append(metrica)
        if paciencia is None:
            continue
        if hist["metVal"][-1] > melhorMet + 1e-4:
            melhor, melhorMet, semMelhora, melhorEpoca = deepcopy(modelo), hist["metVal"][-1], 0, epoca
            continue
        semMelhora += 1
        if semMelhora >= paciencia:
            break
    hist["tempo"] = time.time() - inicio
    hist["epocas"] = len(hist["lossVal"])
    hist["melhorEpoca"] = melhorEpoca if paciencia else hist["epocas"]
    return melhor or modelo, hist


def medirBinario(modelo, x, y):
    prob = modelo.predict_proba(x)
    return log_loss(y, prob, labels=modelo.classes_), roc_auc_score(y, prob[:, 1])


def medirMulticlasse(modelo, x, y):
    return log_loss(y, modelo.predict_proba(x), labels=modelo.classes_), f1_score(y, modelo.predict(x), average="macro")


def medirRegressao(modelo, x, y):
    pred = modelo.predict(x)
    return mean_squared_error(y, pred), r2_score(y, pred)


def criarMlpBinario():
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=1e-3, batch_size=128, alpha=1e-4, random_state=SEMENTE)


def criarMlpMulticlasse():
    return MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=1e-3, batch_size=64, alpha=1e-3, random_state=SEMENTE)


def criarMlpRegressao():
    return MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate_init=1e-3, batch_size=128, alpha=1e-3, random_state=SEMENTE)


def criarMlpOptuna(p):
    return MLPClassifier(
        hidden_layer_sizes=tuple(p["neuronios"]),
        activation=p["ativacao"],
        solver=p["otimizador"],
        learning_rate_init=p["lr"],
        alpha=p["alpha"],
        batch_size=p["batch"],
        momentum=p.get("momentum", 0.9),
        random_state=SEMENTE,
    )


def criarClassificadores():
    return {
        "Regressão Logística": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEMENTE),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=SEMENTE),
        "Dummy (mais frequente)": DummyClassifier(),
    }


def criarPreprocessador(contagens, numericas, categoricas):
    return ColumnTransformer(
        [
            ("contagens", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), contagens),
            ("numericas", StandardScaler(), numericas),
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas),
        ]
    )


def dividirTemporal(df):
    return df[df["anoMonit"] <= 2012], df[df["anoMonit"] == 2013], df[df["anoMonit"] == 2014]


def dividirAleatorio(df, alvo):
    tr, te = train_test_split(df, test_size=0.2, stratify=df[alvo], random_state=SEMENTE)
    tr, va = train_test_split(tr, test_size=0.25, stratify=tr[alvo], random_state=SEMENTE)
    return tr, va, te


def dividirEspacial(df):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMENTE)
    idxTr, idxTe = next(gss.split(df, groups=df["bloco"]))
    tr, te = df.iloc[idxTr], df.iloc[idxTe]
    idxTr, idxVa = next(gss.split(tr, groups=tr["bloco"]))
    return tr.iloc[idxTr], tr.iloc[idxVa], te


def escolherLimiar(y, prob):
    limiares = np.linspace(0.05, 0.95, 91)
    return limiares[np.argmax([f1_score(y, (prob >= l).astype(int), zero_division=0) for l in limiares])]


def avaliarBinario(nome, y, prob, limiar):
    pred = (prob >= limiar).astype(int)
    return {
        "modelo": nome,
        "limiar": limiar,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "rocAuc": roc_auc_score(y, prob),
    }


def avaliarMulticlasse(nome, y, pred):
    return {
        "modelo": nome,
        "accuracy": accuracy_score(y, pred),
        "precisionMacro": precision_score(y, pred, average="macro", zero_division=0),
        "recallMacro": recall_score(y, pred, average="macro"),
        "f1Macro": f1_score(y, pred, average="macro"),
    }


def avaliarRegressao(nome, y, pred):
    mse = mean_squared_error(y, pred)
    return {"modelo": nome, "MAE": mean_absolute_error(y, pred), "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y, pred)}


def plotarCurvas(resultados, nomeMetrica, arquivo, nomeLoss="log loss"):
    fig, axs = plt.subplots(2, len(resultados), figsize=(5 * len(resultados), 7), squeeze=False)
    for i, (nome, hist) in enumerate(resultados):
        axs[0, i].plot(hist["lossTreino"], label="treino")
        axs[0, i].plot(hist["lossVal"], label="validação")
        axs[0, i].set(title=nome, ylabel=nomeLoss)
        axs[1, i].plot(hist["metTreino"], label="treino")
        axs[1, i].plot(hist["metVal"], label="validação")
        axs[1, i].set(xlabel="época", ylabel=nomeMetrica)
        if hist["melhorEpoca"] < hist["epocas"]:
            for ax in axs[:, i]:
                ax.axvline(hist["melhorEpoca"] - 1, color="gray", ls=":")
        axs[0, i].legend()
    salvarFigura(fig, arquivo)


def salvarTabela(linhas, arquivo):
    tabela = pd.DataFrame(linhas)
    tabela.round(4).to_csv(RES / arquivo, index=False)
    print(f"\n[{arquivo}]")
    print(tabela.round(4).to_string(index=False))
    return tabela


def salvarFigura(fig, nome):
    fig.tight_layout()
    fig.savefig(FIG / nome, dpi=120)
    plt.close(fig)


def titulo(texto):
    print(f"\n{'=' * 90}\n{texto}\n{'=' * 90}")


if __name__ == "__main__":
    main()
