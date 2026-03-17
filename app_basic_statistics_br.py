"""
Basic Statistics for Analytical Chemistry
Interactive course for teaching statistics concepts
Author: Ricardo M. Borges
"""
import io
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy import stats

st.set_page_config(
    page_title="Basic Statistics for Analytical Chemistry",
    layout="wide",
)

# -----------------------------
# LOGOs (optional)
# -----------------------------

STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["LAABio.png", "Basic_Statistics_Course.png"]:
    p = STATIC_DIR / logo_name
    try:
        from PIL import Image
        st.sidebar.image(Image.open(p), use_container_width=True)
    except Exception:
        pass

st.sidebar.divider()

# =========================================================
# Helpers
# =========================================================

def add_help_text(title: str, text: str):
    with st.expander(f"Help — {title}", expanded=False):
        st.markdown(text)

def add_help_text_sidebar(title: str, text: str):
    with st.sidebar.expander(f"Help — {title}", expanded=False):
        st.markdown(text)

def generate_example_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate a larger example dataset with three groups (A, B, C),
    including 2 outliers per group for boxplot and distribution demonstrations.
    """
    rng = np.random.default_rng(seed)

    n_per_group = 50

    # Main data
    group_a_main = rng.normal(loc=100.0, scale=5.0, size=n_per_group)
    group_b_main = rng.normal(loc=106.0, scale=6.0, size=n_per_group)
    group_c_main = rng.normal(loc=112.0, scale=5.5, size=n_per_group)

    # Two outliers per group
    group_a_outliers = np.array([82.0, 118.0])
    group_b_outliers = np.array([86.0, 126.0])
    group_c_outliers = np.array([92.0, 132.0])

    # Final groups
    group_a = np.concatenate([group_a_main, group_a_outliers])
    group_b = np.concatenate([group_b_main, group_b_outliers])
    group_c = np.concatenate([group_c_main, group_c_outliers])

    response = np.concatenate([group_a, group_b, group_c])

    # Second numeric variable for correlation/regression demos
    response2 = 0.75 * response + 20 + rng.normal(0, 4, len(response))

    df = pd.DataFrame({
        "Sample": (
            [f"A{i+1}" for i in range(len(group_a))] +
            [f"B{i+1}" for i in range(len(group_b))] +
            [f"C{i+1}" for i in range(len(group_c))]
        ),
        "Group": (
            ["A"] * len(group_a) +
            ["B"] * len(group_b) +
            ["C"] * len(group_c)
        ),
        "Response": response,
        "Response2": response2,
    })

    return df

def generate_ANOVA_example_dataset(seed: int = 123, group_specs=None) -> pd.DataFrame:
    """
    Generate an ANOVA – Análise de Variância example dataset with any number of groups.
    group_specs should be a list of tuples: (group_name, mean, sd, n)
    """
    rng = np.random.default_rng(seed)

    if group_specs is None:
        group_specs = [
            ("G1", 98, 5.0, 45),
            ("G2", 104, 5.5, 45),
            ("G3", 111, 6.0, 45),
        ]

    samples = []
    groups = []
    response = []

    for group_name, mean, sd, n in group_specs:
        values = rng.normal(mean, sd, n)
        response.extend(values)
        groups.extend([group_name] * n)
        samples.extend([f"{group_name}_{i+1}" for i in range(n)])

    return pd.DataFrame({
        "Sample": samples,
        "Group": groups,
        "Response": response,
    })

def numeric_columns(df: pd.DataFrame):
    return df.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def descriptive_stats(series: pd.Series) -> dict:
    """
    Return descriptive statistics for a numeric series.
    """
    x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()

    if len(x) == 0:
        return {}

    mean = np.mean(x)
    median = np.median(x)

    mode_res = stats.mode(x, keepdims=False)
    mode_val = float(mode_res.mode) if np.size(mode_res.mode) else np.nan

    std_sample = np.std(x, ddof=1) if len(x) > 1 else np.nan
    var_sample = np.var(x, ddof=1) if len(x) > 1 else np.nan
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    xmin = np.min(x)
    xmax = np.max(x)
    cv = (std_sample / mean * 100.0) if mean != 0 and np.isfinite(std_sample) else np.nan

    if len(x) > 1:
        sem = stats.sem(x)
        tcrit = stats.t.ppf(0.975, df=len(x) - 1)
        ci_low = mean - tcrit * sem
        ci_high = mean + tcrit * sem
    else:
        ci_low = np.nan
        ci_high = np.nan

    return {
        "n": len(x),
        "mean": mean,
        "median": median,
        "mode": mode_val,
        "std": std_sample,
        "variance": var_sample,
        "min": xmin,
        "q1": q1,
        "q3": q3,
        "max": xmax,
        "iqr": iqr,
        "cv_percent": cv,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }


def normal_curve_for_hist(data: np.ndarray, bins: int = 20, n_points: int = 400):
    """
    Create a normal curve scaled to histogram counts.
    """
    x = np.asarray(data, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return None, None

    mu = np.mean(x)
    sigma = np.std(x, ddof=1)

    if sigma <= 0 or not np.isfinite(sigma):
        return None, None

    xmin, xmax = np.min(x), np.max(x)
    x_line = np.linspace(xmin, xmax, n_points)

    pdf = stats.norm.pdf(x_line, loc=mu, scale=sigma)

    hist_counts, hist_edges = np.histogram(x, bins=bins)
    bin_width = hist_edges[1] - hist_edges[0]

    y_line = pdf * len(x) * bin_width
    return x_line, y_line


def add_descriptive_lines(fig: go.Figure, stats_dict: dict):
    """
    Add vertical lines for mean, median, and CI95 on a distribution plot.
    """
    if not stats_dict:
        return fig

    mean = stats_dict["mean"]
    median = stats_dict["median"]
    ci_low = stats_dict["ci95_low"]
    ci_high = stats_dict["ci95_high"]

    fig.add_vline(x=mean, line_dash="solid", annotation_text="Mean", annotation_position="top")
    fig.add_vline(x=median, line_dash="dot", annotation_text="Median", annotation_position="top")

    if np.isfinite(ci_low):
        fig.add_vline(x=ci_low, line_dash="dash", annotation_text="CI95 low", annotation_position="bottom")
    if np.isfinite(ci_high):
        fig.add_vline(x=ci_high, line_dash="dash", annotation_text="CI95 high", annotation_position="bottom")

    return fig


def f_test_variances(x1: np.ndarray, x2: np.ndarray) -> dict:
    """
    Classical F-test for comparing variances (two-sided).
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return {"F": np.nan, "df1": np.nan, "df2": np.nan, "p_value": np.nan}

    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)

    if s1 >= s2:
        F = s1 / s2 if s2 > 0 else np.nan
        df1 = n1 - 1
        df2 = n2 - 1
    else:
        F = s2 / s1 if s1 > 0 else np.nan
        df1 = n2 - 1
        df2 = n1 - 1

    if np.isfinite(F):
        cdf_val = stats.f.cdf(F, df1, df2)
        p_two_sided = 2 * min(cdf_val, 1 - cdf_val)
        p_two_sided = min(max(p_two_sided, 0.0), 1.0)
    else:
        p_two_sided = np.nan

    return {"F": F, "df1": df1, "df2": df2, "p_value": p_two_sided}


def cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Cohen's d for two independent groups.
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    n1, n2 = len(x1), len(x2)
    if n1 < 2 or n2 < 2:
        return np.nan

    s1 = np.var(x1, ddof=1)
    s2 = np.var(x2, ddof=1)

    pooled_sd = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return np.nan

    return (np.mean(x1) - np.mean(x2)) / pooled_sd


def ci_difference_means_welch(x1: np.ndarray, x2: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Confidence interval for difference in means using Welch approach.
    Returns: diff, low, high
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    m1, m2 = np.mean(x1), np.mean(x2)
    v1, v2 = np.var(x1, ddof=1), np.var(x2, ddof=1)
    n1, n2 = len(x1), len(x2)

    diff = m1 - m2
    se = np.sqrt(v1 / n1 + v2 / n2)

    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = ((v1 / n1) ** 2) / (n1 - 1) + ((v2 / n2) ** 2) / (n2 - 1)
    df = df_num / df_den if df_den != 0 else np.nan

    tcrit = stats.t.ppf(1 - alpha / 2, df) if np.isfinite(df) else np.nan

    low = diff - tcrit * se if np.isfinite(tcrit) else np.nan
    high = diff + tcrit * se if np.isfinite(tcrit) else np.nan

    return diff, low, high


def run_two_group_tests(x1: np.ndarray, x2: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """
    Run useful hypothesis tests for two independent groups.
    """
    results = []

    sh1 = stats.shapiro(x1) if len(x1) >= 3 else None
    sh2 = stats.shapiro(x2) if len(x2) >= 3 else None

    if sh1 is not None:
        results.append({
            "Test": "Shapiro-Wilk (Group 1)",
            "Statistic": sh1.statistic,
            "p-value": sh1.pvalue,
            "Interpretation": "Normal" if sh1.pvalue > alpha else "Non-normal"
        })

    if sh2 is not None:
        results.append({
            "Test": "Shapiro-Wilk (Group 2)",
            "Statistic": sh2.statistic,
            "p-value": sh2.pvalue,
            "Interpretation": "Normal" if sh2.pvalue > alpha else "Non-normal"
        })

    lev = stats.levene(x1, x2, center="median")
    results.append({
        "Test": "Levene (equal variances)",
        "Statistic": lev.statistic,
        "p-value": lev.pvalue,
        "Interpretation": "Equal variances plausible" if lev.pvalue > alpha else "Variances may differ"
    })

    ftest = f_test_variances(x1, x2)
    results.append({
        "Test": "F-test (variances)",
        "Statistic": ftest["F"],
        "p-value": ftest["p_value"],
        "Interpretation": "No strong variance difference" if ftest["p_value"] > alpha else "Variance difference"
    })

    t_equal = stats.ttest_ind(x1, x2, equal_var=True)
    results.append({
        "Test": "Student t-test (equal variances)",
        "Statistic": t_equal.statistic,
        "p-value": t_equal.pvalue,
        "Interpretation": "Means differ" if t_equal.pvalue <= alpha else "No evidence of mean difference"
    })

    t_welch = stats.ttest_ind(x1, x2, equal_var=False)
    results.append({
        "Test": "Welch t-test",
        "Statistic": t_welch.statistic,
        "p-value": t_welch.pvalue,
        "Interpretation": "Means differ" if t_welch.pvalue <= alpha else "No evidence of mean difference"
    })

    mw = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    results.append({
        "Test": "Mann-Whitney U",
        "Statistic": mw.statistic,
        "p-value": mw.pvalue,
        "Interpretation": "Groups differ" if mw.pvalue <= alpha else "No evidence of group difference"
    })

    return pd.DataFrame(results)


def linear_regression_summary(x: np.ndarray, y: np.ndarray) -> dict:
    """
    Compute basic linear regression statistics using scipy.
    """
    res = stats.linregress(x, y)
    return {
        "slope": res.slope,
        "intercept": res.intercept,
        "r_value": res.rvalue,
        "r_squared": res.rvalue ** 2,
        "p_value": res.pvalue,
        "std_err": res.stderr,
    }


def confidence_interval_mean(x: np.ndarray, confidence: float = 0.95) -> tuple:
    """
    Confidence interval for a mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    n = len(x)
    mean = np.mean(x)
    sd = np.std(x, ddof=1)
    sem = sd / np.sqrt(n)
    alpha = 1 - confidence
    tcrit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    low = mean - tcrit * sem
    high = mean + tcrit * sem
    return mean, low, high, sd, sem


# =========================================================
# Session state
# =========================================================

if "df" not in st.session_state:
    st.session_state["df"] = None

if "ANOVA_df" not in st.session_state:
    st.session_state["ANOVA_df"] = generate_ANOVA_example_dataset()


# =========================================================
# Title
# =========================================================

st.title("Estatística Básica para Química Analítica")
st.caption("Curso interativo em Streamlit para ensino de Estatística em Química Analítica e Quimiometria")

st.caption("""
**Por que falamos em Estatística básica em pesquisa?**
    """)

st.caption("""
* Estatística é a ciência que envolve a coleta, a organização e a interpretação de dados adquiridos.
    """)

st.caption("""
* Na parte que nos cabe, é a ciência de “Como realizar uma pesquisa científica”. 
  
    """)
    
st.caption("""  
* De certa forma, a estatística também pode ser definida como “A ciência da incerteza que é usada para ajudar a chegar a conclusões e tomar decisões nos mais diferentes aspectos”
    """)


tabs = st.tabs([
    "Explorar Dados",
    "Estatística Descritiva",
    "Distribuições",
    "Boxplots e Outliers",
    "Testes de Hipótese",
    "Simulador de Distribuição Normal",
    "Amostragem e Teorema Central do Limite",
    "Intervalo de Confiança",
    "ANOVA – Análise de Variância",
    "Correlação e Regressão",
    "Erros em Química Analítica",
])

# =========================================================
# Sidebar - Import
# =========================================================

st.sidebar.header("Importar Dados")

add_help_text_sidebar(
    "Import Data",
    """
Nesta seção você pode:

- enviar um arquivo CSV com seus próprios dados experimentais, ou
- carregar um conjunto de dados de exemplo.

O conjunto de exemplo contém grupos simulados e é útil para demonstrações em aula."""
)

c1, c2 = st.columns([1, 1])

with c1:
    if st.sidebar.button("Carregar conjunto de dados de exemplo", use_container_width=True,help="Load a built-in example dataset with two groups and 10 replicates each for quick classroom demonstrations."):
        st.session_state["df"] = generate_example_dataset()
        st.success("Example dataset loaded.")

with c2:
    uploaded = st.sidebar.file_uploader("Carregar arquivo CSV", type=["csv"],help="Upload a CSV table containing your analytical data. Ideally include numeric columns for measurements and categorical columns for groups.")

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.sidebar.session_state["df"] = df_up
        st.sidebar.success("CSV uploaded successfully.")

#df = st.sidebar.session_state["df"]

#if df is not None:
#    st.sidebar.subheader("Conjunto de dados atual")
#    st.sidebar.dataframe(df, use_container_width=True)

# =========================================================
# Tab 1 - Exploração dos Dados
# =========================================================

with tabs[0]:
    st.header("Exploração dos Dados")

    df = st.session_state["df"]
    if df is None:
        st.warning("Adicione primeiro os seus dados ou o dado de exemplo.")
    else:
        add_help_text(
            "Exploração dos Dados",
            """
Antes de iniciar qualquer análise estatística, é importante inspecionar os dados.

Verifique:

- dimensões da tabela
- tipos de variáveis
- primeiras linhas do conjunto de dados
- valores ausentes
- resumo estatístico inicial"""
        )

        c1, c2 = st.columns(2)
        with c1:
            st.write("Formato:", df.shape)
            st.write("Colunas Numéricas:", numeric_columns(df))
            st.write("Colunas Categóricas:", categorical_columns(df))
        with c2:
            st.write("Valores ausentes por coluna:")
            st.dataframe(df.isna().sum().rename("Ausentes"), use_container_width=True)

        st.subheader("Preview")
        st.dataframe(df.head(), use_container_width=True)

        num_cols = numeric_columns(df)
        if num_cols:
            st.subheader("Resumo numérico")
            st.dataframe(df[num_cols].describe().T, use_container_width=True)

# =========================================================
# Tab 2 - Estatística Descritiva
# =========================================================

with tabs[1]:
    st.header("Estatística Descritiva")

    df = st.session_state["df"]
    if df is None:
        st.warning("Adicione primeiro os seus dados ou o dado de exemplo.")
    else:
        add_help_text(
            "Estatística Descritiva",
            """
A estatística descritiva resume as principais características de um conjunto de dados.

Alguns dos parâmetros mais importantes são:

- média
- mediana
- moda
- variância
- desvio padrão
- coeficiente de variação
- quartis
- intervalo de confiança
"""
        )

        num_cols = numeric_columns(df)
        cat_cols = categorical_columns(df)

        col1, col2, col3 = st.columns(3)

        with col1:
            response_col = st.selectbox("Select numeric variable", num_cols,
		help="Choose the numeric variable that will be analyzed statistically.")

        with col2:
            group_col = st.selectbox(
                "Select group column (optional)",
                options=["<None>"] + cat_cols,
                index=1 if "Group" in cat_cols else 0,
                help = "Choose the column that identifies the experimental groups, such as control vs treated, or group A vs group B.")

        with col3:
            show_by_group = st.checkbox("Show statistics by group", value=(group_col != "<None>"),help="If selected, the descriptive statistics will be calculated separately for each group.")

        if group_col == "<None>":
            stats_all = descriptive_stats(df[response_col])
            st.subheader("Estatística Descritiva Geral")
            st.dataframe(pd.DataFrame([stats_all]).T.rename(columns={0: "Value"}), use_container_width=True)
        else:
            if show_by_group:
                rows = []
                for g, sub in df.groupby(group_col):
                    ds = descriptive_stats(sub[response_col])
                    ds[group_col] = g
                    rows.append(ds)
                out = pd.DataFrame(rows)
                cols = [group_col] + [c for c in out.columns if c != group_col]
                st.subheader("Descriptive statistics by group")
                st.dataframe(out[cols], use_container_width=True)
            else:
                stats_all = descriptive_stats(df[response_col])
                st.subheader("Overall descriptive statistics")
                st.dataframe(pd.DataFrame([stats_all]).T.rename(columns={0: "Value"}), use_container_width=True)

# =========================================================
# Tab 3 - Distribuição dos Dados
# =========================================================

with tabs[2]:
    st.header("Distribuição dos Dados")

    df = st.session_state["df"]
    if df is None:
        st.warning("Adicione primeiro os seus dados ou o dado de exemplo.")
    else:
        add_help_text(
            "Distribuição dos Dados",
            """
Gráficos de distribuição ajudam a entender:

- onde os dados estão concentrados
- qual é a dispersão dos valores
- se a distribuição se aproxima de uma distribuição normal
- se diferentes grupos apresentam diferenças na posição ou na variabilidade"""
        )

        num_cols = numeric_columns(df)
        cat_cols = categorical_columns(df)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            response_col = st.selectbox("Numeric variable", num_cols, key="dist_response")
        with c2:
            group_col = st.selectbox(
                "Group column",
                options=cat_cols,
                index=cat_cols.index("Group") if "Group" in cat_cols else 0,
                key="dist_group"
            )
        with c3:
            bins = st.slider("Histogram bins", 5, 40, 12,help="Controls how many bars are used in the histogram. More bins show more detail, fewer bins give a smoother overview.")
        with c4:
            plot_mode = st.selectbox(
                "Plot mode",
                [
                    "Both groups together (overlay)",
                    "Facet by group",
                    "Overall only",
                ],help="Choose how the Distribuição dos Dados will be displayed: overlaid groups, separate panels by group, or all observations together."
            )

        show_normal = st.checkbox("Show theoretical normal curve", value=True,help="Displays a normal curve estimated from the data mean and standard deviation.")
        show_lines = st.checkbox("Show mean / median / CI95 lines", value=True,help="Adds vertical reference lines for mean, median, and 95% confidence interval when available.")
        show_rug = st.checkbox("Show rug / points", value=False,help="Shows individual observations along the axis to help visualize data density and spread.")

        data = df[[response_col, group_col]].dropna().copy()

        if plot_mode == "Overall only":
            x = data[response_col].to_numpy()

            fig = px.histogram(
                data,
                x=response_col,
                nbins=bins,
                marginal="rug" if show_rug else None,
                opacity=0.75,
                title=f"Distribution of {response_col} (overall)"
            )

            ds = descriptive_stats(data[response_col])

            if show_normal:
                x_line, y_line = normal_curve_for_hist(x, bins=bins)
                if x_line is not None:
                    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Normal curve"))

            if show_lines:
                fig = add_descriptive_lines(fig, ds)

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Descriptive parameters shown on the distribution")
            st.dataframe(pd.DataFrame([ds]).T.rename(columns={0: "Value"}), use_container_width=True)

        elif plot_mode == "Both groups together (overlay)":
            fig = px.histogram(
                data,
                x=response_col,
                color=group_col,
                nbins=bins,
                barmode="overlay",
                opacity=0.55,
                marginal="rug" if show_rug else None,
                title=f"Distribution of {response_col} by {group_col}"
            )

            if show_normal:
                for g in data[group_col].unique():
                    sub = data.loc[data[group_col] == g, response_col].to_numpy()
                    x_line, y_line = normal_curve_for_hist(sub, bins=bins)
                    if x_line is not None:
                        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"Normal curve ({g})"))

            if show_lines:
                for g in data[group_col].unique():
                    ds = descriptive_stats(data.loc[data[group_col] == g, response_col])
                    mean = ds["mean"]
                    median = ds["median"]
                    fig.add_vline(x=mean, line_dash="solid", annotation_text=f"Mean {g}")
                    fig.add_vline(x=median, line_dash="dot", annotation_text=f"Median {g}")

            st.plotly_chart(fig, use_container_width=True)

            rows = []
            for g, sub in data.groupby(group_col):
                ds = descriptive_stats(sub[response_col])
                ds[group_col] = g
                rows.append(ds)
            out = pd.DataFrame(rows)
            cols = [group_col] + [c for c in out.columns if c != group_col]
            st.subheader("Descriptive parameters by group")
            st.dataframe(out[cols], use_container_width=True)

        else:
            fig = px.histogram(
                data,
                x=response_col,
                nbins=bins,
                facet_col=group_col,
                marginal="rug" if show_rug else None,
                title=f"Distribution of {response_col} faceted by {group_col}"
            )

            st.plotly_chart(fig, use_container_width=True)

            rows = []
            for g, sub in data.groupby(group_col):
                ds = descriptive_stats(sub[response_col])
                ds[group_col] = g
                rows.append(ds)
            out = pd.DataFrame(rows)
            cols = [group_col] + [c for c in out.columns if c != group_col]
            st.subheader("Descriptive parameters by group")
            st.dataframe(out[cols], use_container_width=True)

# =========================================================
# Tab 4 - Boxplots
# =========================================================

with tabs[3]:
    st.header("Boxplots e Outliers")

    df = st.session_state["df"]
    if df is None:
        st.warning("Adicione primeiro os seus dados ou o dado de exemplo.")
    else:
        add_help_text(
            "Boxplots e Outliers",
            """
Boxplots são úteis porque mostram rapidamente:

- mediana: o valor central do conjunto de dados quando os valores são ordenados do menor para o maior. Metade dos dados está acima da mediana e metade está abaixo.
- quartis: são valores que dividem os dados ordenados em quatro partes iguais
- intervalo interquartil: a diferença entre o terceiro e o primeiro quartil
- possíveis valores extremos (outliers)

Eles são uma das formas mais rápidas de comparar grupos experimentalmente."""
        )

        num_cols = numeric_columns(df)
        cat_cols = categorical_columns(df)

        c1, c2, c3 = st.columns(3)
        with c1:
            response_col = st.selectbox("Numeric variable", num_cols, key="box_response",help="Choose the numeric variable to compare across groups using a boxplot.")
        with c2:
            group_col = st.selectbox(
                "Group column",
                cat_cols,
                index=cat_cols.index("Group") if "Group" in cat_cols else 0,
                key="box_group"
            )
        with c3:
            points_mode = st.selectbox("Show points", ["all", "outliers", False], index=0,help="Choose whether to display all observations, only suspected outliers, or no points on the boxplot.")

        fig_box = px.box(
            df,
            x=group_col,
            y=response_col,
            points=points_mode,
            color=group_col,
            title=f"Boxplot of {response_col} by {group_col}"
        )

        st.plotly_chart(fig_box, use_container_width=True)

        show_violin = st.checkbox("Also show violin plot", value=True)
        if show_violin:
            fig_violin = px.violin(
                df,
                x=group_col,
                y=response_col,
                box=True,
                points="all",
                color=group_col,
                title=f"Violin plot of {response_col} by {group_col}"
            )
            st.plotly_chart(fig_violin, use_container_width=True)

# =========================================================
# Tab 5 - Testes de Hipótese
# =========================================================

with tabs[4]:
    st.header("Testes de Hipótese")

    df = st.session_state["df"]
    if df is None:
        st.warning("Adicione primeiro os seus dados ou o dado de exemplo.")
    else:
        add_help_text(
            "Testes de Hipótese",
            """
Nesta seção avaliamos se dois grupos apresentam diferenças estatisticamente significativas.

São apresentados vários testes estatísticos importantes:

- teste de normalidade (Shapiro-Wilk)
- teste de igualdade de variâncias (Levene)
- teste F para variâncias
- teste t de Student
- teste t de Welch
- teste não paramétrico Mann-Whitney"""
        )

        num_cols = numeric_columns(df)
        cat_cols = categorical_columns(df)

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            response_col = st.selectbox("Variável numérica", num_cols, key="hyp_response",help="Escolha a variável numérica que será comparada entre os dois grupos selecionados.")
        with c2:
            group_col = st.selectbox(
                "Group column",
                cat_cols,
                index=cat_cols.index("Group") if "Group" in cat_cols else 0,
                key="hyp_group",help="Escolha a coluna categórica que define os grupos para os Testes de Hipótese.")
        with c3:
            alpha = st.selectbox("Nível de significância (alpha)", [0.10, 0.05, 0.01], index=1,help="Alpha é o limiar usado para avaliar a significância estatística. O valor mais comum é 0,05.")
        with c4:
            groups = list(df[group_col].dropna().unique())
            if len(groups) >= 2:
                g1 = st.selectbox("Grupo 1", groups, index=0,help="Escolha o primeiro grupo para comparação.")
                g2 = st.selectbox("Grupo 2", groups, index=1,help="Escolha o segundo grupo para comparação.")

        groups = list(df[group_col].dropna().unique())
        if len(groups) < 2:
            st.error("You need at least two groups in the selected group column.")
        else:
            x1 = pd.to_numeric(df.loc[df[group_col] == g1, response_col], errors="coerce").dropna().to_numpy()
            x2 = pd.to_numeric(df.loc[df[group_col] == g2, response_col], errors="coerce").dropna().to_numpy()

            if len(x1) < 2 or len(x2) < 2:
                st.error("Each selected group must contain at least two numeric observations.")
            else:
                st.subheader("Selected data summary")

                ds1 = descriptive_stats(pd.Series(x1))
                ds2 = descriptive_stats(pd.Series(x2))
                sum_df = pd.DataFrame([
                    {"Group": g1, **ds1},
                    {"Group": g2, **ds2},
                ])
                st.dataframe(sum_df, use_container_width=True)

                st.subheader("Test results")
                test_df = run_two_group_tests(x1, x2, alpha=alpha)
                st.dataframe(test_df, use_container_width=True)

                diff, ci_low, ci_high = ci_difference_means_welch(x1, x2, alpha=alpha)
                d = cohens_d(x1, x2)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Difference in means", f"{diff:.4f}")
                with c2:
                    st.metric("CI difference (lower)", f"{ci_low:.4f}")
                with c3:
                    st.metric("CI difference (upper)", f"{ci_high:.4f}")

                st.metric("Cohen's d", f"{d:.4f}")

                st.subheader("Interpretation guide")
                st.markdown(
                    f"""
- **Shapiro-Wilk p > {alpha}**: data are compatible with normality.
- **Levene p > {alpha}**: equal variances are plausible.
- **Student t-test**: use when normality is acceptable and variances are similar.
- **Welch t-test**: safer when variances differ.
- **Mann-Whitney**: useful as a nonparametric alternative.
- **If p ≤ {alpha}**: reject H₀ at the selected significance level.
- **Cohen's d** helps quantify effect size, not just significance.
"""
                )

                st.subheader("Visual comparison of the two groups")
                plot_df = df.loc[df[group_col].isin([g1, g2]), [group_col, response_col]].copy()

                fig_compare = px.box(
                    plot_df,
                    x=group_col,
                    y=response_col,
                    color=group_col,
                    points="all",
                    title=f"{response_col}: {g1} vs {g2}"
                )
                st.plotly_chart(fig_compare, use_container_width=True)

# =========================================================
# Tab 6 - Simulador de Distribuição Normal
# =========================================================

with tabs[5]:
    st.header("Simulador de Distribuição Normal")

    add_help_text(
        "Simulador de Distribuição Normal",
        """
Este simulador permite visualizar como funciona a distribuição normal.

Você pode alterar:

- média
- desvio padrão
- tamanho da amostra

Isso ajuda a entender como essas variáveis influenciam a forma da distribuição.
"""
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        mu = st.slider(
            "Mean (μ)",
            -20.0, 20.0, 0.0, 0.5,
            help="Controls the center of the simulated normal distribution."
        )
    with c2:
        sigma = st.slider(
            "Standard deviation (σ)",
            0.5, 10.0, 2.0, 0.5,
            help="Controls the spread of the simulated normal distribution. Smaller values make the curve narrower and taller."
        )
    with c3:
        n = st.slider(
            "Simulated sample size",
            3, 5000, 50, 1,
            help="Controls how many simulated observations are generated. Very small sample sizes, such as 3, usually do not resemble a smooth normal distribution."
        )

    show_empirical_curve = st.checkbox(
        "Show empirical distribution curve",
        value=True,
        help="Shows a smooth curve estimated from the sampled data, allowing comparison between the theoretical normal distribution and the real sampled distribution."
    )

    rng = np.random.default_rng(12345)
    x = rng.normal(loc=mu, scale=sigma, size=n)

    # Fixed x-axis to make width/height comparisons easier
    x_min = -20
    x_max = 20
    nbins = 30

    fig = px.histogram(
        x=x,
        nbins=nbins,
        title="Simulated normal distribution",
        opacity=0.75,
    )

    # Fixed histogram range for consistency
    fig.update_traces(xbins=dict(start=x_min, end=x_max, size=(x_max - x_min) / nbins))

    x_line = np.linspace(x_min, x_max, 400)

    # Theoretical curve
    y_pdf = stats.norm.pdf(x_line, loc=mu, scale=sigma)
    hist_counts, hist_edges = np.histogram(x, bins=nbins, range=(x_min, x_max))
    bin_width = hist_edges[1] - hist_edges[0]
    y_scaled = y_pdf * len(x) * bin_width

    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_scaled,
            mode="lines",
            name="Theoretical normal curve"
        )
    )

    # Empirical curve from sampled data
    if show_empirical_curve and len(x) > 1 and np.std(x, ddof=1) > 0:
        kde = stats.gaussian_kde(x)
        y_kde = kde(x_line)
        y_kde_scaled = y_kde * len(x) * bin_width

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_kde_scaled,
                mode="lines",
                name="Empirical distribution curve",
                line=dict(color="red", width=3)
            )
        )

    # Reference lines from sampled data
    sample_mean = np.mean(x)
    sample_sd = np.std(x, ddof=1) if len(x) > 1 else np.nan

    fig.add_vline(x=sample_mean, line_dash="solid", annotation_text="Sample mean")
    if np.isfinite(sample_sd) and sample_sd > 0:
        fig.add_vline(x=sample_mean - sample_sd, line_dash="dot", annotation_text="-1 SD")
        fig.add_vline(x=sample_mean + sample_sd, line_dash="dot", annotation_text="+1 SD")

    fig.update_xaxes(range=[x_min, x_max], title="Value")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig, use_container_width=True)

    if n <= 5:
        st.warning(
            "With very few observations, the histogram and the empirical curve look irregular and may poorly represent a normal distribution."
        )
    elif n <= 15:
        st.info(
            "With a small sample size, the sampled distribution begins to resemble a normal shape, but it still looks rough and unstable."
        )
    else:
        st.success(
            "With more observations, the sampled distribution more closely resembles the theoretical normal curve."
        )

    st.markdown(
    """
**Como interpretar este gráfico**

- As **barras** mostram as observações amostradas.
- A **curva normal teórica** representa a distribuição ideal definida pela média e pelo desvio padrão selecionados.
- A **curva da distribuição empírica** mostra a distribuição suavizada estimada a partir dos pontos amostrados.
- Quando o tamanho da amostra é muito pequeno, a forma da distribuição empírica pode diferir bastante da distribuição teórica.
"""
    )

    ds = descriptive_stats(pd.Series(x))
    st.dataframe(pd.DataFrame([ds]).T.rename(columns={0: "Value"}), use_container_width=True)


#  =========================================================
# Tab 7 - Amostragem e Teorema Central do Limite
# =========================================================
with tabs[6]:
    st.header("Amostragem e Teorema Central do Limite")

    add_help_text(
        "Amostragem e Teorema Central do Limite",
        """
O Teorema Central do Limite mostra que, à medida que o tamanho da amostra aumenta, a distribuição das médias amostrais tende a se aproximar de uma distribuição normal.

Essa ideia é uma das mais importantes de toda a estatística, pois explica por que métodos baseados na distribuição normal funcionam em muitos tipos diferentes de dados."""
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pop_mean = st.slider("Média da População", 50.0, 150.0, 100.0, 1.0,help="Sets the mean of the population from which repeated samples will be drawn.")
    with c2:
        pop_sd = st.slider("Desvio Padrão da População", 1.0, 30.0, 10.0, 1.0,help="Sets the population standard deviation. Larger values create a more dispersed population.")
    with c3:
        sample_size = st.slider("Tamanho da Amostra (n)", 2, 100, 5, 1,help="Number of observations in each simulated sample. Larger sample sizes usually produce more stable sample means.")
    with c4:
        n_samples = st.slider("Número de amostras repetidas", 10, 5000, 1000, 10,help="Defines how many repeated samples will be generated to build the sampling distribution of the mean.")

    rng = np.random.default_rng(2026)
    population = rng.normal(pop_mean, pop_sd, 100000)
    sample_means = [rng.choice(population, size=sample_size, replace=True).mean() for _ in range(n_samples)]
    sample_means = np.array(sample_means)

    st.subheader("Population distribution")
    fig_pop = px.histogram(population[:3000], nbins=40, title="Population (subset shown)")
    st.plotly_chart(fig_pop, use_container_width=True)

    st.subheader("Distribution of sample means")
    fig_means = px.histogram(sample_means, nbins=40, title="Sampling distribution of the mean")
    x_line, y_line = normal_curve_for_hist(sample_means, bins=40)
    if x_line is not None:
        fig_means.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name="Normal curve"))
    st.plotly_chart(fig_means, use_container_width=True)

    st.write("Média da População:", round(np.mean(population), 4))
    st.write("Média das médias amostrais:", round(np.mean(sample_means), 4))
    st.write("Desvio Padrão da População:", round(np.std(population, ddof=1), 4))
    st.write("Desvio padrão das médias amostrais (erro padrão):", round(np.std(sample_means, ddof=1), 4))

# =========================================================
# Tab 8 - Confidence Interval Visualizer
# =========================================================

with tabs[7]:
    st.header("Confidence Interval Visualizer")

    add_help_text(
        "Confidence Interval Visualizer",
        """
Um intervalo de confiança fornece uma faixa de valores plausíveis para a média da população com base nos dados observados em uma amostra.

Esta aba permite visualizar como diferentes fatores influenciam o intervalo de confiança:

- Maior variabilidade nos dados → o intervalo de confiança fica mais largo

- Maior tamanho de amostra → o intervalo de confiança fica mais estreito

- Maior nível de confiança → o intervalo de confiança fica mais largo

Esses efeitos ajudam a entender como a precisão das estimativas depende da qualidade e da quantidade dos dados coletados.
"""
    )

    df = st.session_state["df"]

    source = st.radio("Data source", ["Usar conjunto de dados atual", "Simular novos dados"], horizontal=True,help="Escolha se o intervalo de confiança será calculado a partir do conjunto de dados atual ou de valores recém-simulados."
)

    if source == "Use current dataset" and df is not None:
        num_cols = numeric_columns(df)
        response_col = st.selectbox("Numeric variable", num_cols, key="ci_response",help="Choose the numeric variable used to calculate the confidence interval.")
        x = pd.to_numeric(df[response_col], errors="coerce").dropna().to_numpy()
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            sim_mean = st.slider("Simulated mean", 0.0, 200.0, 100.0, 1.0,help="Sets the mean of the simulated dataset.")
        with c2:
            sim_sd = st.slider("Simulated SD", 1.0, 30.0, 8.0, 1.0,help="Sets the standard deviation of the simulated dataset."
)
        with c3:
            sim_n = st.slider("Simulated n", 3, 200, 20, 1,help="Defines how many observations will be simulated.")
        rng = np.random.default_rng(321)
        x = rng.normal(sim_mean, sim_sd, sim_n)

    conf = st.selectbox("Confidence level", [0.90, 0.95, 0.99], index=1,help="Defines how wide the confidence interval will be. Higher confidence gives a wider interval.")

    if len(x) >= 2:
        mean, low, high, sd, sem = confidence_interval_mean(x, confidence=conf)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Mean", f"{mean:.2f}")
        with c2:
            st.metric("SD", f"{sd:.2f}")
        with c3:
            st.metric("SEM", f"{sem:.2f}")
        with c4:
            st.metric(f"{int(conf*100)}% CI width", f"{(high-low):.2f}")

        fig_ci = go.Figure()
        fig_ci.add_trace(go.Scatter(
            x=[low, mean, high],
            y=[1, 1, 1],
            mode="lines+markers+text",
            text=[f"{low:.3f}", f"mean={mean:.3f}", f"{high:.3f}"],
            textposition="top center",
            name="Confidence Interval"
        ))
        fig_ci.update_yaxes(visible=False)
        fig_ci.update_layout(title=f"{int(conf*100)}% Intervalo de Confiança para a Média")
        st.plotly_chart(fig_ci, use_container_width=True)

        fig_hist = px.histogram(x=x, nbins=20, title="Distribuição dos dados utilizada para o cálculo do intervalo de confiança")
        fig_hist.add_vline(x=mean, line_dash="solid", annotation_text=f"Mean: {mean:.2f}")
        fig_hist.add_vline(x=low, line_dash="dash", annotation_text="CI low")
        fig_hist.add_vline(x=high, line_dash="dash", annotation_text="CI high")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.warning("Need at least two observations.")

# =========================================================
# Tab 9 - ANOVA – Análise de Variância
# =========================================================

with tabs[8]:
    st.header("ANOVA – Análise de Variância")

    add_help_text(
        "ANOVA – Análise de Variância",
        """
A ANOVA (Análise de Variância) é um método estatístico utilizado para comparar as médias de três ou mais grupos.

Ela avalia se as diferenças observadas entre os grupos podem ser explicadas apenas pela variabilidade natural dos dados ou se indicam diferenças reais entre as médias.

Hipótese nula (H₀):
- Todas as médias dos grupos são iguais.

Hipótese alternativa (H₁):
- Pelo menos uma das médias dos grupos é diferente das outras.

Nesta seção, o aplicativo detecta automaticamente quantos grupos existem na coluna selecionada para agrupamento e realiza a análise considerando todos esses grupos."""
    )

    if st.button(
        "Load example ANOVA – Análise de Variância dataset",
        use_container_width=False,
        help="Load a built-in example dataset for ANOVA – Análise de Variância demonstrations. The app will automatically detect how many groups are present."
    ):
        st.session_state["ANOVA_df"] = generate_ANOVA_example_dataset()
        st.success("ANOVA – Análise de Variância example dataset loaded.")

    ANOVA_df = st.session_state["ANOVA_df"]

    if ANOVA_df is None or ANOVA_df.empty:
        st.warning("Carregue primeiro um conjunto de dados de exemplo para ANOVA – Análise de Variância.")
    else:
        st.dataframe(ANOVA_df, use_container_width=True)

        num_cols = numeric_columns(ANOVA_df)
        cat_cols = categorical_columns(ANOVA_df)

        c1, c2 = st.columns(2)
        with c1:
            response_col = st.selectbox(
                "Variável numérica",
                num_cols,
                key="ANOVA – Análise de Variância_resp",
                help="Escolha a variável numérica cujas médias serão comparadas entre todos os grupos detectados."
            )
        with c2:
            group_col = st.selectbox(
                "Group column",
                cat_cols,
                key="ANOVA – Análise de Variância_group",
                help="Escolha a variável categórica que define os grupos utilizados na ANOVA – Análise de Variância."
            )

        alpha = st.selectbox(
            "Alpha",
            [0.10, 0.05, 0.01],
            index=1,
            key="ANOVA – Análise de Variância_alpha",
            help="Limiar de significância usado para decidir se o resultado da ANOVA – Análise de Variância é estatisticamente significativo."
        )

        # Keep only needed columns and remove missing values
        work_df = ANOVA_df[[group_col, response_col]].copy()
        work_df[response_col] = pd.to_numeric(work_df[response_col], errors="coerce")
        work_df = work_df.dropna(subset=[group_col, response_col])

        # Detect all groups automatically
        detected_groups = list(work_df[group_col].unique())

        # Keep only groups with at least 2 observations
        valid_groups = []
        valid_samples = []
        excluded_groups = []

        for g in detected_groups:
            s = work_df.loc[work_df[group_col] == g, response_col].to_numpy()
            if len(s) >= 2:
                valid_groups.append(g)
                valid_samples.append(s)
            else:
                excluded_groups.append(g)

        st.write(f"Grupos detectados: **{len(detected_groups)}**")
        st.write(f"Grupos válidos utilizados na ANOVA – Análise de Variância: **{len(valid_groups)}**")

        if excluded_groups:
            st.info(
                "Os seguintes grupos foram excluídos porque tinham menos de 2 observações: "
                + ", ".join(map(str, excluded_groups))
            )

        if len(valid_samples) >= 2:
            f_stat, p_val = stats.f_oneway(*valid_samples)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown("**F statistic**  \n*Compara a variabilidade entre os grupos com a variabilidade dentro dos grupos.*")
                st.metric(label="", value=f"{f_stat:.4f}")
                st.caption(
                        "**F = variância entre os grupos / variância dentro dos grupos.**")
                st.caption(
                        "Valores altos de F indicam evidências mais fortes de que pelo menos uma média de grupo é diferente.")
            with m2:
                st.metric("p-value", f"{p_val:.3e} | {p_val:.20f} ")

                if p_val <= alpha:
                    st.success("Rejeitar H₀: pelo menos uma média de grupo é diferente.")
                else:
                   st.warning("Não rejeitar H₀: não há evidência clara de que as médias sejam diferentes.")

            plot_df = work_df[work_df[group_col].isin(valid_groups)].copy()

            fig_box = px.box(
                plot_df,
                x=group_col,
                y=response_col,
                color=group_col,
                points="all",
                title=f"ANOVA – Análise de Variância visual comparison: {response_col} by {group_col}"
            )
            st.plotly_chart(fig_box, use_container_width=True)

            rows = []
            for g, s in zip(valid_groups, valid_samples):
                ds = descriptive_stats(pd.Series(s))
                ds[group_col] = g
                rows.append(ds)

            out = pd.DataFrame(rows)
            cols = [group_col] + [c for c in out.columns if c != group_col]

            st.subheader("Descriptive statistics by group")
            st.dataframe(out[cols], use_container_width=True)

        else:
            st.warning("A ANOVA – Análise de Variância requer pelo menos dois grupos, cada um com pelo menos duas observações numéricas.")


# =========================================================
# Tab 9 - Correlação e Regressão
# =========================================================

with tabs[9]:
    st.header("Correlação e Regressão")

    add_help_text(
        "Correlação e Regressão",
        """
Esta seção é especialmente útil em química analítica, por exemplo na construção e avaliação de curvas de calibração.

Aqui você pode:

- comparar duas variáveis numéricas

- calcular a correlação de Pearson

- ajustar uma regressão linear

- avaliar os parâmetros do modelo, como:

  * inclinação da reta (slope)

  * intercepto

  * coeficiente de determinação (R²)

  * valor de p (p-value)

Essas ferramentas ajudam a entender a relação entre duas variáveis e a avaliar a qualidade de um modelo de calibração ou previsão.
"""
    )

    df = st.session_state["df"]

    if df is None:
        st.warning("Adicione primeiro os seus dados ou o dado de exemplo.")
    else:
        work_df = df.copy()

        if "Response2" not in work_df.columns and "Response" in work_df.columns:
            rng = np.random.default_rng(777)
            work_df["Response2"] = 0.8 * work_df["Response"] + 15 + rng.normal(0, 3, len(work_df))

        num_cols = numeric_columns(work_df)

        c1, c2, c3 = st.columns(3)
        with c1:
            x_col = st.selectbox("Variável X", num_cols, key="reg_x",help="Escolha a variável que será colocada no eixo x, geralmente a variável independente.")
        with c2:
            y_col = st.selectbox("Variável Y", num_cols, index=min(1, len(num_cols) - 1), key="reg_y",help="Escolha a variável que será colocada no eixo y, geralmente a variável dependente ou variável resposta.")
        with c3:
            color_col = st.selectbox("Colorir por (opcional)", ["<None>"] + categorical_columns(work_df), key="reg_color",help="Opcionalmente, colorir os pontos por uma variável categórica para revelar padrões entre grupos.")

        plot_df = work_df[[x_col, y_col] + ([] if color_col == "<None>" else [color_col])].dropna().copy()

        x = plot_df[x_col].to_numpy(dtype=float)
        y = plot_df[y_col].to_numpy(dtype=float)

        corr_pearson = stats.pearsonr(x, y)
        corr_spearman = stats.spearmanr(x, y)
        reg = linear_regression_summary(x, y)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Pearson r", f"{corr_pearson.statistic:.4f}")
        with c2:
            st.metric("Pearson p-value", f"{corr_pearson.pvalue:.6f}")
        with c3:
            st.metric("Spearman rho", f"{corr_spearman.statistic:.4f}")
        with c4:
            st.metric("R²", f"{reg['r_squared']:.4f}")

        title = f"{y_col} vs {x_col}"
        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            color=None if color_col == "<None>" else color_col,
            title=title,
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)

        reg_df = pd.DataFrame([reg]).T.rename(columns={0: "Value"})
        reg_df.loc["p_value", "Value"] = f"{reg_df.loc['p_value', 'Value']:.3e}"
        st.subheader("Regression summary")
        st.dataframe(reg_df, use_container_width=True)

# =========================================================
# Tab 10 - Analytical Errors
# =========================================================

with tabs[10]:
    st.header("Erros em Química Analítica")

    add_help_text(
        "Erros em Química Analítica",
        """
Esta seção ilustra alguns conceitos fundamentais relacionados a erros de medição em experimentos analíticos:

- erro aleatório (random error)

- erro sistemático (systematic error)

- precisão (precision)

- exatidão ou acurácia (accuracy)

Erro aleatório provoca variações imprevisíveis entre medições repetidas e aumenta a dispersão dos dados.

Erro sistemático produz um desvio consistente nas medições, fazendo com que a média dos resultados se afaste do valor verdadeiro.
"""
    )

    add_help_text(
    "Erros em Química Analítica",
    """
Esta seção ilustra os principais tipos de erro de medição encontrados em química analítica.

Em qualquer experimento analítico, o valor medido raramente coincide exatamente com o valor verdadeiro.
Essa diferença surge principalmente devido a dois tipos de erro.

1. Erro aleatório (Random error)

- Erros aleatórios são variações imprevisíveis que ocorrem entre medições repetidas.

- Eles podem surgir de pequenas flutuações no sistema experimental, como:

- ruído instrumental

- variações ambientais (temperatura, vibração, umidade)

- pequenas diferenças no preparo ou manuseio da amostra

O erro aleatório afeta principalmente a precisão, ou seja, o grau de concordância entre medições repetidas.

Quando o erro aleatório aumenta:

- a dispersão das medições aumenta

- o desvio padrão aumenta

- o histograma das medições fica mais largo

2. Erro sistemático (Systematic error)

Erros sistemáticos são desvios consistentes que deslocam todas as medições na mesma direção.

Eles podem ser causados por:

- problemas de calibração

- padrões incorretos

- deriva instrumental

- erros no procedimento experimental

O erro sistemático afeta principalmente a exatidão (ou acurácia), que indica o quão próximo o resultado medido está do valor verdadeiro.

Quando o erro sistemático aumenta:

- toda a distribuição dos dados se desloca em relação ao valor verdadeiro

- a média das medições fica enviesada

- o erro permanece mesmo com muitas replicatas

- Precisão vs. Exatidão

Precisão e exatidão descrevem propriedades diferentes das medições:

- Precisão indica o quão próximas as medições repetidas estão entre si.

Exatidão (acurácia) indica o quão próximo o resultado médio está do valor verdadeiro.

Assim, é possível ter diferentes situações experimentais:

- alta precisão e baixa exatidão

- alta exatidão e baixa precisão

- alta precisão e alta exatidão

- baixa precisão e baixa exatidão"""
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        true_value = st.slider("Valor Real", 50.0, 150.0, 100.0, 1.0,help="Representa o valor de referência ou o valor verdadeiro aceito para o analito.")
    with c2:
        random_sd = st.slider("Erro aleatório (desvio padrão – DP)", 0.1, 20.0, 3.0, 0.1,help="Representa a variação aleatória entre medições repetidas, afetando principalmente a precisão.")
    with c3:
        systematic_bias = st.slider("Systematic error (bias)", -20.0, 20.0, 0.0, 0.5,help="Representa um desvio consistente em relação ao valor verdadeiro, afetando a exatidão.")
    with c4:
        n_rep = st.slider("Number of replicate measurements", 5, 200, 30, 1,help="Define quantas medições analíticas repetidas serão simuladas.")

    rng = np.random.default_rng(2024)
    measured = true_value + systematic_bias + rng.normal(0, random_sd, n_rep)

    mean_measured = np.mean(measured)
    sd_measured = np.std(measured, ddof=1)
    abs_error = mean_measured - true_value
    rel_error = (abs_error / true_value) * 100 if true_value != 0 else np.nan
    cv = (sd_measured / mean_measured) * 100 if mean_measured != 0 else np.nan

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Média medida", f"{mean_measured:.4f}")
    with c2:
        st.metric("Desvio padrão (DP)", f"{sd_measured:.4f}")
    with c3:
        st.metric("Erro absoluto", f"{abs_error:.4f}")
    with c4:
        st.metric("Erro relativo (%)", f"{rel_error:.4f}")

    st.metric("Coeficiente de variação (%)", f"{cv:.4f}")

    fig = px.histogram(
        x=measured,
        nbins=20,
        title="Distribuição de medições analíticas replicadas"
    )
    fig.add_vline(x=true_value, line_dash="solid", annotation_text="True value")
    fig.add_vline(x=mean_measured, line_dash="dash", annotation_text="Measured mean")
    st.plotly_chart(fig, use_container_width=True)

    interpretation = []
    if abs(systematic_bias) < 1e-9 and random_sd <= 2:
        interpretation.append("Alta exatidão e alta precisão.")
    elif abs(systematic_bias) > 2 and random_sd <= 2:
        interpretation.append("Baixa exatidão, mas alta precisão.")
    elif abs(systematic_bias) <= 2 and random_sd > 5:
        interpretation.append("Alta exatidão, mas baixa precisão.")

    elif abs(systematic_bias) > 2 and random_sd > 5:
        interpretation.append("Baixa exatidão e baixa precisão.")
    else:
        interpretation.append("As medições estão próximas do valor verdadeiro e também são consistentes entre si (alta exatidão e alta precisão).")

    st.markdown("### Interpretação")
    for item in interpretation:
        st.write("-", item)

    st.markdown(
        """
**Interpretação didática:**

- Exatidão (Accuracy) está relacionada a quão próxima a média das medições está do valor verdadeiro.

- Precisão (Precision) está relacionada à dispersão das medições repetidas, ou seja, quão próximas elas estão entre si.

- Erro sistemático afeta principalmente a exatidão, pois desloca os resultados em relação ao valor verdadeiro.

- Erro aleatório afeta principalmente a precisão, pois aumenta a variabilidade entre as medições.
"""
    )

