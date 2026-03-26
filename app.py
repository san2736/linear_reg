import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
import io

st.set_page_config(page_title="Advertising Regression Analysis", layout="wide")

st.title("📊 Advertising Dataset — Regression Analysis")
st.markdown("Explore linear and interaction-effect models predicting **sales** from TV, Radio, and Newspaper ad spend.")

# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Advertising.csv", usecols=[1, 2, 3, 4])
    df["TV2"] = df["TV"] ** 2
    df["radio2"] = df["radio"] ** 2
    df["newspaper2"] = df["newspaper"] ** 2
    return df

df = load_data()

# ─── Sidebar Navigation ──────────────────────────────────────────────────────
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Select Section",
    [
        "📂 Data Overview",
        "📈 Pairplot & Correlations",
        "🔍 Linearity Check",
        "📐 Model Builder",
        "🔮 Predict Sales",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if section == "📂 Data Overview":
    st.header("Data Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Sample Data")
    st.dataframe(df[["TV", "radio", "newspaper", "sales"]].head(20), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(df[["TV", "radio", "newspaper", "sales"]].describe().round(2), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. PAIRPLOT & CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════
elif section == "📈 Pairplot & Correlations":
    st.header("Pairplot & Correlation Matrix")

    st.subheader("Pairplot")
    with st.spinner("Rendering pairplot…"):
        fig = sns.pairplot(df[["TV", "radio", "newspaper", "sales"]], plot_kws={"alpha": 0.5})
        fig.fig.suptitle("Pairplot of Advertising Variables", y=1.02)
        st.pyplot(fig.fig)
    plt.close("all")

    st.subheader("Correlation Heatmap")
    fig2, ax = plt.subplots(figsize=(7, 5))
    corr = df[["TV", "radio", "newspaper", "sales"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig2)
    plt.close("all")

    st.subheader("Correlation Table")
    st.dataframe(df.corr().round(3), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. LINEARITY CHECK (regplot)
# ═══════════════════════════════════════════════════════════════════════════
elif section == "🔍 Linearity Check":
    st.header("Linearity Check — Regression Plots")
    st.markdown(
        "Use `order=1` for linear and `order=2` for quadratic fit. "
        "If the quadratic term's **p-value < 0.05**, the relationship is non-linear."
    )

    predictor = st.selectbox("Select Predictor", ["TV", "radio", "newspaper"])
    order = st.radio("Polynomial Order", [1, 2], horizontal=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.regplot(data=df, x=predictor, y="sales", order=order, ax=ax, scatter_kws={"alpha": 0.4})
    ax.set_title(f"sales ~ {predictor}  (order={order})")
    st.pyplot(fig)
    plt.close("all")

    # Quick linearity test — compare linear vs. quadratic model
    st.subheader("Linearity Test (Linear vs. Quadratic Term)")
    poly_col = predictor + "2"
    linear_fit = smf.ols(f"sales ~ {predictor}", df).fit()
    quad_fit = smf.ols(f"sales ~ {predictor} + {poly_col}", df).fit()

    col1, col2 = st.columns(2)
    col1.metric("Linear R²", f"{linear_fit.rsquared:.4f}")
    col2.metric("Quadratic R²", f"{quad_fit.rsquared:.4f}")

    quad_pvalue = quad_fit.pvalues[poly_col]
    verdict = "✅ Linear" if quad_pvalue > 0.05 else "⚠️ Non-linear"
    st.info(
        f"Quadratic term **{poly_col}** p-value = **{quad_pvalue:.4f}**  →  Relationship is **{verdict}**"
    )

# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL BUILDER
# ═══════════════════════════════════════════════════════════════════════════
elif section == "📐 Model Builder":
    st.header("Model Builder")
    st.markdown("Choose predictors and optional interaction/polynomial terms, then view the full OLS summary.")

    st.subheader("Quick Preset Models")
    preset = st.selectbox(
        "Load a preset",
        [
            "Custom",
            "sales ~ TV",
            "sales ~ radio",
            "sales ~ newspaper",
            "sales ~ TV + radio + newspaper",
            "sales ~ TV + radio",
            "sales ~ TV * radio  (interaction)",
            "sales ~ TV + radio + TV2 + radio2  (polynomial)",
            "sales ~ TV * radio * TV2  (full interaction + polynomial)",
        ],
    )

    preset_map = {
        "sales ~ TV": "sales~TV",
        "sales ~ radio": "sales~radio",
        "sales ~ newspaper": "sales~newspaper",
        "sales ~ TV + radio + newspaper": "sales~TV+radio+newspaper",
        "sales ~ TV + radio": "sales~TV+radio",
        "sales ~ TV * radio  (interaction)": "sales~TV*radio",
        "sales ~ TV + radio + TV2 + radio2  (polynomial)": "sales~TV+radio+TV2+radio2",
        "sales ~ TV * radio * TV2  (full interaction + polynomial)": "sales~TV*radio*TV2",
    }

    if preset != "Custom":
        formula_default = preset_map[preset]
    else:
        formula_default = "sales~TV+radio"

    formula = st.text_input(
        "Formula (statsmodels style)",
        value=formula_default,
        help="Use column names: TV, radio, newspaper, TV2, radio2, newspaper2. "
        "Use * for interaction (includes main effects), : for interaction only.",
    )

    if st.button("Fit Model"):
        try:
            fit = smf.ols(formula, df).fit()

            # ── Key metrics ──────────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("R²", f"{fit.rsquared:.4f}")
            col2.metric("Adj. R²", f"{fit.rsquared_adj:.4f}")
            col3.metric("F-statistic", f"{fit.fvalue:.1f}")
            col4.metric("AIC", f"{fit.aic:.1f}")

            # ── Coefficients table ───────────────────────────────────────
            st.subheader("Coefficients")
            coef_df = pd.DataFrame(
                {
                    "Coefficient": fit.params,
                    "Std Error": fit.bse,
                    "t-value": fit.tvalues,
                    "p-value": fit.pvalues,
                    "[0.025": fit.conf_int()[0],
                    "0.975]": fit.conf_int()[1],
                }
            ).round(6)
            # Highlight significant rows
            def highlight_sig(row):
                color = "background-color: #d4f1c0" if row["p-value"] < 0.05 else ""
                return [color] * len(row)

            st.dataframe(coef_df.style.apply(highlight_sig, axis=1), use_container_width=True)
            st.caption("🟢 Green rows: p-value < 0.05 (statistically significant at 5% level)")

            # ── Full OLS summary ─────────────────────────────────────────
            with st.expander("Full OLS Summary"):
                buf = io.StringIO()
                buf.write(fit.summary().as_text())
                st.code(buf.getvalue())

            # ── Residual plots ───────────────────────────────────────────
            st.subheader("Residual Diagnostics")
            residuals = fit.resid
            fitted = fit.fittedvalues

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].scatter(fitted, residuals, alpha=0.4)
            axes[0].axhline(0, color="red", linestyle="--")
            axes[0].set_xlabel("Fitted Values")
            axes[0].set_ylabel("Residuals")
            axes[0].set_title("Residuals vs. Fitted")

            sm.qqplot(residuals, line="45", ax=axes[1], alpha=0.5)
            axes[1].set_title("Q-Q Plot of Residuals")

            axes[2].hist(residuals, bins=25, edgecolor="black", color="steelblue", alpha=0.7)
            axes[2].set_xlabel("Residuals")
            axes[2].set_title("Residual Distribution")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close("all")

        except Exception as e:
            st.error(f"Model fitting failed: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PREDICT SALES
# ═══════════════════════════════════════════════════════════════════════════
elif section == "🔮 Predict Sales":
    st.header("Predict Sales")
    st.markdown(
        "Uses the best interaction model: **sales ~ TV \\* radio** (R² = 0.968)"
    )

    # Fit the interaction model
    model = smf.ols("sales~TV*radio", df).fit()

    st.subheader("Enter Ad Spend Values")
    col1, col2 = st.columns(2)
    tv_val = col1.slider("TV Budget ($000s)", 0.0, 300.0, 150.0, step=0.5)
    radio_val = col2.slider("Radio Budget ($000s)", 0.0, 50.0, 20.0, step=0.5)

    input_df = pd.DataFrame({"TV": [tv_val], "radio": [radio_val]})
    prediction = model.predict(input_df)[0]

    st.metric("Predicted Sales (units $000s)", f"{prediction:.2f}")

    # Show formula calculation
    c = model.params
    st.markdown(
        f"""
        **Formula:**  
        sales = {c['Intercept']:.4f}  
        &emsp;&emsp;+ {c['TV']:.4f} × {tv_val}  
        &emsp;&emsp;+ {c['radio']:.4f} × {radio_val}  
        &emsp;&emsp;+ {c['TV:radio']:.6f} × {tv_val} × {radio_val}  
        &emsp;&emsp;= **{prediction:.4f}**
        """
    )

    # Heatmap of predictions across TV & radio values
    st.subheader("Prediction Heatmap (TV × Radio)")
    tv_range = np.linspace(0, 300, 40)
    radio_range = np.linspace(0, 50, 40)
    TV_grid, Radio_grid = np.meshgrid(tv_range, radio_range)
    grid_df = pd.DataFrame({"TV": TV_grid.ravel(), "radio": Radio_grid.ravel()})
    preds = model.predict(grid_df).values.reshape(TV_grid.shape)

    fig, ax = plt.subplots(figsize=(9, 5))
    cf = ax.contourf(TV_grid, Radio_grid, preds, levels=20, cmap="YlOrRd")
    plt.colorbar(cf, ax=ax, label="Predicted Sales ($000s)")
    ax.set_xlabel("TV Budget ($000s)")
    ax.set_ylabel("Radio Budget ($000s)")
    ax.set_title("Predicted Sales Across TV & Radio Budgets")
    ax.scatter([tv_val], [radio_val], color="blue", s=120, zorder=5, label=f"Your input ({prediction:.1f})")
    ax.legend()
    st.pyplot(fig)
    plt.close("all")
