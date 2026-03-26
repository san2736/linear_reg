import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf

st.title("Advanced Linear Regression (OLS) App")

# Upload CSV
file = st.file_uploader("Upload CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.write(df.head())

    columns = df.columns.tolist()

    # Select dependent variable
    y = st.selectbox("Select Dependent Variable (Y)", columns)

    # Select independent variables (multiple)
    X = st.multiselect("Select Independent Variables (X)", columns)

    if y and X:

        # Option: polynomial term
        poly_var = st.selectbox("Optional: Add squared term for variable", ["None"] + X)

        if poly_var != "None":
            df[f"{poly_var}2"] = df[poly_var] ** 2

        # Option: interaction
        st.subheader("Optional Interaction Term")
        inter1 = st.selectbox("Interaction Variable 1", ["None"] + X)
        inter2 = st.selectbox("Interaction Variable 2", ["None"] + X)

        # Build formula
        formula = y + " ~ " + " + ".join(X)

        if poly_var != "None":
            formula += f" + {poly_var}2"

        if inter1 != "None" and inter2 != "None" and inter1 != inter2:
            formula += f" + {inter1}:{inter2}"

        st.write("Model Formula:", formula)

        if st.button("Run Regression"):

            model = smf.ols(formula, df).fit()

            st.subheader("Regression Summary")
            st.text(model.summary())

            # Extract p-values
            pvals = model.pvalues

            st.subheader("P-Values Interpretation")

            results = []
            for var, p in pvals.items():
                if p < 0.05:
                    results.append(f"{var} → Significant (p = {round(p,5)})")
                else:
                    results.append(f"{var} → NOT Significant (p = {round(p,5)})")

            for r in results:
                st.write(r)
