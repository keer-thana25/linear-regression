"""
Streamlit ML Pipeline Application - Linear Regression
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="ML Pipeline - Linear Regression",
    page_icon="📊",
    layout="wide",
)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_style("whitegrid")

# =====================================================
# SESSION STATE
# =====================================================
if "current_step" not in st.session_state:
    st.session_state.current_step = 1

keys = [
    "data","cleaned_data","target_variable","feature_variables",
    "model","predictions","X_train","X_test","y_train","y_test"
]

for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None

# =====================================================
# STEP NAMES
# =====================================================
step_names = {
    1: "Upload Dataset",
    2: "Understand Dataset",
    3: "Data Cleaning",
    4: "Feature Selection",
    5: "Visualization",
    6: "Train/Test Split",
    7: "Model Training",
    8: "Prediction",
    9: "Model Evaluation",
}

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🧭 Navigation")

st.sidebar.markdown("### Current Step")
st.sidebar.info(
    f"{st.session_state.current_step}. {step_names[st.session_state.current_step]}"
)

selected_step = st.sidebar.radio(
    "Jump to Step",
    options=list(step_names.values()),
    index=st.session_state.current_step - 1,
)

jump_step = list(step_names.keys())[list(step_names.values()).index(selected_step)]

if jump_step != st.session_state.current_step:
    st.session_state.current_step = jump_step
    st.rerun()

if st.sidebar.button("🔄 Reset Pipeline"):
    st.session_state.clear()
    st.session_state.current_step = 1
    st.rerun()

# =====================================================
# HEADER
# =====================================================
st.markdown(
    "<h1 style='text-align:center;color:#1f77b4'>📊 Machine Learning Pipeline - Linear Regression</h1>",
    unsafe_allow_html=True,
)

st.progress(st.session_state.current_step / 9)

st.markdown(
    f"**Step {st.session_state.current_step} of 9:** {step_names[st.session_state.current_step]}"
)

st.markdown("---")

# =====================================================
# STEP 1 : UPLOAD DATASET
# =====================================================
if st.session_state.current_step == 1:

    st.header("Upload Dataset")

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded:

        df = pd.read_csv(uploaded)

        st.session_state.data = df

        st.success("Dataset loaded")

        st.dataframe(df.head())

# =====================================================
# STEP 2 : UNDERSTAND DATASET
# =====================================================
elif st.session_state.current_step == 2:

    st.header("Understand Dataset")

    if st.session_state.data is None:
        st.warning("Upload dataset first")

    else:

        df = st.session_state.data

        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("Column Types")
        st.write(df.dtypes)

        st.subheader("Statistics")
        st.write(df.describe())

# =====================================================
# STEP 3 : DATA CLEANING
# =====================================================
elif st.session_state.current_step == 3:

    st.header("Data Cleaning")

    if st.session_state.data is None:
        st.warning("Upload dataset first")

    else:

        df = st.session_state.data

        if st.button("Clean Data"):

            cleaned = df.dropna()

            st.session_state.cleaned_data = cleaned

            st.success("Missing values removed")

            st.write("New Shape:", cleaned.shape)

# =====================================================
# STEP 4 : FEATURE SELECTION
# =====================================================
elif st.session_state.current_step == 4:

    st.header("Feature Selection")

    df = (
        st.session_state.cleaned_data
        if st.session_state.cleaned_data is not None
        else st.session_state.data
    )

    if df is None:
        st.warning("Upload dataset")

    else:

        numeric_cols = df.select_dtypes(include=np.number).columns

        target = st.selectbox("Target Variable", numeric_cols)

        features = st.multiselect(
            "Features",
            [c for c in numeric_cols if c != target],
        )

        st.session_state.target_variable = target
        st.session_state.feature_variables = features

# =====================================================
# STEP 5 : VISUALIZATION
# =====================================================
elif st.session_state.current_step == 5:

    st.header("📊 Data Visualization")

    df = (
        st.session_state.cleaned_data
        if st.session_state.cleaned_data is not None
        else st.session_state.data
    )

    if df is None:
        st.warning("Upload dataset first")

    else:

        viz_type = st.selectbox(
            "Choose Chart",
            [
                "Target vs Features",
                "Histogram",
                "Bar Plot",
                "Line Plot",
                "Scatter Plot",
                "Box Plot",
                "Correlation Heatmap",
                "Pair Plot"
            ]
        )

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()

        target = st.session_state.target_variable
        features = st.session_state.feature_variables

        # =====================================================
        # TARGET VS FEATURES (NEW)
        # =====================================================
        if viz_type == "Target vs Features":

            if target is None or not features:
                st.warning("Please select Target and Features in Step 4")

            else:

                st.subheader(f"{target} vs Selected Features")

                for feature in features:

                    fig, ax = plt.subplots()

                    # If feature is numeric → scatter
                    if feature in numeric_cols:

                        sns.scatterplot(
                            x=df[feature],
                            y=df[target],
                            ax=ax
                        )

                    # If feature is categorical → boxplot
                    else:

                        sns.boxplot(
                            x=df[feature],
                            y=df[target],
                            ax=ax
                        )

                        plt.xticks(rotation=45)

                    ax.set_xlabel(feature)
                    ax.set_ylabel(target)
                    ax.set_title(f"{target} vs {feature}")

                    st.pyplot(fig)

        # =====================================================
        # HISTOGRAM
        # =====================================================
        elif viz_type == "Histogram":

            column = st.selectbox("Select Column", numeric_cols)

            fig, ax = plt.subplots()

            sns.histplot(df[column], kde=True, ax=ax)

            st.pyplot(fig)

        # =====================================================
        # BAR PLOT
        # =====================================================
        elif viz_type == "Bar Plot":

            x_col = st.selectbox("X Axis", all_cols)
            y_col = st.selectbox("Y Axis", numeric_cols)

            fig, ax = plt.subplots()

            sns.barplot(x=df[x_col], y=df[y_col], ax=ax)

            plt.xticks(rotation=45)

            st.pyplot(fig)

        # =====================================================
        # LINE PLOT
        # =====================================================
        elif viz_type == "Line Plot":

            x_col = st.selectbox("X Axis", numeric_cols)
            y_col = st.selectbox("Y Axis", numeric_cols)

            fig, ax = plt.subplots()

            sns.lineplot(x=df[x_col], y=df[y_col], ax=ax)

            st.pyplot(fig)

        # =====================================================
        # SCATTER PLOT
        # =====================================================
        elif viz_type == "Scatter Plot":

            x_col = st.selectbox("X Axis", numeric_cols)
            y_col = st.selectbox("Y Axis", numeric_cols)

            fig, ax = plt.subplots()

            sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)

            st.pyplot(fig)

        # =====================================================
        # BOX PLOT
        # =====================================================
        elif viz_type == "Box Plot":

            column = st.selectbox("Select Column", numeric_cols)

            fig, ax = plt.subplots()

            sns.boxplot(y=df[column], ax=ax)

            st.pyplot(fig)

        # =====================================================
        # CORRELATION HEATMAP
        # =====================================================
        elif viz_type == "Correlation Heatmap":

            numeric_df = df.select_dtypes(include=np.number)

            fig, ax = plt.subplots(figsize=(10,6))

            sns.heatmap(
                numeric_df.corr(),
                annot=True,
                cmap="coolwarm",
                ax=ax
            )

            st.pyplot(fig)

        # =====================================================
        # PAIR PLOT
        # =====================================================
        elif viz_type == "Pair Plot":

            pair_df = df[numeric_cols]

            fig = sns.pairplot(pair_df)

            st.pyplot(fig)

# =====================================================
# STEP 6 : TRAIN TEST SPLIT
# =====================================================
elif st.session_state.current_step == 6:

    st.markdown("## ✂️ Train/Test Split")

    df = (
        st.session_state.cleaned_data
        if st.session_state.cleaned_data is not None
        else st.session_state.data
    )

    if df is None:
        st.warning("Upload dataset first")

    elif not st.session_state.feature_variables:
        st.warning("Select features first")

    else:

        X = df[st.session_state.feature_variables]
        y = df[st.session_state.target_variable]

        st.write("Test Size: 20%")
        st.write("Random State: 42")

        if st.button("Split Data"):

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            st.success("Data split successfully!")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("X Train", X_train.shape[0])
            col2.metric("X Test", X_test.shape[0])
            col3.metric("y Train", len(y_train))
            col4.metric("y Test", len(y_test))

# =====================================================
# STEP 7 : MODEL TRAINING
# =====================================================
elif st.session_state.current_step == 7:

    st.markdown("## 🤖 Model Training")

    if st.session_state.X_train is None:
        st.warning("Split data first")

    else:

        X_train = st.session_state.X_train
        y_train = st.session_state.y_train

        st.info("Algorithm: Linear Regression")
        st.info(f"Training Samples: {len(X_train)}")
        st.info(f"Features: {len(X_train.columns)}")

        if st.button("Train Model"):

            model = LinearRegression()

            model.fit(X_train, y_train)

            st.session_state.model = model

            st.success("Model trained successfully!")

            coef_df = pd.DataFrame({
                "Feature": X_train.columns,
                "Coefficient": model.coef_
            })

            st.dataframe(coef_df)

            col1, col2 = st.columns(2)

            col1.metric("Intercept", f"{model.intercept_:.4f}")

            train_score = model.score(X_train, y_train)

            col2.metric("Training R²", f"{train_score:.4f}")

# =====================================================
# STEP 8 : PREDICTION
# =====================================================
elif st.session_state.current_step == 8:

    st.markdown("## 🔮 Prediction")

    if st.session_state.model is None:
        st.warning("Train model first")

    else:

        if st.button("Generate Predictions"):

            preds = st.session_state.model.predict(
                st.session_state.X_test
            )

            st.session_state.predictions = preds

            results_df = pd.DataFrame({
                "Actual": st.session_state.y_test.values,
                "Predicted": preds
            })

            st.dataframe(results_df.head(20))

# =====================================================
# STEP 9 : MODEL EVALUATION
# =====================================================
elif st.session_state.current_step == 9:

    st.markdown("## 📉 Model Evaluation")

    if st.session_state.predictions is None:
        st.warning("Generate predictions first")

    else:

        y_test = st.session_state.y_test
        preds = st.session_state.predictions

        mae = mean_absolute_error(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, preds)

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("MAE", f"{mae:.4f}")
        col2.metric("MSE", f"{mse:.4f}")
        col3.metric("RMSE", f"{rmse:.4f}")
        col4.metric("R²", f"{r2:.4f}")

        if r2 >= 0.7:
            st.success("Good Model Performance")
        elif r2 >= 0.4:
            st.warning("Moderate Model Performance")
        else:
            st.error("Low Model Performance")

        fig, ax = plt.subplots()

        ax.scatter(y_test, preds)

        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()],
                "r--")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

        st.pyplot(fig)

# =====================================================
# FOOTER NAVIGATION
# =====================================================
st.markdown("---")

col1, col2, col3 = st.columns([5,1,1])

with col2:

    if st.button("⬅️ Previous"):

        if st.session_state.current_step > 1:

            st.session_state.current_step -= 1

            st.rerun()

with col3:

    if st.button("Next ➡️"):

        if st.session_state.current_step < 9:

            st.session_state.current_step += 1

            st.rerun()

st.markdown("---")

st.markdown(
    "<div style='text-align:center;color:gray'>ML Pipeline Application | Built with Streamlit & Scikit-learn</div>",
    unsafe_allow_html=True,
)