"""
Streamlit ML Pipeline Application - Linear Regression
=====================================================
A professional Streamlit application demonstrating a complete 
Machine Learning pipeline using Linear Regression.

This app supports Multiple Linear Regression with advanced visualizations
and step-by-step pipeline navigation.

Author: ML Developer
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import io

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="ML Pipeline - Linear Regression",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for matplotlib and seaborn
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style("whitegrid")

# =====================================================
# SESSION STATE INITIALIZATION
# =====================================================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'target_variable' not in st.session_state:
    st.session_state.target_variable = None
if 'feature_variables' not in st.session_state:
    st.session_state.feature_variables = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("🧭 Pipeline Steps")
st.sidebar.markdown("---")

# Step options
steps = [
    "Step 1: Upload Dataset",
    "Step 2: Understand Dataset",
    "Step 3: Data Cleaning",
    "Step 4: Feature Selection",
    "Step 5: Visualization",
    "Step 6: Train/Test Split",
    "Step 7: Model Training",
    "Step 8: Prediction",
    "Step 9: Model Evaluation"
]

selected_step = st.sidebar.radio("Select Step:", steps)

st.sidebar.markdown("---")

# Reset button
if st.sidebar.button("🔄 Reset Pipeline"):
    st.session_state.data = None
    st.session_state.cleaned_data = None
    st.session_state.target_variable = None
    st.session_state.feature_variables = None
    st.session_state.model = None
    st.session_state.predictions = None
    st.rerun()

# =====================================================
# MAIN HEADER
# =====================================================
st.markdown("""
<h1 style='text-align: center; color: #1f77b4; margin-bottom: 30px;'>
    📊 Machine Learning Pipeline - Linear Regression
</h1>
""", unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# STEP 1: UPLOAD DATASET
# =====================================================
if "Step 1" in selected_step:
    st.markdown("## 📁 Step 1: Upload Dataset")
    st.markdown("Upload a CSV file to begin the ML pipeline.")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing your dataset"
    )
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
    
    # Show dataset if already loaded
    if st.session_state.data is not None:
        df = st.session_state.data
        st.success("✅ Dataset loaded!")
        
        st.markdown("### Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        
        st.markdown("### Column Names and Data Types")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str)
        })
        st.dataframe(col_info, use_container_width=True)
    else:
        st.info("👆 Please upload a CSV file to begin.")

# =====================================================
# STEP 2: UNDERSTAND DATASET
# =====================================================
elif "Step 2" in selected_step:
    st.markdown("## 🔍 Step 2: Understand Dataset")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first (Step 1).")
    else:
        df = st.session_state.data
        
        st.markdown("### Dataset Shape")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        
        st.markdown("### Column Names")
        st.write(", ".join(df.columns.tolist()))
        
        st.markdown("### Data Types")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.markdown("### Missing Values")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values,
                'Missing %': (missing.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values found!")
        
        st.markdown("### Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)

# =====================================================
# STEP 3: DATA CLEANING
# =====================================================
elif "Step 3" in selected_step:
    st.markdown("## 🧹 Step 3: Data Cleaning")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first (Step 1).")
    else:
        df = st.session_state.data
        
        st.markdown("### Current Missing Values")
        missing = df.isnull().sum()
        if missing.sum() > 0:
            missing_df = pd.DataFrame({
                'Column': missing.index,
                'Missing Count': missing.values
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        else:
            st.success("No missing values found in the dataset!")
        
        # Check if there are missing values
        missing_count = df.isnull().sum().sum()
        
        if missing_count == 0:
            st.success("No missing values found! You can proceed without cleaning.")
            if st.button("✅ Skip Cleaning (Data is Clean)", type="secondary"):
                st.session_state.cleaned_data = df
                st.success("✅ Proceeding with original data!")
        else:
            st.info(f"Found {missing_count} missing values in the dataset.")
            
            st.markdown("### Cleaning Options")
            cleaning_method = st.radio(
                "Select cleaning method:",
                ["Drop rows with missing values", 
                 "Fill missing values with mean", 
                 "Fill missing values with median"]
            )
            
            if st.button("🧹 Clean Data", type="primary"):
                df_cleaned = df.copy()
                
                if cleaning_method == "Drop rows with missing values":
                    df_cleaned = df.dropna()
                elif cleaning_method == "Fill missing values with mean":
                    for col in df_cleaned.columns:
                        if df_cleaned[col].dtype in ['int64', 'float64']:
                            df_cleaned[col].fillna(df_cleaned[col].mean(), inplace=True)
                elif cleaning_method == "Fill missing values with median":
                    for col in df_cleaned.columns:
                        if df_cleaned[col].dtype in ['int64', 'float64']:
                            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
                
                st.session_state.cleaned_data = df_cleaned
                st.success("✅ Data cleaned successfully!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Rows", df.shape[0])
                with col2:
                    st.metric("Cleaned Rows", df_cleaned.shape[0])
            
            # Allow skipping
            if st.button("➡️ Proceed Without Cleaning"):
                st.session_state.cleaned_data = df
                st.success("✅ Proceeding with original data!")
        
        if st.session_state.cleaned_data is not None:
            st.markdown("### Cleaned Data Preview")
            st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)

# =====================================================
# STEP 4: FEATURE SELECTION
# =====================================================
elif "Step 4" in selected_step:
    st.markdown("## 🎯 Step 4: Feature Selection")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first (Step 1).")
    else:
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if 'User_ID' in numeric_cols:
            numeric_cols.remove('User_ID')
        
        st.markdown("### Select Target Variable")
        target_var = st.selectbox(
            "Target Variable (Output - Y):",
            numeric_cols,
            help="This is the variable you want to predict"
        )
        
        st.markdown("### Select Input Features")
        feature_vars = st.multiselect(
            "Input Features (Independent Variables - X):",
            [col for col in numeric_cols if col != target_var],
            default=[col for col in numeric_cols if col != target_var][:3] if len(numeric_cols) > 1 else [],
            help="Select one or more features for Multiple Linear Regression"
        )
        
        st.session_state.target_variable = target_var
        st.session_state.feature_variables = feature_vars
        
        if target_var and feature_vars:
            st.markdown("### Selection Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Target Variable:** {target_var}")
            with col2:
                st.info(f"**Number of Features:** {len(feature_vars)}")
            
            st.write("**Selected Features:**")
            for feat in feature_vars:
                st.write(f"  • {feat}")

# =====================================================
# STEP 5: VISUALIZATION
# =====================================================
elif "Step 5" in selected_step:
    st.markdown("## 📈 Step 5: Visualization")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first (Step 1).")
    else:
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        viz_type = st.selectbox(
            "Select Visualization Type:",
            ["Correlation Heatmap", "Histograms", "Boxplots", 
             "Scatter Plots (Feature vs Target)", "Line Plots"]
        )
        
        if viz_type == "Correlation Heatmap":
            st.markdown("### Correlation Heatmap")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5, ax=ax)
            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            st.pyplot(fig)
            
            st.markdown("""
            **Interpretation:**
            - Values close to 1: Strong positive correlation
            - Values close to -1: Strong negative correlation
            - Values close to 0: No correlation
            """)
        
        elif viz_type == "Histograms":
            st.markdown("### Histograms for Numeric Columns")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:4])
            
            if selected_cols:
                n_cols = len(selected_cols)
                fig, axes = plt.subplots(n_cols, 1, figsize=(10, 3*n_cols))
                if n_cols == 1:
                    axes = [axes]
                for i, col in enumerate(selected_cols):
                    axes[i].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    axes[i].set_title(f'{col} Distribution', fontsize=12)
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
                plt.tight_layout()
                st.pyplot(fig)
        
        elif viz_type == "Boxplots":
            st.markdown("### Boxplots for Numeric Columns")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:4])
            
            if selected_cols:
                fig, ax = plt.subplots(figsize=(12, 6))
                df[selected_cols].boxplot(ax=ax)
                ax.set_title('Boxplots of Selected Columns', fontsize=14, fontweight='bold')
                ax.set_xticklabels(selected_cols, rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
        
        elif viz_type == "Scatter Plots (Feature vs Target)":
            st.markdown("### Scatter Plots: Features vs Target")
            
            if st.session_state.target_variable and st.session_state.feature_variables:
                target = st.session_state.target_variable
                features = st.session_state.feature_variables
                
                n_features = len(features)
                fig, axes = plt.subplots(n_features, 1, figsize=(10, 4*n_features))
                if n_features == 1:
                    axes = [axes]
                
                for i, feat in enumerate(features):
                    axes[i].scatter(df[feat], df[target], alpha=0.5)
                    axes[i].set_xlabel(feat, fontsize=12)
                    axes[i].set_ylabel(target, fontsize=12)
                    axes[i].set_title(f'{feat} vs {target}', fontsize=12, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("Please select target and features in Step 4 first.")
        
        elif viz_type == "Line Plots":
            st.markdown("### Line Plots for Numeric Columns")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_col = st.selectbox("Select column:", numeric_cols)
            
            if selected_col:
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(df[selected_col].values, alpha=0.7)
                ax.set_xlabel('Index', fontsize=12)
                ax.set_ylabel(selected_col, fontsize=12)
                ax.set_title(f'{selected_col} over Index', fontsize=14, fontweight='bold')
                st.pyplot(fig)

# =====================================================
# STEP 6: TRAIN/TEST SPLIT
# =====================================================
elif "Step 6" in selected_step:
    st.markdown("## ✂️ Step 6: Train/Test Split")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first (Step 1).")
    elif st.session_state.target_variable is None or st.session_state.feature_variables is None:
        st.warning("⚠️ Please select target and features in Step 4 first.")
    else:
        df = st.session_state.cleaned_data if st.session_state.cleaned_data is not None else st.session_state.data
        
        target = st.session_state.target_variable
        features = st.session_state.feature_variables
        
        X = df[features]
        y = df[target]
        
        st.markdown("### Split Configuration")
        st.write(f"Test Size: 20%")
        st.write(f"Random State: 42")
        
        if st.button("✂️ Split Data", type="primary"):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("✅ Data split successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("X Train", X_train.shape[0])
            with col2:
                st.metric("X Test", X_test.shape[0])
            with col3:
                st.metric("y Train", len(y_train))
            with col4:
                st.metric("y Test", len(y_test))

# =====================================================
# STEP 7: MODEL TRAINING
# =====================================================
elif "Step 7" in selected_step:
    st.markdown("## 🤖 Step 7: Model Training")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Please split the data in Step 6 first.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        
        st.markdown("### Model Configuration")
        st.info("**Algorithm:** Linear Regression")
        st.info(f"**Training Samples:** {len(X_train)}")
        st.info(f"**Number of Features:** {len(X_train.columns)}")
        
        if st.button("🤖 Train Model", type="primary"):
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            st.session_state.model = model
            
            st.success("✅ Model trained successfully!")
            
            st.markdown("### Model Coefficients")
            coef_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Coefficient': model.coef_
            })
            st.dataframe(coef_df, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Intercept", f"{model.intercept_:.4f}")
            with col2:
                train_score = model.score(X_train, y_train)
                st.metric("R² Score (Training)", f"{train_score:.4f}")

# =====================================================
# STEP 8: PREDICTION
# =====================================================
elif "Step 8" in selected_step:
    st.markdown("## 🔮 Step 8: Prediction")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model in Step 7 first.")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        if st.button("🔮 Generate Predictions", type="primary"):
            predictions = model.predict(X_test)
            st.session_state.predictions = predictions
            
            st.success("✅ Predictions generated successfully!")
            
            st.markdown("### Actual vs Predicted Values")
            results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': predictions
            })
            st.dataframe(results_df.head(20), use_container_width=True)
            
            st.caption("*Showing first 20 predictions*")

# =====================================================
# STEP 9: MODEL EVALUATION
# =====================================================
elif "Step 9" in selected_step:
    st.markdown("## 📉 Step 9: Model Evaluation")
    
    if st.session_state.predictions is None:
        st.warning("⚠️ Please generate predictions in Step 8 first.")
    else:
        y_test = st.session_state.y_test
        predictions = st.session_state.predictions
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        st.success("✅ Model evaluation complete!")
        
        st.markdown("### Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"{mae:.4f}")
            st.caption("Mean Absolute Error")
        with col2:
            st.metric("MSE", f"{mse:.4f}")
            st.caption("Mean Squared Error")
        with col3:
            st.metric("RMSE", f"{rmse:.4f}")
            st.caption("Root Mean Squared Error")
        with col4:
            st.metric("R² Score", f"{r2:.4f}")
            st.caption("Coefficient of Determination")
        
        st.markdown("### Interpretation")
        if r2 >= 0.7:
            st.success(f"✅ Good model! R² = {r2:.4f} indicates the model explains {r2*100:.1f}% of variance.")
        elif r2 >= 0.4:
            st.warning(f"⚠️ Moderate model. R² = {r2:.4f} indicates {r2*100:.1f}% of variance explained.")
        else:
            st.error(f"❌ Low R² = {r2:.4f}. Consider different features or model.")
        
        st.markdown("### Predicted vs Actual Values")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_test, predictions, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
               'r--', lw=2, label='Perfect Prediction')
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title('Predicted vs Actual Values', fontsize=14, fontweight='bold')
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("### Residual Plot")
        residuals = y_test.values - predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(predictions, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        
        st.markdown("### Download Model")
        buffer = io.BytesIO()
        joblib.dump(st.session_state.model, buffer)
        buffer.seek(0)
        
        st.download_button(
            label="💾 Download Model (.pkl)",
            data=buffer,
            file_name="linear_regression_model.pkl",
            mime="application/octet-stream"
        )

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p> Built with Streamlit & Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
