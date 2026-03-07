# 📊 Machine Learning Pipeline Application – Linear Regression

## 📌 Project Overview

This project is a **Streamlit-based Machine Learning Pipeline Application** that demonstrates the complete workflow of building a **Linear Regression model**.
The application allows users to upload a dataset and go through each stage of the machine learning process step-by-step.

The pipeline is designed to help users **understand how machine learning models are built**, from data loading to model evaluation, with interactive visualizations and controls.

---

# 🚀 Features

The application implements a **9-step Machine Learning Pipeline**:

1. **Upload Dataset**

   * Upload a CSV dataset.
   * Preview the data.

2. **Understand Dataset**

   * View dataset shape.
   * Check column data types.
   * View summary statistics.

3. **Data Cleaning**

   * Automatically removes missing values.

4. **Feature Selection**

   * Select **Target Variable**
   * Select **Feature Variables**

5. **Data Visualization**
   Includes multiple visualization options:

   * Target vs Features (important for ML analysis)
   * Histogram
   * Bar Plot
   * Line Plot
   * Scatter Plot
   * Box Plot
   * Correlation Heatmap
   * Pair Plot

6. **Train/Test Split**

   * Splits dataset into training and testing sets.
   * Default split: **80% training / 20% testing**

7. **Model Training**

   * Uses **Linear Regression**
   * Displays:

     * Feature coefficients
     * Intercept
     * Training R² score

8. **Prediction**

   * Generates predictions using the trained model.
   * Displays **Actual vs Predicted values**.

9. **Model Evaluation**
   Displays important regression metrics:

   * MAE (Mean Absolute Error)
   * MSE (Mean Squared Error)
   * RMSE (Root Mean Squared Error)
   * R² Score

---

# 🧠 Technologies Used

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**

---

# 📂 Project Structure

```
ml-pipeline-streamlit/
│
├── app.py                # Main Streamlit application
├── dataset.csv           # Example dataset (optional)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/ml-pipeline-streamlit.git
cd ml-pipeline-streamlit
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements file is not available:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

---

# ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The app will open automatically in your browser.

---

# 📊 Example Workflow

1. Upload a dataset (CSV)
2. Explore and understand the data
3. Clean the dataset
4. Select target and feature variables
5. Visualize feature relationships
6. Split the dataset
7. Train the Linear Regression model
8. Generate predictions
9. Evaluate model performance

---

# 📈 Example Use Cases

This project can be used for:

* Learning **Machine Learning pipelines**
* Understanding **Linear Regression**
* Practicing **data visualization**
* Demonstrating **ML workflow in Streamlit**
* Academic or portfolio projects

---

# 🔮 Future Improvements

Possible enhancements include:

* Support for multiple ML algorithms
* Feature importance visualization
* Model download option
* Hyperparameter tuning
* Classification models
* Advanced data preprocessing

---

# 👩‍💻 Author

Developed as a **Machine Learning learning project using Streamlit** to demonstrate a full ML workflow from data processing to model evaluation.

---
## 🌐 Live Demo

Try the application here:

https://linear-regressionmodel.streamlit.app/
