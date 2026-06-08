# House Price Prediction 🏠📊

An end-to-end machine learning project for predicting house prices using the **California Housing Dataset**.

This project covers the full ML workflow: data loading, exploratory data analysis (EDA), preprocessing, feature engineering, model training, model comparison, hyperparameter tuning, and final evaluation.

---

## 📌 Project Objective

The goal of this project is to build a machine learning model that can predict house prices based on district-level housing data such as:

- Median income
- Housing age
- Number of rooms
- Number of bedrooms
- Population
- Households
- Location coordinates
- Ocean proximity

---

## 📂 Project Structure

```text
House_Price_SL/
│
├── house_prices_improved.ipynb   # Main notebook
├── README.md                     # Project documentation
├── .gitignore                    # Ignored files
└── datasets/                     # Dataset folder, created when running notebook
```

> Large model files such as `.pkl`, `.joblib`, and `.sav` should not be pushed to GitHub.

---

## 🧠 Machine Learning Workflow

The notebook follows these main steps:

### 1. Load the Dataset

The project uses the California housing dataset from the *Hands-On Machine Learning* dataset repository.

If the dataset is not found locally, the notebook downloads it automatically.

### 2. Explore the Data

Initial exploration includes:

- Dataset shape
- Data types
- Missing values
- Descriptive statistics
- Categorical feature counts

### 3. Exploratory Data Analysis

The EDA section includes:

- Feature distributions
- Geographic scatter plots
- Correlation analysis
- Median income vs median house value
- Ocean proximity analysis

### 4. Train/Test Split

The data is split using **stratified sampling** based on income categories.

This helps keep the train and test sets balanced and representative.

### 5. Feature Engineering

New useful features are created:

- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

These ratios are often more meaningful than the original raw totals.

### 6. Data Preprocessing

The preprocessing pipeline handles:

- Missing numerical values
- Feature scaling
- Categorical encoding using One-Hot Encoding
- Custom feature engineering transformer

---

## 🤖 Models Used

The project compares several regression models:

| Model | Description |
|---|---|
| Linear Regression | Basic linear model |
| Ridge Regression | Linear model with L2 regularization |
| Lasso Regression | Linear model with L1 regularization |
| Decision Tree Regressor | Tree-based nonlinear model |
| Random Forest Regressor | Ensemble model using many decision trees |

---

## 🔍 L1 vs L2 Regularization

Regularization helps reduce overfitting by adding a penalty to the model.

### L1 Regularization — Lasso

L1 adds a penalty based on the **absolute value** of the coefficients.

It can shrink some coefficients exactly to zero, so it can work like automatic feature selection.

Use L1 when you want the model to ignore weak or less useful features.

### L2 Regularization — Ridge

L2 adds a penalty based on the **squared value** of the coefficients.

It shrinks coefficients toward zero but usually does not make them exactly zero.

Use L2 when you want a more stable model that keeps most features.

---

## 📊 Evaluation Metrics

The models are evaluated using:

| Metric | Meaning |
|---|---|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| R² Score | How much variance the model explains |

Lower RMSE and MAE are better.  
Higher R² is better.

---

## ⚙️ Hyperparameter Tuning

The Random Forest model is tuned using `RandomizedSearchCV`.

The tuning searches for better values of:

- Number of trees
- Maximum features
- Bootstrap usage

The best model is then evaluated on the test set.

---

## 📈 Feature Importance

The final Random Forest model is used to show the most important features for prediction.

This helps explain which features had the biggest impact on house price predictions.

---

## 💾 Model Saving

The notebook saves the final model using `joblib`.

Example:

```python
joblib.dump(final_model, "house_price_model.pkl")
```

However, because model files can be large, they should be ignored by Git.

Recommended `.gitignore`:

```gitignore
*.pkl
*.joblib
*.sav
__pycache__/
.ipynb_checkpoints/
datasets/
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/MohandAmr/House_Price_SL.git
```

### 2. Open the project folder

```bash
cd House_Price_SL
```

### 3. Install the required libraries

```bash
pip install numpy pandas matplotlib scikit-learn joblib
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
house_prices_improved.ipynb
```

Run the notebook cells from top to bottom.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib
- Jupyter Notebook

---

## ✅ What This Project Demonstrates

This project demonstrates important supervised machine learning skills:

- Data cleaning
- Data visualization
- EDA
- Feature engineering
- Pipelines
- Encoding categorical variables
- Scaling numerical variables
- Training regression models
- Comparing models
- Cross-validation
- Hyperparameter tuning
- Final model evaluation
- Saving trained models

---

## 📌 Notes

This project is for learning and practice.  
It is a good beginner-friendly end-to-end machine learning project because it covers the full workflow from raw data to final model evaluation.

---

## 👤 Author

**Mohand Amr**

GitHub: [MohandAmr](https://github.com/MohandAmr)
