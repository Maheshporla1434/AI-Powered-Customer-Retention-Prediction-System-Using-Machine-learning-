

# 🧠 AI-Powered Customer Retention Prediction System Using Machine Learning

## 📄 Overview

This project implements an **AI-powered system to predict customer churn** — identifying customers likely to stop using a company’s service. The solution leverages machine learning models to analyze customer demographics, service usage, and billing behavior, helping businesses take proactive measures to improve retention and reduce revenue loss.

---

## 📚 Table of Contents

* [Overview](#-overview)
* [Objective](#-objective)
* [Features](#-features)
* [Tech Stack](#-tech-stack)
* [Dataset Description](#-dataset-description)
* [Project Workflow](#-project-workflow)
* [Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
* [Feature Engineering](#-feature-engineering)
* [Data Preprocessing](#-data-preprocessing)
* [Model Selection & Training](#-model-selection--training)
* [Best Model](#-best-model)
* [Results](#-results)
* [Future Scope](#-future-scope)
* [How to Run](#-how-to-run)
* [Project Structure](#-project-structure)
* [References](#-references)
* [Author](#-author)

---

## 🎯 Objective

To design and implement a **machine learning-based customer retention prediction system** that:

* Predicts potential customer churn.
* Identifies key factors contributing to churn.
* Provides actionable insights to help businesses improve retention strategies.

---

## 🌟 Features

* Data preprocessing including handling missing values, outliers, and imbalanced classes.
* Multiple feature transformation and scaling techniques.
* Model comparison and hyperparameter tuning using GridSearchCV.
* Explainable AI methods for interpretability.
* Comprehensive visualization and statistical insights.
* High accuracy churn prediction using Logistic Regression.

---

## 🧰 Tech Stack

| Category         | Tools / Libraries                    |
| ---------------- | ------------------------------------ |
| Language         | Python                               |
| Data Handling    | Pandas, NumPy                        |
| Visualization    | Matplotlib, Seaborn                  |
| Machine Learning | Scikit-learn, XGBoost                |
| Model Evaluation | ROC-AUC, F1-score, Precision, Recall |
| Development      | Jupyter Notebook / VS Code           |

---

## 📊 Dataset Description

The dataset includes customer information such as demographics, services, contracts, billing, and churn status.

**Key Columns:**

* `customerID`: Unique customer identifier
* `gender`, `SeniorCitizen`, `Partner`, `Dependents`
* `tenure`, `InternetService`, `OnlineSecurity`, `TechSupport`
* `Contract`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`
* `Churn`: Target variable (Yes/No)

---

## 🔄 Project Workflow

1. **Data Collection & Loading**

   * Imported dataset using Pandas and performed basic structure analysis.

2. **Exploratory Data Analysis (EDA)**

   * Visualized churn rates, contract types, payment methods, and correlations.
   * Discovered that short-term contracts, higher monthly charges, and lack of tech support strongly correlate with churn.

3. **Feature Engineering**

   * Missing value imputation using KNN, Random Sample, and Decision Tree Imputer.
   * Outlier handling via Winsorization.
   * Encoding categorical variables with Label and One-Hot Encoding.

4. **Feature Transformation**

   * Applied transformations (Log, Yeo-Johnson, Quantile) to normalize skewed data.

5. **Feature Selection**

   * Selected important variables using Chi-Square, Mutual Information, and F-test.

6. **Data Balancing**

   * Applied SMOTE (Synthetic Minority Oversampling Technique) to handle class imbalance.

7. **Feature Scaling**

   * Used Z-score Standardization for uniform feature scaling.

8. **Model Selection**

   * Trained and compared models: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost, and KNN.

9. **Hyperparameter Tuning**

   * Tuned parameters using **GridSearchCV** for optimized performance.

10. **Final Model**

    * Selected **Logistic Regression** as the best-performing model with balanced accuracy and interpretability.

---

## 🧮 Model Selection & Training

| Model               | Accuracy | ROC-AUC     | Remarks                        |
| ------------------- | -------- | ----------- | ------------------------------ |
| Logistic Regression | ✅ High   | ✅ Excellent | Best performing, interpretable |
| Random Forest       | Good     | High        | Slightly overfit               |
| Decision Tree       | Moderate | Medium      | Overfitting observed           |
| XGBoost             | High     | High        | Complex but similar accuracy   |
| SVM                 | Good     | High        | Longer training time           |
| KNN                 | Moderate | Low         | Sensitive to scaling           |

**Final Choice:** Logistic Regression

---

## 🏆 Best Model: Logistic Regression

* **Reason for Selection:**

  * High accuracy and AUC.
  * Computationally efficient.
  * Easily interpretable.
* **Optimized Parameters (via GridSearchCV):**

  * `C = 1.0`
  * `solver = 'lbfgs'`
  * `class_weight = 'balanced'`
  * `max_iter = 200`
  * `random_state = 42`

---

## 📈 Results

* **Accuracy:** 85–90% (approx.)
* **ROC-AUC:** 0.90
* **Insights:**

  * Customers with month-to-month contracts and high charges are most likely to churn.
  * Automatic payments (credit/bank transfer) correlate with higher loyalty.
  * Tech support and online security significantly reduce churn rates.

---

## 🚀 Future Scope

* Integration with real-time CRM dashboards.
* Use of deep learning for complex feature extraction.
* Incorporation of customer feedback and sentiment data.
* Deployment via Flask/Django web interface for business use.

---

## 🧩 Project Structure

churn_app/
│
├─ app.py
├─ churn_model.pkl
├─ standard_scalar.pkl
├─ templates/
│   ├─ base.html
│   ├─ index.html
│   ├─ result.html
│   ├─ about.html
│   └─ developer.html
└─ static/
    ├─ style.css
    └─ siva_image/
        ├─ jio.webp
        ├─ airtel.png
        ├─ vi.webp
        ├─ bsnl.webp
        └─ mahesh.png


---

## ⚙️ How to Run

1. **Clone Repository**

   ```bash
   git clone https://github.com/<your-username>/AI-Customer-Retention.git
   cd AI-Customer-Retention
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate       # (Linux/Mac)
   venv\Scripts\activate          # (Windows)
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run Jupyter Notebook**

   ```bash
   jupyter notebook
   ```

5. **Execute Model Script**

   ```bash
   python src/model_training.py
   ```

---

## 📚 References

* Verbeke, W., Martens, D., Mues, C., & Baesens, B. (2012). *Expert Systems with Applications.*
* Huang, B., Ling, C. X. (2005). *IEEE Transactions on Knowledge and Data Engineering.*
* Coussement, K., & Van den Poel, D. (2008). *Information & Management.*
* Idris, A., Khan, A., & Lee, Y. S. (2012). *Applied Intelligence.*

---

## 👩‍💻 Author
* Porla Mahesh.
* Data Science, Vihara Tech.
* 📧 Gmail id:maheshporla1434@gmail.com).
* 🌐 [LinkedIn Profile](www.linkedin.com/in/maheshporla264.
* Contact No:7993253813.
* Get here:https://ai-powered-customer-retention-prediction-2s70.onrender.com.


# AI-Powered-Customer-Retention-Prediction-System-Using-Machine-learning-
