# ❤️ Heart Disease Prediction using Machine Learning & Streamlit

A machine learning project that predicts the presence of **heart disease** from clinical attributes using classification algorithms and provides an interactive prediction interface built with **Streamlit**.

> **Disclaimer:** This project is intended for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical advice.

---

## 📌 Project Overview

Heart disease prediction is a binary classification problem where machine learning can be used to identify patterns associated with the presence or absence of heart disease.

In this project, I:

* Performed exploratory data analysis on a heart disease dataset.
* Prepared numerical and categorical features for machine learning.
* Applied **one-hot encoding** to categorical variables.
* Used **StandardScaler** for feature scaling.
* Trained and compared multiple classification algorithms.
* Selected **Logistic Regression** for the Streamlit application.
* Saved the trained model, scaler, and expected feature columns using `joblib`.
* Built an interactive web application using Streamlit for real-time predictions.

---

## 🎯 Objective

The primary objective is to build an end-to-end machine learning workflow that takes patient-related input features and predicts the target variable:

```text
HeartDisease
```

where:

```text
0 → Heart disease not predicted
1 → Heart disease predicted
```

---

## 📊 Dataset

The dataset contains **918 records and 12 columns**, including the target variable `HeartDisease`.

### Features

| Feature          | Description                      |
| ---------------- | -------------------------------- |
| `Age`            | Age of the patient               |
| `Sex`            | Biological sex                   |
| `ChestPainType`  | Type of chest pain               |
| `RestingBP`      | Resting blood pressure           |
| `Cholesterol`    | Cholesterol level                |
| `FastingBS`      | Fasting blood sugar indicator    |
| `RestingECG`     | Resting electrocardiogram result |
| `MaxHR`          | Maximum heart rate               |
| `ExerciseAngina` | Exercise-induced angina          |
| `Oldpeak`        | ST depression                    |
| `ST_Slope`       | Slope of the ST segment          |
| `HeartDisease`   | Target variable                  |

The dataset contains a mixture of numerical and categorical features.

---

## 🔎 Exploratory Data Analysis

The project includes exploratory analysis using:

* Pandas
* NumPy
* Matplotlib
* Seaborn

Examples of analysis performed include:

* Dataset shape and structure
* Data type inspection
* Statistical summaries
* Target distribution
* Feature relationships
* Visualization of numerical variables
* Analysis of feature distributions and patterns

---

## ⚙️ Data Preprocessing

### 1. Categorical Encoding

Categorical features were converted into numerical features using:

```python
pd.get_dummies(df, drop_first=True)
```

The `drop_first=True` option was used to avoid redundant dummy variables.

For example:

```text
Sex
├── Female → 0
└── Male   → 1
```

The preprocessing produces features such as:

```text
Sex_M
ChestPainType_ATA
ChestPainType_NAP
ChestPainType_TA
RestingECG_Normal
RestingECG_ST
ExerciseAngina_Y
ST_Slope_Flat
ST_Slope_Up
```

### 2. Feature Scaling

Numerical/model input features were standardized using:

```python
StandardScaler()
```

The scaler is fitted only on the training data:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The same fitted scaler is then used when processing new inputs in the Streamlit application.

---

## 🤖 Machine Learning Models

The project compares several classification algorithms:

* Logistic Regression
* K-Nearest Neighbors (KNN)
* Gaussian Naive Bayes
* Decision Tree
* Support Vector Machine (SVM)

The models were evaluated using:

* Accuracy
* F1 Score

### Model Comparison

The following results were recorded in the current notebook experiment:

| Model                | Accuracy | F1 Score |
| -------------------- | -------: | -------: |
| Logistic Regression  |   87.32% |   88.96% |
| KNN                  |   85.51% |   87.42% |
| Gaussian Naive Bayes |   86.96% |   88.61% |
| Decision Tree        |   76.09% |   78.85% |
| SVM                  |   86.23% |   88.27% |

These values correspond to the experiment recorded in the notebook and may change when the training pipeline is rerun after preprocessing corrections.

---

## 🧠 Selected Model

**Logistic Regression** is used in the Streamlit application.

The trained model is saved as:

```text
logistic_heart.pkl
```

The preprocessing scaler is saved as:

```text
scaler.pkl
```

The expected model feature columns are saved as:

```text
columns.pkl
```

This allows the Streamlit application to reproduce the same feature structure and scaling process used during training.

---

## 🌐 Streamlit Application

The project includes an interactive Streamlit interface where users can enter:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise-induced angina
* Oldpeak
* ST slope

The application converts the inputs into the same feature representation used during model training and passes the processed data to the trained Logistic Regression model.

### Application Workflow

```text
User Input
    ↓
Streamlit Interface
    ↓
Feature Encoding
    ↓
Column Alignment
    ↓
StandardScaler
    ↓
Logistic Regression
    ↓
Prediction
    ↓
Streamlit Result
```

---

## 🗂️ Project Structure

```text
heart/
│
├── app.py
│
├── heart.ipynb
│
├── heart.csv
│
├── logistic_heart.pkl
├── scaler.pkl
├── columns.pkl
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

### File Description

| File                 | Purpose                                                     |
| -------------------- | ----------------------------------------------------------- |
| `app.py`             | Streamlit application                                       |
| `heart.ipynb`        | Data analysis, preprocessing, model training and evaluation |
| `heart.csv`          | Dataset                                                     |
| `logistic_heart.pkl` | Saved Logistic Regression model                             |
| `scaler.pkl`         | Saved feature scaler                                        |
| `columns.pkl`        | Saved expected feature-column order                         |
| `requirements.txt`   | Python dependencies                                         |
| `README.md`          | Project documentation                                       |
| `.gitignore`         | Files excluded from Git                                     |

---

## 🛠️ Technologies Used

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn

### Model Persistence

* Joblib

### Web Application

* Streamlit

### Development Tools

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/heart-disease-prediction-streamlit.git
```

### 2. Navigate to the project

```bash
cd heart-disease-prediction-streamlit
```

### 3. Create or activate the environment

Using Conda:

```bash
conda activate streamlit_env
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📈 Machine Learning Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Data Preprocessing
   ↓
One-Hot Encoding
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Selection
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

## 💡 Key Learning Outcomes

Through this project, I worked with an end-to-end machine learning workflow, including:

* Understanding a real-world classification dataset
* Exploratory data analysis
* Categorical feature encoding
* Feature scaling
* Training multiple classification models
* Comparing model performance
* Saving trained ML artifacts using Joblib
* Preparing model inputs for deployment
* Building a user-facing ML application with Streamlit
* Connecting a trained machine learning model to a web interface

---

## 🔮 Future Improvements

Some possible improvements for future versions include:

* Hyperparameter tuning
* Cross-validation
* More detailed model evaluation
* ROC-AUC analysis
* Confusion matrix visualization
* Feature importance and model interpretation
* Probability-based prediction display
* Improved input validation
* Better UI/UX
* Deploying the Streamlit application online

---

## ⚠️ Limitations

This project is an educational machine learning application.

The prediction depends on the dataset, preprocessing pipeline, model assumptions, and model performance. A machine learning prediction should **not** be interpreted as a medical diagnosis or used independently to make healthcare decisions.

---

## 👨‍💻 Author

**Himanshu Prasad**

B.Tech — Computer Science & Engineering (Data Science)

Interested in:

* Data Analytics
* Machine Learning
* Data Science
* Applied AI

---

## ⭐ Project Highlights

**End-to-end ML project**

**Multiple classification algorithms compared**

**Feature preprocessing and scaling**

**Saved ML model for inference**

**Interactive Streamlit application**

**GitHub portfolio project**
