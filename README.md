# AI-ML-Intern-Task-1

# Titanic Dataset: Data Cleaning & Preprocessing for Machine Learning

This repository contains a Python script that demonstrates a complete data cleaning and preprocessing pipeline on the classic Titanic dataset. The goal is to transform the raw data into a clean, well-structured format suitable for training machine learning models.

## Project Overview

The project walks through the essential steps of preparing real-world data, addressing common issues like missing values, categorical data, varying feature scales, and outliers. Each step is carefully chosen to enhance the quality of the data and, ultimately, the performance of any predictive model built upon it.

## The Data Preprocessing Pipeline

The entire process is performed in a logical sequence to ensure data integrity at each step.

### 1. Data Loading and Exploration
* The script begins by loading the `Titanic-Dataset.csv` into a Pandas DataFrame.
* Initial exploration is done using `.info()`, `.head()`, and `.isnull().sum()` to understand the dataset's structure, identify data types, and locate columns with missing values.

### 2. Handling Missing Values
To create a complete dataset, missing values were handled with the following strategies:
* **Age:** Missing age values were imputed using the **median** age. The median is chosen over the mean because it is robust to outliers.
* **Embarked:** The few missing 'Embarked' values were filled with the **mode** (the most frequent port of embarkation).
* **Cabin:** The 'Cabin' column was **dropped** entirely as it contained too many missing values to be useful.

### 3. Feature Engineering and Selection
To improve the model's focus on relevant information, the following columns were removed:
* **'PassengerId', 'Name', 'Ticket':** These are unique identifiers with high cardinality and provide no generalizable predictive power. Including them could lead to overfitting.

### 4. Encoding Categorical Features
Machine learning models require numerical input, so categorical features were converted:
* **'Sex' and 'Embarked':** These nominal features were converted into numerical format using **One-Hot Encoding**. This method creates new binary (0/1) columns, avoiding the artificial ordinal relationship that Label Encoding would introduce.

### 5. Feature Scaling
Numerical features were scaled to ensure that no single feature dominates the model due to its scale:
* **'Age', 'Fare', 'Pclass', 'SibSp', 'Parch':** These columns were standardized using `StandardScaler` from Scikit-learn. This process transforms the data to have a mean of 0 and a standard deviation of 1.

### 6. Outlier Handling
Finally, outliers were addressed to prevent them from skewing the model:
* A **boxplot** was used to visualize the distribution of the 'Fare' column, which clearly showed the presence of extreme outliers.
* These outliers were removed using the **Interquartile Range (IQR)** method, resulting in a cleaner, more representative dataset.

## Tools and Libraries Used

* **Python**
* **Pandas:** For data manipulation and analysis.
* **NumPy:** For numerical operations.
* **Scikit-learn:** For encoding (`OneHotEncoder`) and scaling (`StandardScaler`).
* **Matplotlib & Seaborn:** For data visualization, particularly for creating the boxplots.

