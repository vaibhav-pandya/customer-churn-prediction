# Customer Churn Prediction 📉💡
###### URL: https://hub.docker.com/repository/docker/vaibhavpandya/customerchurndetection/general

This project aims to predict customer churn using machine learning techniques. By analyzing customer behavior, demographics, and service usage patterns, the goal is to identify key factors influencing churn and provide actionable insights for businesses to enhance customer retention.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Data Description](#data-description)
4. [Approach](#approach)
5. [Technologies Used](#technologies-used)
6. [Installation and Usage](#installation-and-usage)
7. [Results and Insights](#results-and-insights)
8. [Future Scope](#future-scope)
9. [Contact](#contact)

---

## Introduction
Customer churn is a critical concern for businesses, particularly in subscription-based services. This project leverages machine learning models to predict whether a customer is likely to leave, allowing companies to take proactive measures to improve retention.

---

## Problem Statement
Predict customer churn based on demographic, usage, and service-related data. The model should identify significant factors contributing to churn and help businesses devise strategies to reduce attrition.

---

## Data Description
The dataset used for this project includes:
- **Customer Demographics**: Age, gender, etc.
- **Service Usage**: Monthly charges, contract type, tenure.
- **Account Information**: Payment method, number of services subscribed.
- **Churn Label**: Whether a customer churned (Yes/No).

**Shape**: (7043, 21)<br>
**Target Variable**: `Churn` (Binary: Yes/No)

---

## Approach
1. **Exploratory Data Analysis (EDA)**:
   - Univariate, Bivariate, and Multivariate analysis.
   - Handling missing data, outliers, and skewness.

2. **Feature Engineering**:
   - Encoding categorical variables.
   - Scaling numerical features.
   - Creating new derived features.

3. **Model Development**:
   - Experimented with models such as:
     - XGBoost
   - Evaluated using metrics like rsme, mae, r2

---

## Technologies Used
- **Programming Language**: Python 🐍
- **Libraries**:
  - Pandas, NumPy (Data preprocessing)
  - Matplotlib, Seaborn (Visualization)
  - Scikit-learn, XGBoost (Machine Learning)
  - Jupyter Notebook (Development Environment)
  - Docker (Containerization)
  - DVC (Version Control for Machine Learning)

---

## Installation and Usage
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the model**:
   ```bash
   python train.py
   ```

4. **Deploy the model** (*Optional*):
   ```bash
   python app.py
   ```

---

## Results and Insights
- Identified key factors influencing churn:
  - **Tenure**: Customers with shorter tenures are more likely to churn.
  - **Contract Type**: Month-to-month customers have a higher churn rate.
  - **Monthly Charges**: Higher charges correlate with increased churn risk.
- Visualized customer segments to target retention strategies.

---

## Future Scope
- Implement real-time churn prediction with streaming data.
- Deploy a web-based dashboard for business stakeholders.
- Expand analysis to include customer sentiment from reviews or support tickets.

---

## Contact
For any queries or feedback, please reach out to -<br>
Email: vaibhavpandya2903@gmail.com<br>
[LinkedIn](https://www.linkedin.com/in/vaibhavpandya2903/)

