# CodeAlpha_tasks
CodeAlpha Project
# CodeAlpha_unemployment_analysis

SALES PREDICTION USING PYTHON

Project Overview

=> This project predicts product sales based on advertising expenditures across different media platforms such as TV, Radio, and Newspaper.
=> Using Machine Learning techniques, the project analyzes the relationship between advertising budgets and sales performance and forecasts future sales.

Objective

- Analyze the impact of advertising spending on product sales.
- Perform data cleaning and preprocessing.
- Visualize relationships between advertising channels and sales.
- Build a Linear Regression model for sales prediction.
- Evaluate model performance using standard regression metrics.
- Predict future sales based on advertising budgets.

Dataset

Dataset Name: Advertising_dataset.csv

Features

- TV – Advertising budget spent on TV advertisements.
- Radio – Advertising budget spent on Radio advertisements.
- Newspaper – Advertising budget spent on Newspaper advertisements.
- Sales – Product sales (Target Variable).
  
Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
  
Role of Libraries in the Project

Reading Advertising.csv dataset => (Pandas)

Data cleaning and preprocessing => (Pandas)

Statistical analysis => (Pandas + NumPy)

Feature selection => (Pandas)

Train-Test splitting => (Scikit-learn)

Linear Regression model training => (Scikit-learn)

Sales prediction => (Scikit-learn)

Model evaluation (MAE, MSE, RMSE, R² Score) => (Scikit-learn + NumPy)

Correlation heatmap generation => (Seaborn + Matplotlib)

Pairplot visualization => (Seaborn)

Advertising impact analysis => (Pandas + Seaborn)

Actual vs Predicted Sales visualization => (Matplotlib)

Graph generation and saving => (Matplotlib + Seaborn)

Future sales forecasting => (Scikit-learn)

Exporting prediction results to CSV => (Pandas)


Installation

Install the required libraries using:

pip install -r requirements.txt


Project Workflow

1. Load the dataset.
2. Clean and preprocess the data.
3. Perform Exploratory Data Analysis (EDA).
4. Generate visualizations.
5. Select features and target variable.
6. Split data into training and testing sets.
7. Train the Linear Regression model.
8. Evaluate model performance.
9. Predict future sales.
10. Save predictions and graphs.
    

Machine Learning Model

Algorithm Used: Linear Regression

Linear Regression is used to model the relationship between advertising expenditure and sales, enabling accurate sales forecasting.

Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

Generated Outputs

Graphs

- Correlation Heatmap
- Pairplot
- TV vs Sales
- Radio vs Sales
- Newspaper vs Sales
- Sales Distribution
- Actual vs Predicted Sales
- Feature Importance

Files Generated

- sales_predictions.csv
- correlation_heatmap.png
- pairplot.png
- tv_vs_sales.png
- radio_vs_sales.png
- newspaper_vs_sales.png
- sales_distribution.png
- actual_vs_predicted.png
- feature_importance.png

Project Structure

Sales-Prediction/
│
├── Advertising_dataset.csv
├── sales_prediction.py
├── requirements.txt
├── sales_predictions.csv
├── README.md
│
└── images/
    ├── correlation_heatmap.png
    ├── pairplot.png
    ├── tv_vs_sales.png
    ├── radio_vs_sales.png
    ├── newspaper_vs_sales.png
    ├── sales_distribution.png
    ├── actual_vs_predicted.png
    └── feature_importance.png

Conclusion

=> The project demonstrates how advertising expenditures influence product sales. 
=> The Linear Regression model successfully predicts sales using advertising budgets  
=> It provides insights that can help businesses optimize marketing strategies and improve decision-making.
