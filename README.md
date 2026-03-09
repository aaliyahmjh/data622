# Final Project Proposal

Predicting NYC Public Employee Salaries Using Machine Learning

## Project Overview

This proposal outlines the development of a machine learning-powered web application designed to predict salaries for New York City public employees. Leveraging a decade of publicly available payroll data, the application will serve as a transparency tool for job seekers, current employees, researchers, and policymakers. The project will involve training regression models to estimate annual salaries based on job characteristics and tenure, and deploying these models within an interactive Shiny dashboard.

## Problem Statement

This project uses the NYC Open Data Citywide Payroll dataset (2015–2024), which contains millions of payroll records across more than 170 city agencies, to build a machine learning model that predicts employee salaries based on factors such as job title, agency, and years of service. The resulting application will allow users to explore workforce trends and estimate expected salaries for public sector roles.

## Target Users & Features

The primary users of this application will include:
- Job Seekers / Career Changers: Need to understand fair compensation before applying.
  - Feature: Salary Prediction Tool – Input a desired job title and expected experience to receive a predicted salary range.
- Current Public Employees: Want to benchmark their own compensation against peers.
  - Feature: Salary Benchmarking – Compare an individual's salary against the model's prediction for similar roles, highlighting potential disparities.
- Researchers: Require trends and statistical summaries.
  - Feature: Workforce Insights Dashboard – showing salary distributions, agency comparisons, and pay trends over time.
- Policymakers & Agency Administrators: Need data for workforce planning and equity analysis.
  - Feature: Disparity Analysis View – Aggregate views of pay gaps across demographic or departmental lines 

## App Features

The application will include three core components:
1. Salary Prediction Tool
Users can input characteristics such as job title, years of service, agency, and then the app will return a predicted salary range using the trained machine learning model.
2. Salary Benchmarking
Users can compare predicted compensation against historical data for similar positions.
3. Workforce Insights Dashboard
Interactive visualizations will display patterns such as salary distribution across agencies, compensation growth over time & relationships between tenure and salary

## Interface

The application will be developed using **Shiny App**.

## Data Sources

This project will use the NYC Open Data Citywide Payroll dataset ( publicly available through NYC Open Data), which includes more than 6.8 million payroll records, data from 2015 through 2024, information across approximately 170 city agencies, over 2,000 job titles and compensation, overtime, and workforce characteristics.

https://data.cityofnewyork.us/City-Government/Citywide-Payroll-Data-Fiscal-Year-/k397-673e/about_data

## Data Preparation

Initial data cleaning has already been performed, including the removal of duplicate records, standardization of job titles, creation of tenure variables and filtering to focus on salaried employees
Further preprocessing will include:
- encoding categorical variables (job titles, agencies)
- feature engineering for tenure and job categories
- removal of extreme outliers

## Machine Learning Problems

The primary machine learning task in this project is regression, where we will build a model to predict annual salary based on employee characteristics such as job title, agency, and years of service. 
We will begin with Linear Regression as a baseline model to establish a performance benchmark. More advanced tree-based models, including Random Forest Regression and Gradient Boosting (XGBoost or LightGBM), will then be explored to capture potential nonlinear relationships and improve prediction accuracy.

## Model Performance

Regression models will be evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) & R² (coefficient of determination)
The dataset will be split into training and testing sets, and cross-validation will be used to ensure reliable performance estimates.

## App Performance
- Prediction Latency: The time taken to generate a salary prediction after user input (target: < 2 seconds).
- Usability: Clarity of instructions, intuitive layout, and interpretability of outputs 
- Visualization Clarity: Effectiveness of charts in conveying insights to non-technical users.
Application performance will be evaluated based on prediction response time, clarity of outputs & usability of interactive features. The goal is to ensure that users can quickly generate salary predictions and interpret results easily.

## Model Training Strategy
Model training will follow a **batch training approach**. The model will be trained on historical payroll data and updated periodically when new fiscal year payroll data becomes available. This approach is appropriate because payroll data is updated annually rather than in real time.

## Computational Requirements

The project does not require a large computational infrastructure. Although the dataset is relatively large, it can be processed using standard machine learning tools in R or Python without the need for GPUs or large neural networks.
The code and documentation will be maintained in a GitHub repository for reproducibility.

## Minimally Viable Product (Midterm Check)

The MVP will include:
### Exploratory Data Analysis
- summary statistics
- salary distribution visualizations
- tenure vs compensation analysis

### Data Preprocessing
- cleaned modeling dataset
- engineered features for job categories and tenure

### Baseline Model
- at least one trained regression model
- evaluation metrics reported

### Initial Shiny Interface
- user inputs for job title and years of service
- predicted salary output
- simple supporting visualization
