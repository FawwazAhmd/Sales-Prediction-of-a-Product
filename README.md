# Sales Prediction Model

## Overview

This project implements a machine learning model to predict sales for a retail dataset using XGBoost. The model preprocesses the data, performs exploratory data analysis (EDA), trains an XGBoost regressor, and evaluates its performance.

## Contents

- **Data Preprocessing**: Handles missing values and encodes categorical features.
- **Exploratory Data Analysis (EDA)**: Visualizes various distributions and relationships in the dataset.
- **Model Training**: Trains an XGBoost regressor on the processed data.
- **Performance Evaluation**: Evaluates the model using various metrics.

## Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

You can install the necessary packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```
# Data

The data file Train(1).csv should be placed in the C:\Users\fnn09\Downloads\ directory or updated in the script path accordingly.

## Script Breakdown 

  *1. Data Loading and Preprocessing:*
        Reads the dataset.
        Fills missing values in Item_Weight with the mean.
        Fills missing values in Outlet_Size using mode based on Outlet_Type.
        Encodes categorical features using label encoding.

  *2. Exploratory Data Analysis (EDA):*
        Plots distributions for Item_Weight, Item_Visibility, Item_MRP, Item_Outlet_Sales.
        Plots counts for Outlet_Establishment_Year, Item_Fat_Content, Item_Type, and Outlet_Size.

  *3. Model Training:*
        Splits the data into training and testing sets.
        Trains an XGBoost regressor on the training data.

  *4. Model Evaluation:*
        Calculates and prints Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared value.
        Predicts sales for a sample input data.

# Usage

To run the script, ensure all required libraries are installed and the dataset is in the specified directory. Execute the script in a Python environment:

bash
```
python sales_prediction.py
```
## Example Output

plaintext
```
Mean Absolute Error (MAE): <value>
Mean Squared Error (MSE): <value>
Root Mean Squared Error (RMSE): <value>
R Squared value = <value>
The sales for the first product which is FDA15 in the dataset is predicted as <value>
```
# License

This project is licensed under the MIT License. See the LICENSE file for details.

vbnet
```

You can replace `<value>` with the actual values from your script's output. Let me know if y
