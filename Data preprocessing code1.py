# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 12:20:07 2025

@author: sansk
"""
#Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from feature_engine.outliers import Winsorizer
from feature_engine.imputation import RandomSampleImputer
from sklearn.preprocessing import PowerTransformer
from feature_engine import transformation
from scipy import stats


df=pd.read_excel(r"C:\Users\hp\Downloads\project dataset\Medical Inventory Optimaization Dataset project.xlsx")
df.info()


# Convert Patient_ID columns to string (Nominal)
df['Dateofbill'] = pd.to_datetime(df['Dateofbill'], errors='coerce')
df.dtypes


# 2. FIND & REMOVE DUPLICATES
duplicate_count=df.duplicated().sum()
print("Duplicate Rows:",duplicate_count)
df=df.drop_duplicates()

df.size

# 3. IDENTIFY NULL VALUES & TREATMENT (MISSING VALUE IMPUTATION)
df.isnull().sum()
cols=['Formulation','DrugName','SubCat','SubCat1']
# Mode for categorical columns
imputer=SimpleImputer(strategy='most_frequent')
df[cols]=imputer.fit_transform(df[cols])
df.isnull().sum()



# 4. IDENTIFY OUTLIER & TREATMENT (Quantity, ReturnQuantity, Final_Cost, Final_Sales, RtnMRP)

# Boxplots before treatment

numeric_cols=['Quantity','ReturnQuantity','Final_Cost','Final_Sales','RtnMRP']
for col in numeric_cols:

        plt.boxplot(df[col])
        plt.title(col)
        plt.show()


# Winsorization (IQR method to treat outliers)
num_cols= ['Quantity','Final_Cost','Final_Sales']
winsor_iqr=Winsorizer(
    capping_method='iqr',
    tail='both',
    fold=1.5,
    variables=num_cols
    )
df[num_cols]=winsor_iqr.fit_transform(df[num_cols])

# Winsorization (Gaussian method)
winsor_gauss=Winsorizer(
    capping_method='gaussian',
    tail='both',
    fold=2,
    variables=['ReturnQuantity','RtnMRP'])
df[['ReturnQuantity','RtnMRP']] = winsor_gauss.fit_transform(df[['ReturnQuantity','RtnMRP']])

# Boxplots after winsorization
all_cols = ['Quantity','ReturnQuantity','Final_Cost','Final_Sales','RtnMRP']
for col in all_cols:
    try:
        sns.boxplot(df[col])
        plt.title(col)
        plt.show()
    except:
        pass

# Subplots with Seaborn all plots one scale    
plt.figure(figsize=(12,8))

for i, col in enumerate(all_cols,1):
    plt.subplot(3, 2, i)
    sns.boxplot(df[col])
    plt.title(col)
    
plt.tight_layout()
plt.show()

# 5. DISCRETIZATION
#Quantity
df['Demand_Level']=pd.cut(
                    df['Quantity'],
                    bins=[-1,1,3,float('inf')],
                    labels=['Rarely Used', 'Moderate Demand', 'High Demand']
                    )
df['Demand_Level'].value_counts().sort_index()

#Returnquantity
df['Return_Severity']=pd.cut(
                      df['ReturnQuantity'],
                      bins=[-1, 0, 2, df['ReturnQuantity'].max()],
                      labels=['No Return', 'Low Return', 'High Return']
                      )
df['Return_Severity'].value_counts()
df['ReturnQuantity'].value_counts()


df["Final_Cost"].min()


#Final_Cost
df['Cost_Band'] = pd.cut(
               df['Final_Cost'],
               bins=[0,45,55,80,float('inf')],
               labels=['Essential Low Cost', 'Affordable', 'Standard Branded', 'High Value'])

df['Cost_Band'].value_counts()

#Final_Sales
df['Final_Sales'].describe(percentiles=[.10, .25, .50, .75, .90, .95])

df['Sales_Band'] = pd.qcut(
                   df['Final_Sales'],
                   q=4,
                   labels=['Low Sales', 'Moderate Sales', 'High Sales', 'Very High Sales']
                   )
df['Sales_Band'].value_counts()

#RtnMRP
df['MRP_Category'] = pd.cut(
                     df['RtnMRP'],
                     bins=[-1, 0, 50, 150, float('inf')],
                     labels=['Zero MRP', 'Low MRP', 'Moderate MRP', 'High MRP']
                     )
df['MRP_Category'].value_counts().sort_index()

# Profitability Category -- 1st create profit column
df['Profit'] = df['Final_Sales'] - df['Final_Cost']

df['Profit'].describe(percentiles=[.10,.25,.50,.75,.90])

df['Profit_Category'] = pd.cut(
                        df['Profit'],
                        bins=[-9999999, 0, 50, 150, float('inf')],
                        labels=['Loss', 'Low Margin', 'Medium Margin', 'High Margin']
                        )
df['Profit_Category'].value_counts()


# 6. ONE-HOT ENCODING(convert categorial data into 0,1 form)
cols_to_encode = ['Specialisation','Dept','Formulation','DrugName','SubCat','SubCat1']
for col in cols_to_encode:
    le=LabelEncoder()
    df[col + '_label']=le.fit_transform(df[col].astype(str))


#7. Q-Q PLOTS + TRANSFORMATIONS
cols = ['Quantity', 'ReturnQuantity', 'Final_Cost', 'Final_Sales', 'RtnMRP']

plt.figure(figsize=(14, 12))

for i, col in enumerate(cols, 1):
    plt.subplot(3, 2, i)  # 3 rows, 2 columns
    stats.probplot(df[col], dist='norm', plot=plt)
    plt.title(f"QQ Plot for {col}", fontsize=12)

plt.tight_layout()
plt.show()

# Yeo-Johnson transformation 
pt = PowerTransformer(method='yeo-johnson')

df[['Quantity_pt','ReturnQuantity_pt','Final_Cost_pt','Final_Sales_pt','RtnMRP_pt']] = pt.fit_transform(df[numeric_cols])



# QQ PLOTS after Transformation
pt_cols = ['Quantity_pt','ReturnQuantity_pt','Final_Cost_pt','Final_Sales_pt','RtnMRP_pt']

plt.figure(figsize=(14,12))

for i, col in enumerate(pt_cols, 1):
    plt.subplot(3,2,i)
    stats.probplot(df[col], dist='norm', plot=plt)
    plt.title(f"QQ Plot After Yeo-Johnson: {col}", fontsize=12)

plt.tight_layout()
plt.show()


# 8. STANDARDIZATION & SCALING

numeric_cols = ['Quantity', 'ReturnQuantity', 'Final_Cost', 'Final_Sales', 'RtnMRP']

scaler = StandardScaler()

df_standardized = df.copy()
df_standardized[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# OUTPUT
columns_to_remove = ['Specialisation_label'	,'Dept_label',	'Formulation_label',	'DrugName_label',	'SubCat_label',	'SubCat1_label'
                     'Demand_Level','Return_Severity','Cost_Band','Sales_Band','MRP_Category','Profit_Category','Quantity_pt','ReturnQuantity_pt',
                     'Final_Cost_pt','Final_Sales_pt','RtnMRP_pt','Demand_Level','SubCat1_label'

]


cleaned_data = df.drop(columns=columns_to_remove, errors='ignore')


output_path =(r"C:/Users/hp/Downloads/project all files/Medical Inventory Optimaization Dataset cleaned data.xlsx")
cleaned_data.to_excel(output_path, index=False)


print(f"\n Clean Data Saved Successfully to → {output_path}")






















