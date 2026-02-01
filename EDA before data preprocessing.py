# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:41:32 2025

@author: tejashrii
"""

import pandas as pd 
medical_dataset=pd.read_excel(r"C:\Users\hp\Downloads\Medical Inventory Optimaization Dataset.xlsx")
medical_dataset.dtypes
medical_dataset.Dateofbill = medical_dataset.Dateofbill.astype('datetime64[ns]')


#first moment of business decision
columns_to_calculate=['Quantity','ReturnQuantity','Final_Cost','Final_Sales','RtnMRP']
mean_values=medical_dataset[columns_to_calculate].mean()
print(mean_values)

median_values=medical_dataset[columns_to_calculate].median()
print(median_values)

columns_to_calculate_mode = ['Typeofsales','Formulation','DrugName','SubCat','SubCat1']
mode_values=medical_dataset[columns_to_calculate_mode].mode()
print(mode_values)


#Second Moment Of Business Desicion 
columns_to_calculate_var_std = ['Quantity','ReturnQuantity','Final_Cost','Final_Sales','RtnMRP']
variance_values =medical_dataset[columns_to_calculate_var_std].var()
print(variance_values)
std_values=medical_dataset[columns_to_calculate_var_std].std()
print(std_values)


range_values={}
for col in columns_to_calculate:
    column_data=medical_dataset[columns_to_calculate]
    column_range=column_data.max()-column_data.min()
    range_values=column_range
print(range_values)
    
    

#Third Moment Of Business Desicion
skewness_values=medical_dataset[columns_to_calculate].skew()
print(skewness_values) 


#Fourth Moment Of Business Desicion 
kurtosis_values=medical_dataset[columns_to_calculate].kurt()
print(kurtosis_values)




#using for loop
numerical_columns=medical_dataset.select_dtypes(include=['int64','float64']).columns
for col in numerical_columns:
    print(f"mean_values : {medical_dataset[col].mean()}")
    print(f"median_values : {medical_dataset[col].median()}")
    print(f"var_values : {medical_dataset[col].var()}")
    print(f"std_values : {medical_dataset[col].std()}")
    print(f"skew_values : {medical_dataset[col].skew()}")
    print(f"kurt_values : {medical_dataset[col].kurt()}")
    col_range = medical_dataset[col].max() - medical_dataset[col].min()
    print(col_range)



    
