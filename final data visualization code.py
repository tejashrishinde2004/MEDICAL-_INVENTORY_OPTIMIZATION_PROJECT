# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 21:41:21 2025

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_excel(r"C:\Users\hp\Downloads\project all files\Medical Inventory Optimaization Dataset cleaned data.xlsx")
df.head()

df.columns.to_list()

######################Univariate Analysis##########
#################Histogram##############

columns_hist=df.select_dtypes(include=['object','number']).columns.difference(['Dateofbill'])
for i in columns_hist:
    plt.hist(df[i],color='red')
    plt.title(f"{i}")
    plt.show()

##Pie Chart for categorical
column_pie_categorical=df.select_dtypes(include=["object"]).columns.difference(['Dateofbill'])
print(column_pie_categorical)

for i in column_pie_categorical:
    cnts=df[i].value_counts()
    plt.pie(cnts.values,labels=cnts.index,autopct='%1.1f%%')
    plt.show()
    
# Analysis for dateofbill column
#histogram and pie
###since it is in datetime so we will extract month

df["month"]=df["Dateofbill"].dt.month
df[['month']]

df["month_name"]=df["Dateofbill"].dt.month_name()
df[["month_name"]]

plt.hist(df["month_name"],color="hotpink")
plt.show()

month_counts=df["month_name"].value_counts()
plt.pie(month_counts,labels=month_counts.index,autopct="%1.1f%%")
plt.show()

##################Bivariate##################
###################Bar chart###############

plt.bar(df["SubCat"],df["Quantity"],color='orange',width=0.4)
plt.xlabel("Sub Categories")
plt.xticks(rotation=90)
plt.ylabel("Quantity")

plt.bar(df["SubCat1"],df["Quantity"],color='orange',width=0.4)
plt.xlabel("Sub Categories")
plt.xticks(rotation=90)
plt.ylabel("Quantity")
df.head()
df["DrugName"].nunique()
df["Profit"]=df['Final_Sales']-df['Final_Cost']

df.head()

plt.bar(df['month_name'],df["Profit"],color='orange',width=0.8)
plt.xlabel("Months")
plt.xticks(rotation=90)
plt.ylabel("Profits")

colors=['red' if p <0 else 'orange' for p in df["Profit"]]
plt.bar(df['month_name'],df["Profit"],color=colors,width=0.8)
plt.xlabel("Months")
plt.xticks(rotation=90)
plt.ylabel("Profits")
plt.axhline(0)
plt.show()

plt.bar(df['SubCat'],df['RtnMRP'],color='red',width=0.4)
plt.xlabel("Sub Category")
plt.xticks(rotation=90)
plt.ylabel("Rtnmrp")
plt.show()

#################Bivariate Analysis####################
#######################Scatter plot####################

df.head()

sns.scatterplot(x=df["Quantity"],y=df["ReturnQuantity"])
plt.xlabel('Quantity')
plt.ylabel('Return Quantity')
plt.axhline(0)
plt.title(" Quantity vs Return Quantity")
plt.show()

sns.scatterplot(x=df["Quantity"],y=df["Final_Cost"])
plt.xlabel('Quantity')
plt.ylabel('Final Cost')
plt.axhline(0)
plt.title(" Quantity vs Final Cost")
plt.show()

sns.scatterplot(x=df["Quantity"],y=df["Final_Sales"])
plt.xlabel('Quantity')
plt.ylabel('Final Sales')
plt.axhline(0)
plt.title(" Quantity vs Final Sales")
plt.show()

sns.scatterplot(x=df["Quantity"],y=df["RtnMRP"])
plt.xlabel('Quantity')
plt.ylabel('Return Mrp')
plt.axhline(0)
plt.title(" Quantity vs Return Mrp")
plt.show()

df.columns

sns.scatterplot(x=df['ReturnQuantity'],y=df['Final_Cost'])
plt.xlabel('Return Qty')
plt.ylabel('Final Cost')
plt.show()


plt.subplot(1,2,1)
sns.scatterplot(x=df["ReturnQuantity"],y=df['Final_Sales'])
plt.xlabel("Return Quantity")
plt.ylabel('Final Sales')

plt.subplot(1,2,2)
sns.scatterplot(x=df["ReturnQuantity"],y=df['RtnMRP'])
plt.xlabel("Return Quantity")
plt.ylabel('Return Mrp')

plt.show()

df.columns

fig,axes=plt.subplots(1,2,figsize=(10,4))
sns.scatterplot(x='Final_Cost',y='Final_Sales',data=df,ax=axes[0])
axes[0].set_title('Final Cost vs Final Slaes')

sns.scatterplot(x='Final_Cost',y='RtnMRP',data=df,ax=axes[1])
axes[1].set_title('Final Cost vs RtnMRP')

plt.tight_layout()
plt.show()

sns.scatterplot(x=df['Final_Sales'],y=df['RtnMRP'])
plt.xlabel("Final Sale")
plt.ylabel("RtnMRP")
plt.title("Final Sales vs Return Mrp")
plt.show()

########################Multivariate Analysis#######################
df.columns
sns.heatmap(df[['Quantity','Final_Cost','Final_Sales','Profit']].corr(), annot=True)

# ------- PAIRPLOT -------
selected_cols = ['Quantity','Final_Cost','Final_Sales','Profit']
sns.pairplot(df[selected_cols].dropna(), diag_kind="kde")
plt.suptitle("Pairplot for Key Variables", y=1.02)
plt.show()

