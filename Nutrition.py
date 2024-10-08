#Creating EDA for the given Mcdonalds Data Set Menu for nutrition analysis of every menu item, including salads, beverages, and desserts
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#load the menu.csv file by creating a dataframe as df
df=pd.read_csv('menu.csv')

print(df.head())# first 5 rows and columns to display
print(df.info())
print(df.describe())#summarize the dataset

# Check for non-numeric values in the dataset
df.dtypes

# Convert relevant columns to numeric, coercing errors into NaN
#df['Category'] = pd.to_numeric(df['Category'], errors='coerce')
#df['Item'] = pd.to_numeric(df['Item'], errors='coerce')

# Option 1: Drop rows with NaN values
#df = df.dropna()

# Option 2: Fill NaN values with a default value (e.g., 0)
#df = df.fillna(0)

#Categorize the items based on the sugar and fiber contents in low ,moderate and high
def categorize_sugar(sugar):
    if sugar<5:
        return 'Low Sugar'
    elif 5 <=sugar< 20:
        return 'Moderate Sugar'
    else:
        return 'High Sugar'
def categorize_fiber(fiber):
    if fiber < 2:
        return 'Low Fiber'
    elif 2<=fiber<5:
        return 'Moderate Fiber'
    else:
        return 'High Fiber'
    
#create new columns with above functions 
df['Sugar Category']= df['Sugars'].apply(categorize_sugar)
df['Fiber Category'] = df['Dietary Fiber'].apply(categorize_fiber)   

# Preview the modified columns in dataframe
df[['Item', 'Sugars', 'Sugar Category', 'Dietary Fiber', 'Fiber Category']].head()

# Barchart for Sugars and Fiber Categories 
plt.figure(figsize=(10,5)) #creates a new figure 
sns.countplot(data=df,x='Sugar Category')
plt.title('Count of Menu items by Sugar Category')
plt.show()

plt.figure(figsize=(5,10))
sns.countplot(data=df,x='Fiber Category')
plt.title('Count of Menu items by Fiber Category')
plt.show()

#Pie charts for Sugar and Fiber Categories
plt.figure(figsize=(7,7))
df['Sugar Category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='coolwarm')
plt.title('Distribution of Menu Items by Sugar Content')
plt.show()

plt.figure(figsize=(7,7))
df['Fiber Category'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, cmap='coolwarm') #check again
plt.title('Distribution of Menu Items by Fiber Content')
plt.show()


#Scatterplot for Sugar and Fiber Content
plt.figure(figsize=(10,6))
sns.scatterplot(x='Sugars',y='Dietary Fiber',data=df,hue='Sugar Category',palette='coolwarm')
plt.title("Scatter plot of Sugar vs Fiber")
plt.show()

#Heatmap for correlations
#plt.figure(figsize=(12,8))
#sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
#plt.title(" Correlation HeatMap of Nutrition analysis")
#plt.show()
