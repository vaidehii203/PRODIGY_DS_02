# PRODIGY_DS_02
Data cleaning and Exploratory data analysis on Titanic dataset. 
Dataset link:- https://www.kaggle.com/c/titanic/data


## Project Overview

The main objectives of this project are:
- To perform data cleaning by handling missing values and converting categorical variables.
- To conduct exploratory data analysis (EDA) to uncover patterns and trends in the dataset.
- To visualize distributions and survival rates across various features.

## Key Steps

1. **Data Cleaning**
    - Handling missing values for age and embarked columns.
    - Converting categorical variables (sex and embarked) to numerical values.
    - Dropping irrelevant columns such as cabin, name, and ticket.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset
train_url = 'D:\\intership tasks\\train.csv'
train_df = pd.read_csv(train_url)

def clean_data(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=['Cabin','Name','Ticket'], inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    return df

# Clean the training dataset
train_df = clean_data(train_df)

# Check the cleaned training data
train_df.head()
```

2. **Exploratory Data Analysis (EDA)**
    - Visualizing the distribution of survival, age, passenger class, sex, and embarkation ports.
    - Analyzing survival rates by sex, passenger class, and embarkation ports.
    - Generating a correlation heatmap to understand relationships between variables.

```python
# Visualization and EDA

# Distribution of Survival
sns.countplot(x='Survived', data=train_df)
plt.title('Distribution of Survival')
plt.show()

# Distribution of Age
sns.histplot(train_df['Age'], bins=30, kde=True)
plt.title('Distribution of Age')
plt.show()

# Distribution of Passenger Class
sns.countplot(x='Pclass', data=train_df)
plt.title('Distribution of Passenger Class')
plt.show()

# Distribution of Sex
sns.countplot(x='Sex', data=train_df)
plt.title('Distribution of Sex')
plt.show()

# Distribution of Embarkation Ports
sns.countplot(x='Embarked', data=train_df)
plt.title('Distribution of Embarkation Ports')
plt.show()

# Survival Rate by Sex
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Sex')
plt.show()

# Survival Rate by Passenger Class
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Survival Rate by Embarkation Port
sns.barplot(x='Embarked', y='Survived', data=train_df)
plt.title('Survival Rate by Embarkation Port')
plt.show()

# Survival Rate by Age
plt.figure(figsize=(10, 6))
sns.histplot(train_df[train_df['Survived'] == 1]['Age'], bins=30, kde=True, color='green', label='Survived')
sns.histplot(train_df[train_df['Survived'] == 0]['Age'], bins=30, kde=True, color='red', label='Not Survived')
plt.title('Survival Rate by Age')
plt.legend()
plt.show()

# Pairplot of important features
sns.pairplot(train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']], hue='Survived')
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(train_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()
```

## Conclusion

This project provides a comprehensive analysis of the Titanic dataset, highlighting key trends and patterns that can be used for further predictive modeling.

## Contact

Feel free to reach out if you have any questions or suggestions!

LinkedIn: https://www.linkedin.com/in/vaidehi-kale-b635b7264/
