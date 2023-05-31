!pip install seaborn
!pip install sklearn


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import sklearn
from sklearn.linear_model import LinearRegression

def load_dataset():
    return pd.read_csv("boston.csv")

boston_dataframe = load_dataset()
  
def print_summarize_dataset(dataset):
    print("Dataset dimension:")
    print(boston_dataframe.shape)
    print("First 10 rows of dataset:")
    print(boston_dataframe.head(10))
    print("Statistical summary:")
    print(boston_dataframe.describe)

print_summarize_dataset(boston_dataframe)

    
def clean_dataset(boston_dataframe):
    return boston_dataframe.dropna()

clean_dataset = clean_dataset(boston_dataframe)

def print_histograms(boston_dataframe):
    boston_dataframe.hist(figsize=(20, 16))
    plt.show()

print_histograms(boston_dataframe)
    
def compute_correlations_matrix(dataset):
    correlation = dataset.corr()
    plt.figure(figsize=(15,15))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap="Greens")
    plt.show()
    return correlation
  
def compute_correlations_matrix(boston_dataframe):
    data=boston_dataframe.corr()
    colormap = sns.color_palette("PuBuGn")
    plt.figure(figsize=(25,13))
    ax = sns.heatmap(data, cmap=colormap, annot=True)
    sns.heatmap(data)
    plt.show()
    
compute_correlations_matrix(clean_dataset)

def print_scatter_matrix(boston_dataframe):
    sns.pairplot(boston_dataframe[["CRIM","NOX","RM","LSTAT","MDEV"]])
    plt.show()
    
print_scatter_matrix(boston_dataframe)


  
def print_scatter_matrix(boston_dataframe):
    scatter_matrix(boston_dataframe, figsize = (20,20))
    plt.show()
    
    mdev_lstat_corr = round(boston_dataframe['LSTAT'].corr(boston_dataframe['MDEV']), 3)
    plt.figure(figsize=(6, 3))
    plt.scatter(x=boston_dataframe['LSTAT'], y=boston_dataframe['MDEV'], alpha=0.6, s=80, color='red')
    plt.title(f'MDEV vs LSTAT (Correlation {mdev_lstat_corr})', fontsize=14)
    plt.xlabel('LSTAT', fontsize=14)
    plt.ylabel('MDEV', fontsize=14)
    plt.show()
    
    mdev_age_corr = round(boston_dataframe['AGE'].corr(boston_dataframe['MDEV']), 3)
    plt.figure(figsize=(6, 3))
    plt.scatter(x=boston_dataframe['AGE'], y=boston_dataframe['MDEV'], alpha=0.6, s=80, color='blue')
    plt.title(f'MDEV vs AGE (Correlation {mdev_age_corr})', fontsize=14)
    plt.xlabel('AGE', fontsize=14)
    plt.ylabel('MDEV', fontsize=14)
    plt.show()
    
    mdev_crim_corr = round(boston_dataframe['CRIM'].corr(boston_dataframe['MDEV']), 3)
    plt.figure(figsize=(6, 3))
    plt.scatter(x=boston_dataframe['CRIM'], y=boston_dataframe['MDEV'], alpha=0.6, s=80, color='green')
    plt.title(f'MDEV vs CRIM (Correlation {mdev_crim_corr})', fontsize=14)
    plt.xlabel('CRIM', fontsize=14)
    plt.ylabel('MDEV', fontsize=14)
    plt.show()
    
print_scatter_matrix(clean_dataset)

def boston_fit_model(boston_dataframe):
    # SELECT two columns from our 
    model_dataset = boston_dataframe[["RM","MDEV"]]
    regressor = sklearn.linear_model.LinearRegression()
    # Extract column 1
    x = model_dataset.iloc[:, :-1].values
    # Extract column 2
    y = model_dataset.iloc[:, 1].values
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=2)
    # Train the model
    regressor.fit(x_train, y_train)
    return regressor
  
print(boston_fit_model(boston_dataframe=boston_dataframe))
  
def boston_predict(estimator, array_to_predict):
  result = estimator.predict(array_to_predict)
  return result

def compute_correlations_matrix(dataset):
    correlation = dataset.corr()
    plt.figure(figsize=(15,15))
    sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap="Greens")
    plt.show()
    return correlation
  
  
