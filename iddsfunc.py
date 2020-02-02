# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 00:37:01 2020

@author: sliso
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob as gb
import sklearn


def create_hists_ifo(df):
    """Funkcja tworząca histogramy
       dla trzech typów : int64, float64, object
       argumenty:
       df - ramka danych
       wyjscie:
       histogram dla każdego typu danych
    """

    int_vars=df.select_dtypes(include='int64').columns
    float_vars=df.select_dtypes(include='float64').columns
    object_vars=df.select_dtypes(include='object').columns
    
    vars_dict={'int64':int_vars, 'float64':float_vars, 'object':object_vars}
    for val in vars_dict['int64']:
        num_hist(df,val)
    for val in vars_dict['float64']:
        float_hist(df,val)
    for val in vars_dict['object']:
        object_hist(df,val)
         
def create_hists_ifo_yes(df):
    """Funkcja tworząca histogramy
       dla trzech typów : int64, float64, object
       argumenty:
       df - ramka danych
       wyjscie:
       histogram dla każdego typu danych
    """

    int_vars=df.select_dtypes(include='int64').columns
    float_vars=df.select_dtypes(include='float64').columns
    object_vars=df.select_dtypes(include='object').columns
    
    vars_dict={'int64':int_vars, 'float64':float_vars, 'object':object_vars}
    for val in vars_dict['int64']:
        num_hist_yes(df,val)
    for val in vars_dict['float64']:
        float_hist_yes(df,val)
    for val in vars_dict['object']:
        object_hist_yes(df,val)
               


    


def num_hist(d,v):
    x=d[v]
    plt.hist(x, facecolor='green', ec='black')
    
    plt.xlabel(v)
    plt.ylabel('Number of records')
    plt.title('Distribution of ' + v + ' variable'  )
    ax = plt.axes()
    ax.set_facecolor('#f6f5f6')
    
    plt.grid(True)
    fileTitle=plt.title('Distribution of ' + v + ' variable'  )
    plt.savefig('Wykresy/Rozklady/Distribution of ' + v + ' variable.png')
    
    plt.show()
    
def float_hist(d,v):
    x=d[v]
    plt.hist(x, facecolor='blue', ec='black')
    plt.xlabel(v)
    plt.ylabel('Number of records')
    plt.title('Distribution of ' + v + ' variable'  )
    ax = plt.axes()
    ax.set_facecolor('#f6f5f6')
    plt.grid(True)
    fileTitle=plt.title('Distribution of ' + v + ' variable'  )
    plt.savefig('Wykresy/Rozklady/Distribution of ' + v + ' variable.png')
    
    
    plt.show()
def object_hist(d,v):
    x=d[v]
    
    labels, counts = np.unique(x, return_counts=True)
    plt.bar(labels, counts, color='red',align='center')
    plt.gca().set_xticks(labels)
    plt.xlabel(v)
    plt.ylabel('Number of records')
    plt.xticks(rotation=75, ha='center')
    ax = plt.axes()
    ax.set_facecolor('#f6f5f6')
   
    plt.grid(True)
    plt.title('Distribution of ' + v + ' variable'  )
    fileTitle=plt.title('Distribution of ' + v + ' variable'  )
    plt.savefig('Wykresy/Rozklady/Distribution of ' + v + ' variable.png')
   
    plt.show()
    

def fileagg(ext=""):
    """
    Funkcja zwracająca listę plików z okrelonym rozszerzeniem
    ext - rozszerzenie pliku
    """
    return [f for f in gb.glob(f"*{ext}")]
    


    
def to_plot_df(df_yes,df_original,col_names):
    """
    Funkcja rysująca wykres pokazujący udział sukcesów w stosunku do ogólnej liczby 
    rekordów danej wartosci w poszczególnych zmiennych
    df_yes - ramka danych ze zmienną celu = sukces
    df_original - oryginalna ramka danych
    col_names - nazwy zmiennych
    """
    for i in col_names:
        s=(df_yes[i].value_counts().sort_index()/df_original[i].value_counts().sort_index())*100
        y=pd.Series(s).values
        x=pd.Series(s).index
        plt.bar(x,y)
        plt.title('Distribution of ' + i +' vs deposit_approval')
        plt.xlabel(i)
        plt.ylabel("yes_in_deposit_approval_(%)")
        plt.show()
        
def fit_classifier(alg, X_ucz, X_test, y_ucz, y_test):
    alg.fit(X_ucz, y_ucz)
    y_pred = alg.predict(X_test)
    return {
        "ACC": sklearn.metrics.accuracy_score(y_pred, y_test),
        "P":   sklearn.metrics.precision_score(y_pred, y_test),
        "R":   sklearn.metrics.recall_score(y_pred, y_test),
        "F1":  sklearn.metrics.f1_score(y_pred, y_test)
    }

def fit_regression(X_ucz, X_test, y_ucz, y_test):
    r = sklearn.linear_model.LinearRegression()
    r.fit(X_ucz, y_ucz)
    y_ucz_pred = r.predict(X_ucz)
    y_test_pred = r.predict(X_test)
    r2 = sklearn.metrics.r2_score
    mse = sklearn.metrics.mean_squared_error
    mae = sklearn.metrics.mean_absolute_error
    return {
        "r_score_u": r2(y_ucz, y_ucz_pred),
        "r_score_t": r2(y_test, y_test_pred),
        "MSE_u": mse(y_ucz, y_ucz_pred),
        "MSE_t": mse(y_test, y_test_pred),
        "MAE_u": mae(y_ucz, y_ucz_pred),
        "MAE_t": mae(y_test, y_test_pred)
    }
