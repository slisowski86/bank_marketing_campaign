# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:13:45 2020

@author: sliso
"""
#importujemy potrzebne  pakiety
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob as gb
from fpdf import FPDF as pdf
from matplotlib.ticker import PercentFormatter
from scipy import stats
from iddsfunc import create_hists_ifo_yes as crhiy
from iddsfunc import to_plot_df
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.metrics import precision_score
from iddsfunc import fit_classifier
from iddsfunc import fit_regression
#wczytujemy plik csv
bankData = pd.read_csv("D:/Python/Projekt/bank-additional/bank-additional-full.csv",sep=";", comment="#")


bankData['month'].value_counts()
bankData['duration'].hist()
bankData.info()
bankData['housing'].value_counts()
print(bankData['job'].value_counts())
plt.show()


#Wizualizacja rozkładów poszczególnych zmiennych
# z użyciem stworzonej funkcji create_hists_ifo
from iddsfunc import create_hists_ifo as crhi
#crhi(bankData)

#dla zmiennych informujących o zadłużeniu klienta (default, loan, housing) tworzymy jeden histogram

debtData=bankData.loc[:,['default','housing','loan']]
debtFrame=pd.DataFrame(debtData)
    
# grupujemy dane 

defaultGroup=debtFrame.groupby(['default']).size()
housingGroup=debtFrame.groupby(['housing']).size()
loanGroup=debtFrame.groupby(['loan']).size()

liabilities={'default': defaultGroup, 'housing':housingGroup, 'loan':loanGroup}

debtFrameToPlot=pd.DataFrame(liabilities)    
debtFrameToPlot.index.name='status'
debtFrameToPlot.reset_index(level=debtFrameToPlot.index.names, inplace=True)
debtFrameToPlot=pd.melt(debtFrameToPlot, id_vars="status", var_name="kind_of_debt", value_name="number_of_records" )

# rysujemy histogram
sns.catplot(x='status', y='number_of_records', hue='kind_of_debt', data=debtFrameToPlot, kind='bar')
#zapisujemy do pliku
sns_plot=sns.catplot(x='status', y='number_of_records', hue='kind_of_debt', data=debtFrameToPlot, kind='bar')
sns_plot.savefig('Kind_of_debt.png')

# Tworzymy jeden plik pdf z rozkładami wszystkich zmiennych



#for i in f:
   # filesToPdf.append(plt.imread(i))

    
#row1=np.concatenate(filesToPdf[0:3],axis=1)
#row2=np.concatenate(filesToPdf[3:6], axis=1)
#row3=np.concatenate(filesToPdf[6:9], axis=1)

#new_image = np.concatenate((row1, row2, row3))

#plt.imsave("Distributions_1_9.pdf", new_image)
#row4=np.concatenate(filesToPdf[9:12],axis=1)
#row5=np.concatenate(filesToPdf[12:15], axis=1)
#row6=np.concatenate(filesToPdf[15:18], axis=1)

#new_image2 = np.concatenate((row4, row5, row6))

#plt.imsave("Distributions_10_18.pdf", new_image2)

#row7=np.concatenate(filesToPdf[18:20],axis=1)
#row8=np.concatenate(filesToPdf[20:], axis=1)


#new_image3 = np.concatenate((row7, row8))

#plt.imsave("Distributions_18_22.pdf", new_image3)

# rysujemy wykresy opisujące rozkład zmiennych losowych w stosunku do zmiennej celu
#zmieniamy nazwę kolumny zmiennej celu na deposit_approve oraz w nazwach kolumn, w których występują kropki zastępujemy je podkrelnikiem
# Dla zmiennych duration oraz campaign tworzymy wykresy typu boxplot, ponieważ zawierają obserwacje bardzo odbiegające
# od wartoci redniej i mediany

bankData.columns = bankData.columns.str.replace(".", "_")
bankData=bankData.rename(columns = {'y':'deposit_approve'})
    


ax=sns.boxplot(x='deposit_approve', y='duration', data=bankData)
ax=sns.boxplot(x='deposit_approve', y='campaign', data=bankData)

#sprawdzamy liczbę wartosci odstających
outliers_duration =pd.DataFrame(bankData[bankData['duration'] > bankData['duration'].mean() + 3 * bankData['duration'].std()])
outliers_campaign = pd.DataFrame(bankData[bankData['campaign'] > bankData['campaign'].mean() + 3 * bankData['campaign'].std()])

#sprawdzamy rozkład wartosci odstających względem zmiennej celu

out_dur_camp_plot=pd.DataFrame((outliers_duration['deposit_approve'].value_counts()/outliers_duration['deposit_approve'].count())*100)
out_camp_plot= pd.DataFrame((outliers_campaign['deposit_approve'].value_counts()/outliers_campaign['deposit_approve'].count())*100)

out_dur_camp_plot['deposit_approve_camp']=out_camp_plot['deposit_approve']

out_dur_camp_plot=out_dur_camp_plot.rename(columns={'deposit_approve':'duration_outliers', 'deposit_approve_camp':'campaign_outliers'})
ax=out_dur_camp_plot.T.plot(kind='bar', stacked=True, title='Distribution duration and campaign in outliers variables')
ax.set(ylabel='%_deposit_approve_rate')
age_with_yes=pd.DataFrame(bankData.loc[bankData['deposit_approve']=='yes', 'age'])
plt.hist(age_with_yes['age'], weights= np.ones(len(age_with_yes['age']))/len(age_with_yes['age']),facecolor='green', ec='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.show()


age_with_yes.describe()

#Tworzymy wykres zmiennych opisujących zadłużenie 'default', 'housing', 'loan' względem zmiennej celu

default_with_yes=pd.DataFrame(bankData.loc[bankData['deposit_approve']=='yes', 'default'])
housing_with_yes=pd.DataFrame(bankData.loc[bankData['deposit_approve']=='yes', 'housing'])
loan_with_yes=pd.DataFrame(bankData.loc[bankData['deposit_approve']=='yes', 'loan'])



default_with_yes_plot=(default_with_yes['default'].value_counts().sort_index()/bankData['default'].value_counts().sort_index())*100
housing_with_yes_plot=(housing_with_yes['housing'].value_counts().sort_index()/bankData['housing'].value_counts().sort_index())*100
loan_with_yes_plot=(loan_with_yes['loan'].value_counts().sort_index()/bankData['loan'].value_counts().sort_index())*100

debt_depo_yes_plot=pd.concat([default_with_yes_plot,housing_with_yes_plot,loan_with_yes_plot], axis=1)
ax=debt_depo_yes_plot.T.plot(kind='bar', title='Distribution of debt vs deposit_approval')
ax.set(ylabel='% deposit approval yes')
ax.legend(loc='center right')


#usuwamy kolumnę default, housing, loan

bankData = bankData.drop(['default'], axis=1)
bankData = bankData.drop(['housing'], axis=1)
bankData = bankData.drop(['loan'], axis=1)
#usuwamy wiersze z wartosciami oddalonymi w kolumnach campaign i duration 

k=outliers_duration.index.values
l=outliers_campaign.index.values
to_delete=np.concatenate([k,l])
to_del=np.unique(to_delete)

bankData=bankData.drop(to_del)

# Tworzymy rozkłady pozostałych zmiennych względem zmiennej celu
# Tworzymy wykres dla zmiennych kategorycznych opisujących osoby job maritial education rysunekwykresy 10,11,12
bankData_yes= bankData[bankData['deposit_approve']=='yes']

person_info = bankData_yes[['job', 'marital', 'education']].copy()


person_info['job'].value_counts().sort_index()
bankData['job'].value_counts().sort_index()



pd.DataFrame(jtp2)
job_to_plot=pd.DataFrame((person_info['job'].value_counts().sort_index()/bankData['job'].value_counts().sort_index())*100)
marital_to_plot=pd.DataFrame((person_info['marital'].value_counts().sort_index()/bankData['marital'].value_counts().sort_index())*100)
education_to_plot=pd.DataFrame((person_info['education'].value_counts().sort_index()/bankData['education'].value_counts().sort_index())*100)



job_to_plot = job_to_plot.rename(columns={'job':'yes_in_deposit_approval_(%)'})
g=sns.barplot(x=job_to_plot.index.values, y="yes_in_deposit_approval_(%)", data = job_to_plot)
g.set_title('Distribution of job vs deposit_approval')
g.set_xticklabels(job_to_plot.index.values,rotation=45)
marital_to_plot = marital_to_plot.rename(columns={'marital':'yes_in_deposit_approval_(%)'})
f=sns.barplot(x=marital_to_plot.index.values, y="yes_in_deposit_approval_(%)", data = marital_to_plot)
f.set_title('Distribution of marital vs deposit_approval')
f.set_xticklabels(marital_to_plot.index.values,rotation=45)
education_to_plot = education_to_plot.rename(columns={'education':'yes_in_deposit_approval_(%)'})
e=sns.barplot(x=education_to_plot.index.values, y="yes_in_deposit_approval_(%)", data = education_to_plot)
e.set_title('Distribution of education vs deposit_approval')
e.set_xticklabels(education_to_plot.index.values,rotation=45)



#sprawdzamy rozkłady zmiennych day of week oraz month względem zmiennej celu

day_of_week_to_plot=pd.DataFrame((bankData_yes['day_of_week'].value_counts().sort_index()/bankData['day_of_week'].value_counts().sort_index())*100)
months_to_plot=pd.DataFrame((bankData_yes['month'].value_counts().sort_index()/bankData['month'].value_counts().sort_index())*100)

day_of_week_to_plot=day_of_week_to_plot.rename(columns={'day_of_week':'yes_in_deposit_approval_(%)'})

months_to_plot = months_to_plot.rename(columns={'month':'yes_in_deposit_approval_(%)'})
m=sns.barplot(x=months_to_plot.index.values, y="yes_in_deposit_approval_(%)", data = months_to_plot)
m.set_title('Distribution of month vs deposit_approval')
m.set_xticklabels(months_to_plot.index.values,rotation=45)
print(day_of_week_to_plot)
#zmienna pdays
   
print(bankData.groupby(['pdays', 'deposit_approve']).size())
    
#zmienną pdays zmieniamy na binarną 0 jeżeli pdays=999 czyli nie było kontaktu i 1 w pozostałych przypadkach

bankData['pdays']=np.where(bankData['pdays']==999,0,1)
bankData['pdays'].value_counts()
bankData.groupby(['pdays', 'deposit_approve']).size()

bankData.describe()

#tworzymy wykresy dla pozostałych zmiennych względem zmiennej celu






#usuwamy kolumnę day of week i nr_employed
bankData_yes= bankData_yes.drop(['day_of_week'], axis=1)
bankData = bankData.drop(['day_of_week'], axis=1)

n=bankData_yes['nr_employed'].value_counts()
bankData_yes= bankData_yes.drop(['nr_employed'], axis=1)
bankData=bankData.drop(['nr_employed'], axis=1)

#sprawdzamy rozkłady dla cons_conf_index, euribor3m, cons_price_idx, emp_var_rate, oraz nr_of_employed względem zmiennej celu



ind_rates_info=bankData_yes[['cons_conf_idx', 'euribor3m', 'cons_price_idx', 'emp_var_rate']].copy()
#sprawdzamy wartosci w przedzaiłach dla tej zmiennej




ccidx_to_plot = pd.DataFrame((ind_rates_info['cons_conf_idx'].value_counts().sort_index()/bankData['cons_conf_idx'].value_counts().sort_index())*100)
eur_to_plot = pd.DataFrame((ind_rates_info['euribor3m'].value_counts().sort_index()/bankData['euribor3m'].value_counts().sort_index())*100)
cpidx_to_plot = pd.DataFrame((ind_rates_info['cons_price_idx'].value_counts().sort_index()/bankData['cons_price_idx'].value_counts().sort_index())*100)
evr_to_plot = pd.DataFrame((ind_rates_info['emp_var_rate'].value_counts().sort_index()/bankData['emp_var_rate'].value_counts().sort_index())*100)

#wartosci nan w ramce eur_to_plot zastepujemy srednia

n=eur_to_plot['euribor3m'].mean()
eur_to_plot=eur_to_plot.fillna({'euribor3m':n})
ccidx_to_plot = ccidx_to_plot.rename(columns={'cons_conf_idx':'yes_in_deposit_approval_(%)'})
ccidx_plot=sns.barplot(x=ccidx_to_plot.index.values, y="yes_in_deposit_approval_(%)", data = ccidx_to_plot)
ccidx_plot.set_title('Distribution of cons_conf_idx vs deposit_approval')
ccidx_plot.set_xticklabels(ccidx_to_plot.index.values,rotation=90)


cpidx_to_plot = cpidx_to_plot.rename(columns={'cons_price_idx':'yes_in_deposit_approval_(%)'})
cpidx_to_plot.index=cpidx_to_plot.index.values.round(3)
cpidx_plot=sns.barplot(x=cpidx_to_plot.index.values, y="yes_in_deposit_approval_(%)", data = cpidx_to_plot)
cpidx_plot.set_title('Distribution of cons_price_idx vs deposit_approval')
cpidx_plot.set_xticklabels(cpidx_to_plot.index.values,rotation=90)

bins_eur=[0.5, 1, 1.5, 2, 2.5,3,3.5,4,4.5,5,5.5]
cats_eur = pd.cut(eur_to_plot.index.values, bins_eur, right=False)
pd.value_counts(cats_eur)
eur_to_plot['eur3rate']=cats_eur



eur_to_plot = eur_to_plot.rename(columns={'euribor3m':'yes_in_deposit_approval_(%)'})
eur_plot=sns.barplot(x='eur3rate', y="yes_in_deposit_approval_(%)", data = eur_to_plot)
eur_plot.set_title('Distribution of euribor3m vs deposit_approval')
eur_plot.set_xticklabels(eur_plot.get_xticklabels(),rotation=90)

evr_to_plot = evr_to_plot.rename(columns={'emp_var_rate':'yes_in_deposit_approval_(%)'})
evr_plot=sns.barplot(x=evr_to_plot.index.values, y="yes_in_deposit_approval_(%)", data = evr_to_plot)
evr_plot.set_title('Distribution of emp_var_rate vs deposit_approval')
evr_plot.set_xticklabels(evr_to_plot.index.values,rotation=90)


# sprawdzamy rozkłady pozostałych zmiennych względem zmiennej celu funkcja to_plot_df

bankData['pdays']=np.where(bankData['pdays']==0,'no','yes')
bankData_yes['pdays'].value_counts()
bankData_yes['pdays']=np.where(bankData_yes['pdays']==999,'no','yes')
idxs=['pdays', 'previous', 'poutcome', 'contact']

to_plot_df(bankData_yes, bankData, idxs)

# zostały jeszcze age i duration

bankData_yes['age'].describe()
bankData_yes['duration'].describe()

age_to_plot=pd.DataFrame((bankData_yes['age'].value_counts().sort_index()/bankData['age'].value_counts().sort_index())*100)
age_to_plot=age_to_plot.fillna({'age':age_to_plot['age'].mean()})

bins_age=[10,25,40,55,70,85,100]
cats_age=pd.cut(age_to_plot.index.values, bins_age, right=False)
pd.value_counts(cats_age)
age_to_plot['age_bins']=cats_age

age_to_plot = age_to_plot.rename(columns={'age':'yes_in_deposit_approval_(%)'})
age_plot=sns.barplot(x='age_bins', y="yes_in_deposit_approval_(%)", data = age_to_plot)
age_plot.set_title('Distribution of age vs deposit_approval')
age_plot.set_xticklabels(age_plot.get_xticklabels(),rotation=90)



duration_to_plot=pd.DataFrame((bankData_yes['duration'].value_counts().sort_index()/bankData['duration'].value_counts().sort_index())*100)
duration_to_plot=duration_to_plot.fillna({'duration':duration_to_plot['duration'].mean()})

bins_duration=[0,90,180,270,360,450,540,630,720,810,900,990,1080]
cats_duration=pd.cut(duration_to_plot.index.values, bins_duration, right=False)
pd.value_counts(cats_duration)

duration_to_plot['duration_bins']=cats_duration

duration_to_plot = duration_to_plot.rename(columns={'duration':'yes_in_deposit_approval_(%)'})
duration_plot=sns.barplot(x='duration_bins', y="yes_in_deposit_approval_(%)", data = duration_to_plot)
duration_plot.set_title('Distribution of duration vs deposit_approval')
duration_plot.set_xticklabels(duration_plot.get_xticklabels(),rotation=90)

#zapisujemy ramki danych
bankData.to_csv("bankData_first_clean.csv")
bankData_yes.to_csv("bankData_yes_first_clean.csv")

bankData['marital'].value_counts()
bankData_yes['marital'].value_counts()
#przygootowujemy dane do użycia w algorytmach uczenia maszynowego
#wczytujemy ramki danych

bankData_to_model=pd.read_csv("D:/Python/Projekt/bankData_first_clean.csv",sep=",", comment="#", index_col=0)

#usuwamy kolumnę marital
bankData_to_model=bankData_to_model.drop(['marital'], axis=1)

#sprawdzamy skorlowanie zmiennych numerycznych

bankData_to_model.info()

bankData_numeric=bankData_to_model.select_dtypes(exclude='object')
bankData_numeric=bankData_numeric.drop(bankData_numeric.columns[0], axis=1)

#sprawdzamy korelację zmiennych numrycznych

corr=bankData_numeric.corr()

fig, ax = plt.subplots(1, 1, figsize=(10,6))

ax = sns.heatmap(corr, 
                 ax=ax,           # Axes in which to draw the plot, otherwise use the currently-active Axes.
                 cmap="coolwarm", # Color Map.
                 square=True,    # If True, set the Axes aspect to “equal” so each cell will be square-shaped.
                 annot=True, 
                 fmt='.2f',       # String formatting code to use when adding annotations.
                 #annot_kws={"size": 14},
                 linewidths=.05)



fig.suptitle('Bank Data numeric Correlation Heatmap', 
              fontsize=14, 
              fontweight='bold')


#odzrzucamy zmienne cons_price_idx oraz euribor_3m
bankData_to_model=bankData_to_model.drop(['cons_price_idx', 'euribor3m'], axis=1)

#Nazwę zmiennej ‘pdays’ zmieniamy na ‘prev_contact’ a wartości zamieniamy na 1 w przypadku ‘yes’ i 0 dla ‘no’ 

bankData_to_model=bankData_to_model.rename(columns={'pdays':'prev_conact'})
bankData_to_model=bankData_to_model.rename(columns={'prev_conact':'prev_contact'})
bankData_to_model['prev_contact'].value_counts()
bankData_to_model['prev_contact']=np.where(bankData_to_model['prev_contact']=='no',0,1)

bankData_to_model['poutcome'].value_counts()
#Wartości dla zmiennej ‘contact’ również zmieniamy na 1 dla wartości cellular i 0 dla wartości telephone.
bankData_to_model['contact']=np.where(bankData_to_model['contact']=='cellular',1,0)
bankData_to_model['contact'].value_counts()

#Zmienną celu zamieniamy na 1 w przypadku wartości yes i 0 dla no.
bankData_to_model['deposit_approve']=np.where(bankData_to_model['deposit_approve']=='yes',1,0)
bankData_to_model['deposit_approve'].value_counts()
bankData_to_model.info()

#Zmienne kategoryczne'month', 'poutcome', 'job', 'education' kodujemy binarnie za pomocą pakietu OneHotEncoder z biblioteki sklearn.preprocessing
#Sprawdzimy iloć rekordów z wartoscią unknown w 'job' i 'education'
bankData_to_model['job'].value_counts()
bankData_to_model['education'].value_counts()

#usuwamy rekordy z wartosciami unknown

del_row_job=bankData_to_model[bankData_to_model['job']=='unknown'].index
print(del_row_job)

bankData_to_model=bankData_to_model.drop(del_row_job)
bankData_to_model['education'].value_counts()
del_row_edu=bankData_to_model[bankData_to_model['education']=='unknown'].index
bankData_to_model=bankData_to_model.drop(del_row_edu)
bankData_to_model=bankData_to_model.reset_index()
bankData_to_model=bankData_to_model.drop(['index'], axis=1)
#Zmienne kategoryczne 'month', 'poutcome', 'job', 'education' kodujemy binarnie za pomocą pd.get_dummies 



month_encoded=pd.get_dummies(bankData_to_model['month'].copy())
poutcome_encoded=pd.get_dummies(bankData_to_model['poutcome'].copy())
job_encoded=pd.get_dummies(bankData_to_model['job'].copy())
education_encoded=pd.get_dummies(bankData_to_model['education'].copy())
print(month_encoded)
bank_to_model_encoded=bankData_to_model.copy()

bank_to_model_encoded=bank_to_model_encoded.drop(['month', 'poutcome', 'job', 'education'], axis=1)

bank_to_model_all=pd.concat([bank_to_model_encoded, month_encoded, poutcome_encoded, job_encoded, education_encoded ], axis=1)
bank_to_model_all.info()
#zapisyjemy ramkę do pliku

bank_to_model_all.to_csv('bankData_all_encoded.csv')

#Tworzymy ramkę danych z pierwotną liczbą kategorii kodując według wykresów korelacji zmiennych ze zmienną celu

bankData_encode_cat=bankData_to_model.copy()
# W przypadku zmiennej poutcome jako 1 oznaczymy rekordy z wartością ‘succes’, jako 0 pozostałe rekordy
bankData_encode_cat['poutcome']=np.where(bankData_encode_cat['poutcome']=='success',1,0)

bankData_encode_cat['poutcome'].value_counts()

#Na podstawie wykresu zminnej ‘job’ w stosunku do zmennej deposit approval jako 1 oznaczymy wartości ‘admin’,  ‘retired’, ‘student’, oraz ‘unemployed’, jako 0 pozostałe kategorie

bankData_encode_cat['job']=bankData_encode_cat['job'].replace(['admin.','retired','student','unemployed'],1)
bankData_encode_cat['job'].value_counts()
bankData_encode_cat['job']=bankData_encode_cat['job'].replace(['blue-collar','entrepreneur','housemaid','management', 'self-employed', 'services','technician'],0)
bankData_encode_cat['job'].value_counts()

#Patrząc na wykres korelacji zmiennej month względem zmiennej celu, jako 1 oznaczymy misiące dec, mar, oct oraz sep, a 0 pozostałe miesiące
bankData_encode_cat['month']=bankData_encode_cat['month'].replace(['dec','mar','oct','sep','apr'],1)
bankData_encode_cat['month']=bankData_encode_cat['month'].replace(['aug','jul','jun','may','nov'],0)
bankData_encode_cat['month'].value_counts()

#Zmienną ‘education’ zakodujemy jako 1 dla wykształcenia ‘high school’ i wyższego, oraz 0 dla wykształcenia poniżej ‘high school’.
bankData_encode_cat['education'].value_counts()
bankData_encode_cat['education']=bankData_encode_cat['education'].replace(['high.school','illiterate','professional.course','university.degree'],1)
bankData_encode_cat['education']=bankData_encode_cat['education'].replace(['basic.4y','basic.6y','basic.9y'],0)
bankData_encode_cat['education'].value_counts()

bankData_encode_cat.info()

#Zapisujemy ramkę do pliku
bankData_encode_cat.to_csv('bankData_binary_cat.csv')
bankData_to_model.to_csv('bankData_original_cat_clear.csv')

#W pierwszej kolejności zastosujemy model drzewa decyzyjnego dla danych gdzie stosowaliśmy „OneHotEncoding”, czyli danych z 40 kategoriami.
#Dzielimy model na model uczący i model testowy w stosunku 80% do 20%
bankData_decisionTree_onehotencode = pd.read_csv("D:/Python/Projekt/bankData_all_encoded.csv",sep=",", comment="#",index_col=0)
deposit_approve=bankData_decisionTree_onehotencode.pop('deposit_approve')                                                
bankData_decisionTree_onehotencode['deposit_approve']=deposit_approve
bankData_decisionTree_onehotencode['deposit_approve']=np.where(bankData_decisionTree_onehotencode['deposit_approve']=='no',0,1)

X_dec_tree=bankData_decisionTree_onehotencode.iloc[:, :-1]
y_dec_tree=bankData_decisionTree_onehotencode.iloc[:,-1]

X_ucz_tree, X_test_tree, y_ucz_tree, y_test_tree = sklearn.model_selection.train_test_split(X_dec_tree, y_dec_tree, test_size=0.2, random_state=12345)
print(X_ucz_tree.shape)
print(X_test_tree.shape)
print(y_ucz_tree.shape)
print(y_test_tree.shape)

my_tree=sklearn.tree.DecisionTreeClassifier()
my_tree.fit(X_ucz_tree, y_ucz_tree)

pd.Series(fit_classifier(my_tree, X_ucz_tree, X_test_tree, y_ucz_tree, y_test_tree))

#Podstawimy do modelu dane, w których wartości kategoryczne kodowaliśmy jako 0 i 1 według korelacji zminnych objaśniających i zmiennej celu.

bankData_decisionTree_binary=pd.read_csv("D:/Python/Projekt/bankData_binary_cat.csv",sep=",", comment="#",index_col=0)
                  
X_dec_tree_bin=bankData_decisionTree_binary.iloc[:, :-1]
y_dec_tree_bin=bankData_decisionTree_binary.iloc[:,-1]                                      

X_ucz_tree_bin, X_test_tree_bin, y_ucz_tree_bin, y_test_tree_bin = sklearn.model_selection.train_test_split(X_dec_tree_bin, y_dec_tree_bin, test_size=0.2, random_state=12345)

my_tree_bin=sklearn.tree.DecisionTreeClassifier()
my_tree_bin.fit(X_ucz_tree_bin, y_ucz_tree_bin)
pd.Series(fit_classifier(my_tree_bin, X_ucz_tree_bin, X_test_tree_bin, y_ucz_tree_bin, y_test_tree_bin))


#Model regresji liniowej

bankData_linear=bankData_decisionTree_binary.copy()

X_linear=bankData_linear.iloc[:, :-1]
y_linear=bankData_linear.iloc[:, -1]

reglinear = sklearn.linear_model.LinearRegression()

reglinear.fit(X_linear,y_linear)

reglinear.intercept_

reglinear.coef_

#Aby zastosować model regresji liniowej standaryzujemy dane, które były numeryczne. Dane kategoryczne, którym nadaliśmy wartości 0 i 1 pozostawiamy.

col_to_std=['age', 'duration', 'campaign', 'emp_var_rate', 'cons_conf_idx']
X_std_df=bankData_linear.iloc[:,:-1]
X_std_df[col_to_std]=(X_std_df[col_to_std]-X_std_df[col_to_std].mean(axis=0))/X_std_df[col_to_std].std(axis=0)


reglinear_std = sklearn.linear_model.LinearRegression()
reglinear_std.fit(X_std_df, y_linear)

pd.Series(reglinear_std.coef_, index=X_std_df.columns.to_list()).round(4).sort_values(ascending=False)

X_std=bankData_linear.iloc[:, :-1]
X_std=(X_std-X_std.mean(axis=0))/X_std.std(axis=0)
y_std=bankData_linear.iloc[:, -1]
y_std=(y_std-y_std.mean())/y_std.std()

reglinear_std_all = sklearn.linear_model.LinearRegression()
reglinear_std_all.fit(X_std, y_std)

pd.Series(reglinear_std_all.coef_, index=X_std.columns.to_list()).round(4).sort_values(ascending=False)



X_ucz, X_test, y_ucz, y_test = sklearn.model_selection.train_test_split(X_linear, y_linear, test_size=0.2, random_state=12345)

X_ucz_std, X_test_std, y_ucz_std, y_test_std = sklearn.model_selection.train_test_split(X_std, y_std, test_size=0.2, random_state=12345)
params = ["Reg. liniowa"]
res = [fit_regression(X_ucz, X_test, y_ucz, y_test)]
metric_reg=pd.DataFrame(res, index=params)
print(metric_reg)
matric_list=metric_reg.to_list()
X_ucz_70, X_test_70, y_ucz_70, y_test_70 = sklearn.model_selection.train_test_split(X_linear, y_linear, test_size=0.3, random_state=12345)
res_70 = [fit_regression(X_ucz_70, X_test_70, y_ucz_70, y_test_70)]
pd.DataFrame(res_70, index=params)


#Klasyfikator binarny

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
sgd_clf=SGDClassifier(random_state=42)
sgd_clf.fit(X_ucz, y_ucz)

cross_val_score(sgd_clf, X_ucz, y_ucz, cv=3, scoring="accuracy")

y_train_pred=cross_val_predict(sgd_clf, X_ucz, y_ucz, cv=3)

precision_score(y_ucz, y_train_pred )
recall_score(y_ucz, y_train_pred )