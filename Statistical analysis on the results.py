# ranking results: linear regression works very well on young and almost well in VI failure. but better to
  #try other regression problems

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import Dataset.txt to cvs. remove the header, the first line. now importing to pandas
# defining the name of headers
data_cols=['WR','AR','w','h', 't1','t2','sh','Em','VM', 'TR', 'Young', 'EL1', 'Max_Pri_S', 'EL2', 'Max_S_HI' , 'EL3', 'Max_S_VI']
df = pd.read_csv('ALLX.csv', names=data_cols, header=0)
# drop redundent columns or series
#df.drop(['WR','w', 'h','t1', 't2','EL2', 'EL3'],axis=1, inplace=True)
#delet the negative rows by a for loop or preprocessing data
df = df[(df['Max_S_HI'] >= 0) & (df['Max_S_VI'] >= 0) & (df['Young'] >= 0) & (df['TR'] <= 2)&  (df['TR'] >= -2)& (df['Max_Pri_S'] >= 0)]
df = df[ (df['Max_Pri_S'] < 2) & (df['Max_S_HI'] < 2) & (df['Max_S_VI'] < 2) & (df['Young'] < 2)]
# drop dublicate rows
df = df.drop_duplicates()
# define the element number in brick
brick = [11,  12,  13,  14,  15,  16,  17,  18,  24,  25,  26,  27,  32,  33,  34,  35, 36,  37,  38,  39,  51,  52,  53,  54,  55,  56,  57,  58,  63,  64,  65,  66, 67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
# check if the brick fails by comparing the element number of max stress with brick elements numbers
Bfailure = df.EL1.isin(brick)
df['Bfailure']=Bfailure
#df = df[df.Bfailure]
#df.drop(['EL1'],axis=1, inplace=True)
# df.drop(['Bfailure'],axis=1, inplace=True)
# metal
df = df[ (df['Em'] == 0.5)]
# metal check if the sequence is correct First vertical interface faile, then horizental interface, and then brick
# VI fails first
FailSeq1 = ((df.Max_S_VI/df.Max_S_HI)>1) & ((df.Max_Pri_S/df.Max_S_VI)<2) & (Bfailure)
#brick fails first
Brick_first1 = ((df.Max_Pri_S/df.Max_S_HI)>2) & ((df.Max_Pri_S/df.Max_S_VI)>2) & (Bfailure)
# HI fails first
HI_first1 = ((df.Max_S_HI/df.Max_S_VI)>1) & ((df.Max_Pri_S/df.Max_S_HI)<2) & (Bfailure)
# check if the maximum stress in the composite happens in VI or HI
brick = [11,  12,  13,  14,  15,  16,  17,  18,  24,  25,  26,  27,  32,  33,  34,  35, 36,  37,  38,  39,  51,  52,  53,  54,  55,  56,  57,  58,  63,  64,  65,  66, 67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132]
HI = [1,   6,   7,   8,   9,  10,  19,  20,  21,  22,  23,  28,  29,  30,  31,  40, 41,  42,  43,  44,  49,  50,  59,  60,  61,  62,  83,  84,  85,  86,  87,  88, 89,  90,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108]
VI = [2,  3,  4,  5, 45, 46, 47, 48, 91, 92, 93, 94, 95, 96, 97, 98]
VI_first = df.EL1.isin(VI)
HI_first = df.EL1.isin(HI)
FailSeq = (VI_first) | (FailSeq1)
HI_first = (HI_first) | (HI_first1)
df['FailSeq']=FailSeq
df['Brick_first']=Brick_first1
df['HI_first']=HI_first
df.head()

# resilience and strength definition
df['resilience']= 1/df.Young
df['strength']= df.Max_S_VI
# updating strength column 
df.loc[(df['Brick_first'] == True) ,'strength'] = df.Max_Pri_S
df.loc[(df['HI_first'] == True) ,'strength'] = df.Max_S_HI
df[['FailSeq', 'Max_Pri_S','Max_S_HI', 'Max_S_VI', 'strength']]

# normilizing Max_S_VI to its minimum value (remember first normalize so min number becomes 1)
dfb = df[ (df['FailSeq'] == 1)]
df.Max_S_VI = df.Max_S_VI.divide(dfb.Max_S_VI.min())
df.Max_S_HI = df.Max_S_HI.divide(dfb.Max_S_VI.min())
df.Max_Pri_S = df.Max_Pri_S.divide(dfb.Max_S_VI.min())


#check the portion
#df.FailSeq.value_counts(normalize=True)
#df.Brick_first.value_counts(normalize=True)
#df.HI_first.value_counts(normalize=True)
df.FailSeq
#bplot=sns.boxplot(y='Max_Pri_S', x='Brick_first',data=df)
#bplot=sns.boxplot(y='Max_S_HI', x='HI_first',data=df)
bplot=sns.boxplot(y='Max_S_VI', x='FailSeq',data=df)
plt.ylim(0.9, 3.0)
bplot.axes.set_title("Bad [0]                          Good [1] ", fontsize=16)
bplot.set_xlabel("Brick_first", fontsize=0)
bplot.set_ylabel("Strength",fontsize=16)
bplot.tick_params(labelsize=10)
bplot.tick_params(labelsize=14)
# ********************************
# Machine learning
# Specifying the features and Target

# before creating ML feature, we have to create a more complete data frame, 
#so after training we could compare and rank the FE results and ML results
feature_cols1=['AR', 'sh', 'VM', 'TR', 'Max_S_VI']
X_trn1 = df.loc[:,feature_cols1]  #used for regression
y_trn = df. FailSeq

y_trn = df. FailSeq

# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn1, X_tst1, y_trn, y_tst = train_test_split(X_trn1, y_trn, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn1, X_val1, y_trn, y_val = train_test_split(X_trn1, y_trn, test_size=val_frac, random_state=42)


# removing extra column from X_trn to make the feature for ML
feature_cols=['AR', 'TR']
X_trn = X_trn1.loc[:,feature_cols]
X_val = X_val1.loc[:,feature_cols]
X_tst = X_tst1.loc[:,feature_cols]
# Plot the three subsets
plt.figure()
plt.scatter(X_trn1.TR, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val1.TR, y_val, 12, marker='o', color='green')
plt.scatter(X_tst1.TR, y_tst, 12, marker='o', color='blue')
X_trn1.head()


#Regression model
#Creating data set
df1 = df[(df['FailSeq'] == 1)]
# ********************************
# Machine learning
# Specifying the features and Target

# before creating ML feature, we have to create a more complete data frame, 
#so after training we could compare and rank the FE results and ML results
feature_cols2=['AR', 'sh', 'VM', 'TR']
X_trn1 = df1.loc[:,feature_cols2]  #used for regression
X_trn1.head()

#Regression model
#Creating data set
df1 = df[(df['FailSeq'] == 1)]
# ********************************
# Machine learning
# Specifying the features and Target

# before creating ML feature, we have to create a more complete data frame, 
#so after training we could compare and rank the FE results and ML results
feature_cols2=['AR', 'sh', 'VM', 'TR']
X_trn1 = df1.loc[:,feature_cols2]  #used for regression
y_trn1 = df1.Max_S_VI
y_trn2 = df1.Max_S_VI
# scikit-learn has many tools and utilities for model selection
from sklearn.model_selection import train_test_split
tst_frac = 0.3  # Fraction of examples to sample for the test set
val_frac = 0.1  # Fraction of examples to sample for the validation set

# First, we use train_test_split to partition (X, y) into training and test sets
X_trn1, X_tst1, y_trn1, y_tst1 = train_test_split(X_trn1, y_trn1, test_size=tst_frac, random_state=42)

# Next, we use train_test_split to further partition (X_trn, y_trn) into training and validation sets
X_trn1, X_val1, y_trn1, y_val1 = train_test_split(X_trn1, y_trn1, test_size=val_frac, random_state=42)
X_trn1.head()


data_cols=['AR','sh','VM', 'TR', 'Max_S_VI', 'SVR','R_FE', 'R_SVR']
FE_20 = pd.read_csv('TOP_FE.csv', names=data_cols, header=0)
SVR_20 = pd.read_csv('TOP_SVR.csv', names=data_cols, header=0)



from scipy.stats import norm  
df_g = df[ (df['FailSeq'] == True)]

#df_g.Max_S_VI.plot(kind='hist', normed=False, color='k',alpha=0.2)
#FE_20.Max_S_VI.plot(kind='hist', normed=False,alpha=0.5)
SVR_20.Max_S_VI.plot(kind='hist', normed=False,alpha=0.5)


#range = np.arange(2.04, 2.13, 0.001)
#plt.plot(range, norm.pdf(range,0,1))


from scipy.stats import norm  
df_g = df[ (df['FailSeq'] == True)]

#df_g.sh.plot(kind='hist', normed=True)
FE_20.Max_S_VI.plot(kind='hist', normed=False,color='k',alpha=0.2)
SVR_20.Max_S_VI.plot(kind='hist', normed=False,color='r',alpha=0.2)


#range = np.arange(2.04, 2.13, 0.001)
#plt.plot(range, norm.pdf(range,0,1))


