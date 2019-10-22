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
df.drop(['WR','w', 'h','t1', 't2','EL2', 'EL3'],axis=1, inplace=True)
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


df['FailSeq']=(df['FailSeq'] == True).astype(int)
df['Brick_first']=(df['Brick_first'] == True).astype(int)
df['HI_first']=(df['HI_first'] == True).astype(int)

#check the portion
#df.FailSeq.value_counts(normalize=True)
#df.Brick_first.value_counts(normalize=True)
#df.HI_first.value_counts(normalize=True)

# normilizing Max_S_VI to its minimum value (remember first normalize so min number becomes 1)
dfb = df[ (df['FailSeq'] == 1)]
df.Max_S_VI = df.Max_S_VI.divide(dfb.Max_S_VI.min())
# now inverse Max stress in vertical interface to get the strength (remember first normalize so min number becomes 1)
# for the same load, higher stress in VI means less strength, therefore strength=1/generated_stress
#df.Max_S_VI = (1/df.Max_S_VI)
#dfb = df[ (df['FailSeq'] == 1)]
#df.Max_S_VI = df.Max_S_VI.divide(dfb.Max_S_VI.min())

#
df.head()

# ********************************
# Machine learning
# Specifying the features and Target

# before creating ML feature, we have to create a more complete data frame, 
#so after training we could compare and rank the FE results and ML results
feature_cols1=['AR', 'sh', 'VM', 'TR', 'Max_S_VI']
X_trn1 = df.loc[:,feature_cols1].copy()  #used for regression
y_trn = df. FailSeq.copy()

y_trn = df. FailSeq.copy()

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
X_trn = X_trn1.loc[:,feature_cols].copy()
X_val = X_val1.loc[:,feature_cols].copy()
X_tst = X_tst1.loc[:,feature_cols].copy()
# Plot the three subsets
plt.figure()
plt.scatter(X_trn1.TR, y_trn, 12, marker='o', color='orange')
plt.scatter(X_val1.TR, y_val, 12, marker='o', color='green')
plt.scatter(X_tst1.TR, y_tst, 12, marker='o', color='blue')
X_trn1.head()


# Learn decision trees with different depths
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_score


d_values = np.arange(1, 12, dtype='int')
models = dict()
trnErr = dict()
va_score = dict()
tst_score = dict()
roc_auc = dict()
percision_v = dict()
percision_t = dict()


for d in d_values:
    models[d] = DecisionTreeClassifier(max_depth = d, min_samples_split=2, min_samples_leaf=0.0001)
    models[d].fit(X_trn,y_trn)
    y_trn_predicted = models[d].predict(X_trn)
    y_val_predicted = models[d].predict(X_val)
    y_tst_predicted = models[d].predict(X_tst)
    tst_score[d] = models[d].score(X_tst,y_tst)
    va_score[d] = models[d].score(X_val,y_val)
    percision_v [d] = precision_score(y_val, y_val_predicted, average='macro')  
    percision_t [d] = precision_score(y_tst, y_tst_predicted, average='macro')  

    
    # INSERT YOUR CODE HERE

    y_score = dict()
    fpr = dict() #false positive rate, horizental axis in ROC curve
    tpr = dict() #true positive rate, vertical axis in ROC curve
    y_score = models[d].predict_proba(X_val)
    fpr, tpr, _ = roc_curve(y_val, y_score[:,1])
    roc_auc[d] = auc(fpr, tpr)
    plt.plot(fpr,tpr,label='Depth='+str(d) )
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend()
    
    

    print('Learning a decision tree with d = {0}.'.format(d))
# score plot
plt.figure()
plt.plot(d_values,tst_score.values(), label = 'Accuracy for Test data')
plt.plot(d_values,va_score.values(), label = 'Accuracy for validation data')
plt.xlabel('Depth')
plt.ylabel('Accuracy')
# percision
plt.figure()
plt.legend()
plt.plot(d_values,percision_v.values(),  '-ko', label = 'percision for validation data', )
plt.xlabel('Depth')
plt.ylabel('Percision')
# auc plot
plt.figure()
plt.title('AUC curve')
plt.plot(d_values,roc_auc.values(), label = 'Score for validation data')
plt.xlabel('Depth')
plt.ylabel('AUC')


#
#  DO NOT EDIT THIS FUNCTION; IF YOU WANT TO PLAY AROUND WITH VISUALIZATION, 
#  MAKE A COPY OF THIS FUNCTION AND THEN EDIT 
#
def visualize(models, X, y):
    # Initialize plotting
    if len(models) % 3 == 0:
        nrows = len(models) // 3
    else:
        nrows = len(models) // 3 + 1

    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 5.0 * nrows))
    cmap = ListedColormap(['#b30065', '#178000'])

    # Create a mesh
    X[X.columns[0]]
    xMin, xMax = X[X.columns[0]].min() , X[X.columns[0]].max() 
    yMin, yMax = X[X.columns[1]].min() , X[X.columns[1]].max() 
    xMesh, yMesh = np.meshgrid(np.arange(xMin, xMax, 0.01), 
                               np.arange(yMin, yMax, 0.01))

    for i, (p, clf) in enumerate(models.items()):
        r, c = np.divmod(i, 3)
        if nrows == 1:
            ax = axes[c]
        else:
            ax = axes[r, c]

        # Plot contours
        zMesh = clf.predict(np.c_[xMesh.ravel(), yMesh.ravel()])
        zMesh = zMesh.reshape(xMesh.shape)
        ax.contourf(xMesh, yMesh, zMesh, cmap=plt.cm.PiYG, alpha=0.6)

        # Plot data remove the below line to remove the scatter point in the visulalization
        #ax.scatter(X[X.columns[0]], X[X.columns[1]], c=y, cmap=cmap, edgecolors='k')
        ax.set_title('Tree Depth = {0}'.format(p))
		
# Learn decision trees with different depths
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import roc_curve, auc


d_values = np.arange(1, 12, dtype='int')
models = dict()
trnErr = dict()
va_score = dict()
tst_score = dict()
roc_auc = dict()

for d in d_values:
    models[d] = DecisionTreeClassifier(max_depth = d, min_samples_split=2, min_samples_leaf=0.01233265)
    models[d].fit(X_trn,y_trn)
    y_trn_predicted = models[d].predict(X_trn)
    y_val_predicted = models[d].predict(X_val)
    y_tst_predicted = models[d].predict(X_tst)
    tst_score[d] = models[d].score(X_tst,y_tst)
    va_score[d] = models[d].score(X_val,y_val)

   
    

    print('Learning a decision tree with d = {0}.'.format(d))
# score plot
visualize(models, X_trn, y_trn)



