# -*- coding: utf-8 -*-
"""
The aim is to find the best cross-validated model for predicting loan default using this data.
Conclusion: SVC Radial fine-tuned one, with an AUC of 77.783%. The Naive Bayes also performed well. 

"""
# Import the necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
# %matplotlib inline

# Read the excel file
your_path = "/Users/Jeremymeyer/Desktop/GermanDataClean.csv"
df = pd.read_csv(your_path) 

#   //////////////////////////////////////////////////////////////
#  ///////// Preparing the data for Machine Learning ////////////
# //////////////////////////////////////////////////////////////

#Let's split our Dataframe into two sub-dataframes: one with the defaults and one with the non_defaults
non_default = df[df['Default'] == 0]
default = df[df['Default'] == 1]

# Look at the shape of the data
print('Shape of the data:', df.shape)

# Examining the dataframe for missing data and examine the type of data
print(df.isnull().any().any(), df.info())

# Look at the defaulted loans numbers
plt.figure(figsize=(7,6))
sns.set(style="ticks", color_codes=True)
sns.countplot(x="Default", data=df);

#   //////////////////////////////////////////////////////////////
#  /////  Dealing with the unbalanced data with undersampling ///
# //////////////////////////////////////////////////////////////

#Our loan data are very unbalanced: 700 defaults for 300 non-defaults
#Then, we may want to transform are dataset by selecting only a portion of the 700 defaults to not overtrain our model
#Furthermore, our model would then focus as much on the positive class as on the negative class
#In loan predictions the false positives have less impact than the false negatives

#Let's shuffle the non_default dataframe
shuffle(default)

#Drop a certain amount of rows in the non_default dataframe
default = default.drop(default.index[0:400])

#Merge the two dataframes to create our main dataframe
df = default.append(non_default)
df = df.reset_index(drop = True)

# Look balanced data set
plt.figure(figsize=(7,6))
sns.set(style="ticks", color_codes=True)
sns.countplot(x="Default", data=df);

# Drop the label column in the main dataframe
y = df.Default
df= df.drop('Default', axis=1) 

# We define the number of Default and Viable loans in our sample
B,M = y.value_counts()
print('Number of viable loans: ',B)
print('Number of default loans : ',M)

# Standardize the data in a new DF
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
scaled_data = Scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data , columns=df.columns)
X=df_scaled.values

#   //////////////////////////////////////////////////////////////
#  //////// Determining which features are important  ///////////
# //////////////////////////////////////////////////////////////

# Display the importance of some features using the Extra Tree Classifier 
from sklearn.ensemble import ExtraTreesClassifier

# fit an Extra Trees model to ALL the data
clf = ExtraTreesClassifier()
clf.fit(df, y)

# display the relative importance of each attribute
importance_list=clf.feature_importances_
    
# plot the bar chart 
plt.figure(figsize=(12,6))
y_pos = np.arange(len(df.columns))
plt.barh(y_pos,importance_list,align='center')
plt.yticks(range(len(df.columns)),df.columns)
plt.xlabel('Relative Importance in the Random Forest')
plt.ylabel('Features')
plt.title('Relative importance of Each Feature in %');

#Let's select the n best features (we choose n = 15)

#We create a duplicate of our list of importance scores
copy_of_importance_list = importance_list.tolist()

list_of_best_features = []
our_index = 0
for n in range(0,20):
    our_index = copy_of_importance_list.index(max(copy_of_importance_list))
    list_of_best_features.append(df.columns[our_index])
    copy_of_importance_list[our_index] = 0 

#We create a new array containing only the n features 
df_scaled_new = df_scaled
df_scaled_new = df_scaled[list_of_best_features]
X = df_scaled_new.values

# Examine the dispersion of the defaults/non defaulted amongst the features in a multivariate setting
from pandas.plotting import radviz
plt.figure(figsize=(8,8))
df_all_data = pd.read_csv(your_path) 
Scaler = StandardScaler()
scaled_data = Scaler.fit_transform(df_all_data)
df_all_data_scaled = pd.DataFrame(scaled_data , columns=df_all_data.columns)
radviz(df_all_data_scaled, 'Default', color='BGR');

# Same thing but this time dropping some of the features for clarity
plt.figure(figsize=(10,10))
df_all_data = pd.read_csv(your_path) 
less_interesting_features=['Guarantors','Sex & Marital Status','Instalment per cent', 'Duration in Current address','Age (years)','Most valuable available asset', 'Concurrent Credits', 'Type of apartment', 'No of Credits at this Bank','Occupation', 'No of dependents', 'Telephone', 'Foreign Worker','Value Savings/Stocks']
df_all_data_dropped = df_all_data.drop(less_interesting_features, axis=1)
radviz(df_all_data_dropped, 'Default',color='BGR');

#   //////////////////////////////////////////////////////////////
#  ////////              Train - Test - Split             ///////
# //////////////////////////////////////////////////////////////

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.30, random_state=7)

print('Shape of the training set', X_train.shape)
print('Shape of the testing set', X_test.shape)
print('Shape of the labels for the training set', y_train.shape)
print('Shape of the labels for the testing set', y_test.shape)

#   //////////////////////////////////////////////////////////////
#  ////////      Model Selection and Accuracy Evaluation  ///////
# //////////////////////////////////////////////////////////////

# Running a PCA Analysis with 3 principal components
from sklearn.decomposition import PCA
pcaModel=PCA(n_components=3)
pcaModel.fit(df_scaled)
df_scaled_pca = pcaModel.transform(df_scaled)

#   //////////////////////////////////////////////////////////////
#  ////////////// Import some usefull functions  ////////////////
# //////////////////////////////////////////////////////////////

# Definition of function to train and provide some metrics to evaluate models
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

# For a summary of the metrics
def display_model_evaluation(clf):
    
    # Evaluate the accuracy in and out-of-sample
    print('')
    print('Metrics & Accuracy Evaluation:', clf)
    print('')
    
    # Fit the model with the in sample data
    clf.fit(X_train, y_train)
    
    # Issue a prediction with the training set
    y_pred= clf.predict(X_train)
    print('')
    print ("Average Precision on training set:",precision_score(y_train, y_pred, average='weighted'))
    print ("Average Recall on training set:",recall_score(y_train, y_pred, average='weighted'))
    print ("Average F1 Score on training set:",f1_score(y_train, y_pred, average='weighted'))
    print('') 
    print("Root Mean Squared Error (RMSE):",np.sqrt(mean_squared_error(y_train, y_pred)))
    print("Mean Absolute Error (MAE):",mean_absolute_error(y_train, y_pred))
    print('')
    print('================================================')
    
    # Issue a prediction with the testing 
    y_pred= clf.predict(X_test)
    print('')
    print ("Average Precision on testing set:",precision_score(y_test, y_pred, average='weighted'))
    print ("Average Recall on testing set:",recall_score(y_test, y_pred, average='weighted'))
    print ("Average F1 Score on testing set:",f1_score(y_test, y_pred, average='weighted'))
    print('')
    print("Root Mean Squared Error (RMSE):",np.sqrt(mean_squared_error(y_test, y_pred)))
    print("Mean Absolute Error (MAE):",mean_absolute_error(y_test, y_pred))
    print('')
    
    # Provide Classification Report
    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    
# For plotting nice confusion matrices
# http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Function to rapidly call the confusion matrices
def display_confusion_matrices(y_test,y_pred):
    
    # Define the figure 
    plt.figure(figsize=(14,5))
    
    # Display the confusion Matrix 
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Plot non-normalized confusion matrix
    plt.subplot(1,2,1)
    plt.tight_layout()
    plot_confusion_matrix(confusion_mat, classes=np.array(['0','1']), title='Confusion matrix, without normalization');
    print('')
    
    # Plot normalized confusion matrix
    plt.subplot(1,2,2)
    plt.tight_layout()
    plot_confusion_matrix(confusion_mat, classes=np.array(['0','1']), normalize=True, title='Normalized confusion matrix');

#   //////////////////////////////////////////////////////////////
#  ////////                     Fine Tuning               ///////
# //////////////////////////////////////////////////////////////

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

################# Hyper-parameter tuning on SVC ########################################

print("Fine Tuning on SVC models")
parameters = {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel':['linear','rbf','sigmoid','poly']}
grid = GridSearchCV(SVC(probability = True), parameters, verbose=3) 
grid.fit(X_train,y_train)
print(grid.best_params_)
display_model_evaluation(grid)
y_pred = grid.predict(X_test)
display_confusion_matrices(y_test,y_pred)

################# Hyper-parameter tuning on Logistic Regression ###########################

print("Fine Tuning on Logistic Regression")
grid=GridSearchCV(cv=None,
       estimator=LogisticRegression(C=1.0, intercept_scaling=1, dual=False, fit_intercept=True,
          penalty='l2', tol=0.0001),
       fit_params={}, iid=True, n_jobs=1,
       param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
       pre_dispatch='2*n_jobs', refit=True, verbose=3)
grid.fit(X_train, y_train)
print(grid.best_params_)
display_model_evaluation(grid)
y_pred = grid.predict(X_test)
display_confusion_matrices(y_test,y_pred)

################# Hyper-parameter tuning on KNN ####################################################

print("Fine Tuning on KNN model")
k = np.arange(200)+1
parameters = {'n_neighbors': k}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn,parameters,cv=10,verbose=3)
grid.fit(X_train,y_train)
print(grid.best_params_)
display_model_evaluation(grid)
y_pred = grid.predict(X_test)
display_confusion_matrices(y_test,y_pred)


################# Hyper-parameter tuning on MLP Classifier ########################################

print("Fine Tuning on MLPClassifier")
param_grid = {
        'hidden_layer_sizes': [(7, 7), (128,), (128, 7)],
        'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'epsilon': [1e-3, 1e-7, 1e-8, 1e-9, 1e-8]
        }
grid= GridSearchCV(MLPClassifier(activation='relu',learning_rate='adaptive', learning_rate_init=1., early_stopping=True, shuffle=True),param_grid=param_grid, n_jobs=-1)
grid.fit(X_train,y_train)
print(grid.best_params_)
display_model_evaluation(grid)
y_pred = grid.predict(X_test)
display_confusion_matrices(y_test,y_pred)

## Another method to fine tune the MLP 
# =============================================================================
# from sklearn.model_selection import LeaveOneOut
# from sklearn.pipeline import make_pipeline
# 
# sc = StandardScaler()
# mlc = MLPClassifier(activation = 'relu', random_state=1,nesterovs_momentum=True)
# loo = LeaveOneOut()
# pipe = make_pipeline(sc, mlc)
# parameters = {"mlpclassifier__hidden_layer_sizes":[(168,),(126,),(498,),(166,)],"mlpclassifier__solver" : ('sgd','adam'), "mlpclassifier__alpha": [0.001,0.0001],"mlpclassifier__learning_rate_init":[0.005,0.001] }
# clf = GridSearchCV(pipe, parameters,n_jobs= -1,cv = loo)
# clf.fit(X_train,y_train)
# estimators = clf.best_estimator_
# print(estimators)
# print(clf.best_params_)
# display_model_evaluation(clf)
# y_pred = clf.predict(X_test)
# display_confusion_matrices(y_test,y_pred)
# 
# =============================================================================

################# Hyper-parameter tuning on Decision Tree ########################################

print("Fine Tuning on Decision Tree ")
# Define the parameter values that should be searched
dtc = DecisionTreeClassifier(min_samples_split=100, random_state=1)
param_grid={'min_samples_split': list(range(2, 108)) }
grid = GridSearchCV(dtc, param_grid, cv=10, scoring='accuracy')
grid.fit(X_train,y_train)
print(grid.best_params_)
display_model_evaluation(grid)
y_pred = grid.predict(X_test)
display_confusion_matrices(y_test,y_pred)

#   //////////////////////////////////////////////////////////////
#  ////////        AUC and Features importance ranking    ///////
# //////////////////////////////////////////////////////////////

# Define the list of the models used
model1=LogisticRegression()
model8=SVC(kernel='poly',C=1, gamma=0.001)
model9=SVC(kernel='rbf',C=100, gamma=0.001)
model10 = SVC(kernel='linear',C=1, gamma=0.001)

models = [model1,model8,model9,model10]
models_names= ['Logistic Regression','SVC Polynomial','SVC Radial','SVC linear','Naive Bayes','KNN model','Neural Network','Decision Tree Classifier']

auc_list=[]
fpr_list=[]
tpr_list=[]

for model in models:
    model=model.fit(X_train,y_train)
    y_score = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    auc_list.append(auc(fpr,tpr))

#Let's integrate the Naive Bayes model to our model to build the ROC curve
model11 = GaussianNB()
model11= model11.fit(X_train,y_train)
display_model_evaluation(model11)
y_pred = model11.predict(X_test)
display_confusion_matrices(y_test,y_pred)
model_11_score = model11.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, model_11_score[:, 1])
fpr_list.append(fpr)
tpr_list.append(tpr)
auc_list.append(auc(fpr,tpr))

# Let's integrate the KNN model to the ROC curve
model12  = KNeighborsClassifier(n_neighbors=24)
model12= model12.fit(X_train,y_train)
model_12_score = model12.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, model_12_score[:, 1])
fpr_list.append(fpr)
tpr_list.append(tpr)
auc_list.append(auc(fpr,tpr))

# Let's integrate the neural network model to the ROC curve
model13  = MLPClassifier(activation='relu',epsilon=0.001,tol=0.01, hidden_layer_sizes = (128,))
model13= model13.fit(X_train,y_train)
model_13_score = model13.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, model_13_score[:, 1])
fpr_list.append(fpr)
tpr_list.append(tpr)
auc_list.append(auc(fpr,tpr))

# Let's integrate the Decision Tree Model to the ROC curve
model7=DecisionTreeClassifier(min_samples_split=106)
model7= model7.fit(X_train,y_train)
model_7_score = model7.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, model_7_score[:, 1])
fpr_list.append(fpr)
tpr_list.append(tpr)
auc_list.append(auc(fpr,tpr))

print("The AUC for logistic regression is: ",auc_list[0])
print("The AUC for SVC Polynomial is: ",auc_list[1])
print("The AUC for SVC Radial is: ",auc_list[2]) 
print("The AUC for SVR Linear is: ",auc_list[3]) 
print("The AUC for Naive Bayes is: ",auc_list[4]) 
print("The AUC for KNN is : ",auc_list[5])
print("The AUC for Neural Networks is : ",auc_list[6]) 
print("The AUC for Decision Tree is : ",auc_list[7]) 

# plot the Receiver operating characteristic curve (ROC)
plt.figure(figsize=(12,12))
ax1=plt.plot(fpr_list[0], tpr_list[0])
ax2=plt.plot(fpr_list[1], tpr_list[1])
ax3=plt.plot(fpr_list[2], tpr_list[2])
ax4=plt.plot(fpr_list[3], tpr_list[3])
ax4=plt.plot(fpr_list[4], tpr_list[4])
ax4=plt.plot(fpr_list[5], tpr_list[5])
ax4=plt.plot(fpr_list[6], tpr_list[6])
ax4=plt.plot(fpr_list[7], tpr_list[7])
plt.legend(['Logistic Regression','SVC Polynomial','SVC Radial','SVC Linear','Naive bayes','KNN','Neural Networks','Decision Tree Classifier'],loc='lower right')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0]) ; plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')

# plot the AUC Bar Chart 
plt.figure(figsize=(7,4))
y_pos = np.arange(len(auc_list))
plot_bar = plt.barh(y_pos,auc_list,align='center')

# Check where is the make in the list
max_index = auc_list.index(max(auc_list))
min_index = auc_list.index(min(auc_list))
plot_bar[max_index].set_color('lime')
plot_bar[min_index].set_color('red')

plt.xlim(.6)
plt.yticks(range(len(models_names)),models_names)
plt.xlabel('AUC of the model')
plt.ylabel('Models used')
plt.title('Ranking of the model AUC');

################## Recursive Feature Elimination ##########################

from sklearn.feature_selection import RFE

# =============================================================================

# Create the RFE object for SVC linear
clf = SVC(kernel="linear",C=1, gamma=0.001)
rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
rfe.fit(X, y)
ranking = rfe.ranking_
rank=ranking

# ranking = rfe.ranking_.reshape((4,4))
# plt.matshow(ranking, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking with RFE on SVC Linear Fine-Tuned")
# plt.show()

##### Logistic regression #####
# Create the RFE object
clf_2 = LogisticRegression(C=1)
rfe_2 = RFE(estimator=clf_2, n_features_to_select=1, step=1)
rfe_2.fit(X, y)
ranking_2 = rfe_2.ranking_
rank_2=ranking_2

# ranking_2 = rfe_2.ranking_.reshape((4,4))
# plt.matshow(ranking_2, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking with Logistic Regression Fine-Tune")
# plt.show()

# Plot the results
plt.figure(figsize=(12,6))
y_pos = np.arange(len(df.columns))
plt.barh(y_pos,rank)
plt.yticks(range(len(df.columns)),df.columns)
plt.xlabel('Importance of the feature in the SVC Linear')
plt.ylabel('Features')
plt.title('Importance of the features: Features Ranking (1 is best)');

plt.figure(figsize=(12,6))
y_pos = np.arange(len(df.columns))
plt.barh(y_pos,rank_2)
plt.yticks(range(len(df.columns)),df.columns)
plt.xlabel('Importance of the feature in the Logistic Regression')
plt.ylabel('Features')
plt.title('Importance of the features: Features Ranking (1 is best)');

# =============================================================================

##########################################################################

#STRATIFIED K-FOLD Cross Validation

#We create here a new dataframe with the original data (without any rebalancing)
df = pd.read_csv(your_path)

# Drop the label column in the main dataframe
y = df.Default
df= df.drop('Default', axis=1) 

# Standardize the data in a new DF
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()
scaled_data = Scaler.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data , columns=df.columns)
X=df_scaled.values
auc_list_stratified =  []

#We define here the number of splits we wank to make 
kf = StratifiedKFold(n_splits=150)
kf.get_n_splits(X)

#Let's apply the K-fold strategy to our model 
for model in models:
    auc_average=[]
    for train, test in kf.split(X, y):
        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        model.fit(X_train, y_train)
        y_score = model.decision_function(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        auc_average.append(auc(fpr,tpr))
    auc_list_stratified.append(sum(auc_average) / len(auc_average))

#As we cannot use "decision_function" for the following model, we have to treat each case separately
#Naive Bayes
model11 = GaussianNB()
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
model11= model11.fit(X_train,y_train)
model_11_score = model11.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, model_11_score[:, 1])
auc_average.append(auc(fpr,tpr))
auc_list_stratified.append(sum(auc_average) / len(auc_average))
   
#KNN 
model12  = KNeighborsClassifier(n_neighbors=24)
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
model12= model12.fit(X_train,y_train)
model_12_score = model12.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, model_12_score[:, 1])
auc_average.append(auc(fpr,tpr))
auc_list_stratified.append(sum(auc_average) / len(auc_average))

#Neural Network
model13  = MLPClassifier(activation='relu',epsilon=0.001,tol=0.01, hidden_layer_sizes = (128,))
model13= model13.fit(X_train,y_train)
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
model_13_score = model13.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, model_13_score[:, 1])
auc_average.append(auc(fpr,tpr))
auc_list_stratified.append(sum(auc_average) / len(auc_average))

#Decision Tree
model7= model7.fit(X_train,y_train)
model7 = model7.fit(X_train,y_train)
X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]
model_7_score = model7.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, model_7_score[:, 1])
auc_average.append(auc(fpr,tpr))
auc_list_stratified.append(sum(auc_average) / len(auc_average))

# plot the AUC Bar Chart 
plt.figure(figsize=(7,4))
y_pos = np.arange(len(models_names))
plot_bar = plt.barh(y_pos,auc_list_stratified,align='center')

# Check where is the make in the list
max_index = auc_list_stratified.index(max(auc_list_stratified))
min_index = auc_list_stratified.index(min(auc_list_stratified))
plot_bar[max_index].set_color('lime')
plot_bar[min_index].set_color('red')

plt.xlim(.6)
plt.yticks(range(len(models_names)),models_names)
plt.xlabel('AUC of the model')
plt.ylabel('Models used')
plt.title('Ranking of the model AUC using a Stratified K-fold strategy');