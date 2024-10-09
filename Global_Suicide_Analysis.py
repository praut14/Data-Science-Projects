import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC,SVR
#from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,confusion_matrix,accuracy_score,classification_report, precision_score, recall_score, f1_score
#import the global suicide dataset
df=pd.read_csv('master.csv')
print(df.head())
print(df.info())
print(df.describe())

# Step 1: Data Cleaning and Feature Engineering
df_clean=df.copy()

#remove rows with missing values in essential columns 
df_clean=df_clean.dropna(subset=['suicides_no','population','gdp_per_capita ($)'])

#create a new column 'suicides_per_100k'
df_clean['suicides_per_100k']=(df_clean['suicides_no']/df_clean['population'])*100000

#Step 2: Visulization 
sns.set(style='whitegrid')

# Step 3: Global trend of Suicides per 100k over time 
plt.figure(figsize=(10, 6))
global_trend=df_clean.groupby('year')['suicides_per_100k'].mean().reset_index()
sns.lineplot(x='year',y='suicides_per_100k',data=global_trend,markers=0)
plt.title('Global Trend of Suicides Per 100K (1985-2015)', fontsize=14)
plt.xlabel('Year',fontsize=12)
plt.ylabel('Suicides per 100k',fontsize=12)
plt.grid(True)
plt.show()

# Step 4: Relationship between Suicide rates and GDP per capita
plt.figure(figsize=(10, 6))
sns.scatterplot(x='gdp_per_capita ($)',y='suicides_per_100k',data=df_clean,alpha=0.5)
plt.title(' Suicides rates vs GDP per Capita ', fontsize=14)
plt.xlabel('GDP per Capita ($)',fontsize=12)
plt.ylabel('Suicides per 100k',fontsize=12)
plt.grid(True)
plt.show()

# Step 5: Suicide rates by Age Group
plt.figure(figsize=(10, 6))
sns.boxplot(x='age',y='suicides_per_100k',data=df_clean)
plt.title(' Suicides rates vs Age Group ', fontsize=14)
plt.xlabel('Age Group',fontsize=12)
plt.ylabel('Suicides per 100k',fontsize=12)
plt.grid(True)
plt.show()


# Step 6: Data Preprocessing for Machine Learning: Selecting relevant features for prediction
features = ['year', 'sex', 'age', 'gdp_per_capita ($)', 'population']
target = 'suicides_per_100k'

#Encoding Categorical features like age and sex
label_encoders = {}
for col in ['sex', 'age']:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le

#Step 7 : Train Test split 
X=df_clean[features]
Y=df_clean[target]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# Normalize the features for SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Step 8 : Train the models 
# 1. Linear Regression
lr=LinearRegression()
lr.fit(X_train,Y_train)

#2.Decision Tree
dl=DecisionTreeRegressor(random_state=42)
dl.fit(X_train,Y_train)

#3. Random forest
rf=RandomForestRegressor(random_state=42,n_estimators=100)
rf.fit(X_train,Y_train)

#4. Support Vector Machine
svc=SVR(kernel='linear')
svc.fit(X_train_scaled,Y_train)

# 5. KNN
#KNN=KNeighborsClassifier(n_neighbors=5)
#KNN.fit=(X_train_scaled,Y_train)

#Step 9: Model Evaluation
def evaluate_model(model, X_test, y_test, scaled=False):
    if scaled:
        X_test = X_test_scaled
    y_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test, y_pred)
    mae = mean_absolute_error(Y_test, y_pred)
    rmse = mean_squared_error(Y_test, y_pred, squared=False)
    r2 = r2_score(Y_test, y_pred)
    return mse, mae, rmse, r2

# Step 10:Evaluate all the models 
linear_metrics = evaluate_model(lr, X_test, Y_test)
decision_metrics = evaluate_model(dl, X_test, Y_test)
random_metrics = evaluate_model(rf, X_test, Y_test)
support_vector_metrics = evaluate_model(svc, X_test_scaled, Y_test, scaled=True)
#k_nearest_metrics = evaluate_model(KNN, X_test_scaled, Y_test, scaled=True)

# Step 11: Print Regression Metrics
def print_metrics(name, metrics):
    print(f"{name} - MSE: {metrics[0]:.2f}, MAE: {metrics[1]:.2f}, RMSE: {metrics[2]:.2f}, R-squared: {metrics[3]:.2f}")

print_metrics("Linear Regression",linear_metrics)
print_metrics("Decision Tree",decision_metrics )
print_metrics("Random Forest",random_metrics )
print_metrics("Support Vector Machine (SVR)",support_vector_metrics)
#print_metrics("K-Nearest Neighbors (KNN)",k_nearest_metrics)


# Step 12: Binary Classification for Confusion Matrix
# Define a threshold to classify suicide rate (for example, if suicides_per_100k > 20, classify as high suicide rate)
threshold = 20
Y_train_class = (Y_train > threshold).astype(int)
Y_test_class = (Y_test > threshold).astype(int)

# Train a classification model using Random Forest for binary classification
forest_class = RandomForestRegressor(random_state=42, n_estimators=100)
forest_class.fit(X_train, Y_train_class)

# SVM Classifier
svc = SVC(kernel='linear', random_state=42)
svc.fit(X_train_scaled, Y_train_class)

# KNN Classifier
#knn_class = KNeighborsClassifier(n_neighbors=5)
#knn_class.fit(X_train_scaled, Y_train_class)

# Step 8: Predict class labels and evaluate
# Linear regression
y_pred_lin_reg = lr.predict(X_test)
y_pred_class_lin_reg = (y_pred_lin_reg > threshold).astype(int)

# Decision Tree
y_pred_decision_tree = dl.predict(X_test)
y_pred_class_decision_tree = (y_pred_decision_tree > threshold).astype(int)

# Random Forest
y_pred_class_forest = (forest_class.predict(X_test) > 0.5).astype(int)

# SVM Classifier
y_pred_class_svm = svc.predict(X_test_scaled)

# KNN Classifier
#y_pred_class_knn = knn_class.predict(X_test_scaled)

# Step 13:  Confusion Matrix and Accuracy Score for Classification
def print_classification_metrics(Y_test_class, Y_pred_class, model_name):
    conf_matrix = confusion_matrix(Y_test_class, Y_pred_class)
    acc_score = accuracy_score(Y_test_class, Y_pred_class)
    print(f"\n{model_name} - Confusion Matrix:")
    print(conf_matrix)
    print(f"{model_name} - Accuracy Score: {acc_score:.2f}")

# Print classification metrics for all models
print_classification_metrics(Y_test_class, y_pred_class_forest, "Random Forest Classifier")
print_classification_metrics(Y_test_class, y_pred_class_svm, "SVM Classifier")
#print_classification_metrics(Y_test_class, y_pred_class_knn, "KNN Classifier")


# Function to plot the confusion matrix
def plot_confusion_matrix(Y_test_class, Y_pred_class, model_name):
    conf_matrix = confusion_matrix(Y_test_class, Y_pred_class)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Function to print and visualize accuracy score
def print_and_visualize_accuracy(Y_test_class, Y_pred_class, model_name):
    acc_score = accuracy_score(Y_test_class, Y_pred_class)
    print(f"{model_name} - Accuracy Score: {acc_score:.2f}")
    # Plot accuracy score
    plt.figure(figsize=(6, 4))
    sns.barplot(x=[model_name], y=[acc_score])
    plt.title(f'{model_name} - Accuracy Score')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Ensuring the plot has a consistent range
    plt.show()

    # Linear Regressor
plot_confusion_matrix(Y_test_class, y_pred_class_lin_reg, "Linear Regression (Classification)")
print_and_visualize_accuracy(Y_test_class, y_pred_class_lin_reg, "Linear Regression (Classification)")

   # Decision Tree Classifier
plot_confusion_matrix(Y_test_class, y_pred_class_decision_tree, "Decision Tree (Classification)")
print_and_visualize_accuracy(Y_test_class, y_pred_class_decision_tree, "Decision Tree (Classification)")

    # Random Forest Classifier
plot_confusion_matrix(Y_test_class, y_pred_class_forest, "Random Forest Classifier")
print_and_visualize_accuracy(Y_test_class, y_pred_class_forest, "Random Forest Classifier")

# SVM Classifier
plot_confusion_matrix(Y_test_class, y_pred_class_svm, "SVM Classifier")
print_and_visualize_accuracy(Y_test_class, y_pred_class_svm, "SVM Classifier")


def print_classification_report(y_true, y_pred, model_name):
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred))

def calculate_classification_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1

# For Random Forest
precision_forest, recall_forest, f1_forest = calculate_classification_metrics(Y_test_class, y_pred_class_forest)
print_classification_report(Y_test_class, y_pred_class_forest, "Random Forest Classifier")

# For SVM
precision_svm, recall_svm, f1_svm = calculate_classification_metrics(Y_test_class, y_pred_class_svm)
print_classification_report(Y_test_class, y_pred_class_svm, "SVM Classifier")

# For Linear Regression
precision_lin, recall_lin, f1_lin = calculate_classification_metrics(Y_test_class, y_pred_class_lin_reg)
print_classification_report(Y_test_class, y_pred_class_lin_reg, "Linear Regression")

# For Decision Tree
precision_tree, recall_tree, f1_tree = calculate_classification_metrics(Y_test_class, y_pred_class_decision_tree)
print_classification_report(Y_test_class, y_pred_class_decision_tree, "Decision Tree Classifier")

# Collect the metrics for all models
models = ['Random Forest', 'SVM', 'Linear Regression', 'Decision Tree']
precisions = [precision_forest, precision_svm, precision_lin, precision_tree]
recalls = [recall_forest, recall_svm, recall_lin, recall_tree]
f1_scores = [f1_forest, f1_svm, f1_lin, f1_tree]

# Plotting
x = np.arange(len(models))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, precisions, width, label='Precision')
rects2 = ax.bar(x, recalls, width, label='Recall')
rects3 = ax.bar(x + width, f1_scores, width, label='F1 Score')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Models')
ax.set_title('Precision, Recall, and F1 Score by Model')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

fig.tight_layout()
plt.show()
