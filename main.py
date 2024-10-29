import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from keras.layers import Input, LSTM, Dense
from keras.models import Model
import matplotlib.pyplot as plt

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

math_df = pd.read_csv("/Users/jimmullen/PycharmProjects/james_mullen_midtermproj/finaltermproj/math_df/student-mat.csv", sep=';')

def assign_failure(G3):
    try:
        G3 = pd.to_numeric(G3, errors='coerce')
        if 0 < G3 < 9:
            return 'F'
        else:
            return 'Not Failed'
    except (TypeError, ValueError):
        return 'Not Failed'

math_df['failure'] = math_df['G3'].apply(assign_failure)
math_df.drop(columns=['G3'], inplace=True)

y = math_df['failure']
X = math_df.drop(columns=['failure'])

X = pd.get_dummies(X, drop_first=True)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Decision Tree Classifier
my_decision_tree = DecisionTreeClassifier(max_depth=10, random_state=100, criterion="entropy")
my_decision_tree.fit(X_train, y_train)

print('Accuracy for training data: {:.3f}'.format(my_decision_tree.score(X_train, y_train)))
print('Accuracy for test data: {:.3f}'.format(my_decision_tree.score(X_test, y_test)))

# Feature Importance
my_array = my_decision_tree.feature_importances_
max_index = np.argmax(my_array)
second_max_index = np.argsort(my_array)[-2]

feature_importances = dict(zip(math_df.columns[:-1], my_decision_tree.feature_importances_))
feature_importances = dict(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True))

most_important_feature = max(feature_importances, key=feature_importances.get)
second_most_important_feature = sorted(feature_importances, key=feature_importances.get, reverse=True)[1]

print(f'Most important features:', most_important_feature)
print(f'Second most important features:', {second_most_important_feature})

# Scatter Plot with Color Explanation
colors = np.where(y_train == 'F', 'red', 'blue')
plt.scatter(X_train[:, max_index], X_train[:, second_max_index], c=colors, cmap='coolwarm')
plt.xlabel(most_important_feature)
plt.ylabel(second_most_important_feature)
plt.title("Decision Surface of Decision Tree Classifier")

# Add color legend
red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='F')
blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Not Failed')
plt.legend(handles=[red_patch, blue_patch], title='Class', loc='best')

plt.savefig("scatter_plot_with_color_legend.png")
plt.show()

print("Red points in the scatter plot represent students predicted to fail (class 'F')")
print("Blue points in the scatter plot represent students predicted as not failing (class 'Not Failed')")

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_predictions = rf_classifier.predict(X_test)
rf_confusion_matrix = confusion_matrix(y_test, rf_predictions)

# SVM Classifier
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
svm_predictions = svm_classifier.predict(X_test)
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)

print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)
print("SVM Confusion Matrix:")
print(svm_confusion_matrix)

# LSTM Model
label_encoder = LabelEncoder()
y_train_lstm = label_encoder.fit_transform(y_train)
y_test_lstm = label_encoder.transform(y_test)

X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

inputs = Input(shape=(1, X_train_lstm.shape[2]))
lstm_layer = LSTM(128)(inputs)
outputs = Dense(len(label_encoder.classes_), activation='softmax')(lstm_layer)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

# 10-Fold Cross-Validation
classifiers = [('Random Forest', rf_classifier), ('SVM', svm_classifier)]

for name, classifier in classifiers:
    cv_scores = cross_val_score(classifier, X, y, cv=10)
    print(f'{name} 10-fold cross-validation average accuracy: {cv_scores.mean():.3f}')

# ROC Curves
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--')

classifiers = [('Random Forest', rf_classifier), ('SVM', svm_classifier)]

for name, model in classifiers:
    if name == 'SVM':
        y_score = model.decision_function(X_test)
    else:
        y_score = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_score, pos_label='F')
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f}')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()
plt.show()
plt.savefig("roc_curves.png")
plt.close()
