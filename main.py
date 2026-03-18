import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df=pd.read_csv(r'C:\mL programs\spamdetection\email.csv')
df.head()

df = df[df['Category'].isin(['ham', 'spam'])]
df.loc[df['Category']=='spam','Category']=0
df.loc[df['Category']=='ham','Category']=1
X = df['Message']
Y = df['Category']
# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Convert text data to numerical features using TF-IDF
feature_extraction=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
X_train_features=feature_extraction.fit_transform(X_train)
X_test_features=feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')


# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train_features, Y_train)    
# Make predictions on the test set
Y_pred = model.predict(X_test_features)
# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the model and the feature extraction object
joblib.dump(model, 'spam_classifier_model.joblib')
joblib.dump(feature_extraction, 'feature_extraction.joblib')
print("Model and feature extraction object saved successfully.")





