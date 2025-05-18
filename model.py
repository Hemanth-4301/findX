# train_and_save_model.py
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier

# Load and prepare data
dataset = pd.read_csv('datasets/Training.csv')
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=20)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Save model and encoder
with open('knn_model.pkl', 'wb') as f:
    pickle.dump(knn, f)
    
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model trained and saved successfully!")