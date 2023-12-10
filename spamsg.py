import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Load the SMS dataset
# Try reading the CSV file with different encodings
encodings = ['utf-8', 'latin-1']  # You can add more encodings to the list

for encoding in encodings:
    try:
        df = pd.read_csv('spam.csv', encoding=encoding)
        # If successful, break out of the loop
        break
    except UnicodeDecodeError:
        print(f"Failed to decode using {encoding} encoding. Trying the next one.")



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['v1'], df['v2'], test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Transform the text data into numerical features
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
print(X_train)
print(X_test)

# Train a Linear Support Vector Classifier
classifier = LinearSVC()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
