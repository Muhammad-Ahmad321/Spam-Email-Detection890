import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


import os

# Load the data
data_path = os.path.join(os.path.dirname(__file__), 'spam_data.csv')
data = pd.read_csv(data_path)

# Load the data
import pandas as pd
data = pd.read_csv('spam_data.csv')

# Preprocess data
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
X_train, X_test, y_train, y_test = train_test_split(data.Message, data.Spam, test_size=0.25)

# Create the model pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Train the model
clf.fit(X_train, y_train)

# Streamlit app
st.title("Spam Email Detection App")

# Input email text
input_email = st.text_area("Enter the email content:")

# Button for prediction
if st.button('Predict'):
    if input_email:
        prediction = clf.predict([input_email])
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        st.write(f"The email is: **{result}**")
    else:
        st.write("Please enter some text to classify.")

# Model accuracy
if st.checkbox("Show model accuracy"):
    accuracy = clf.score(X_test, y_test)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
