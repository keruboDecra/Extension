from PIL import Image
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib
import streamlit as st

# Load the SGD classifier, TF-IDF vectorizer, and label encoder
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

# Load the logo image
logo = Image.open('logo.png')

# Global variables
session_state = {
    'user_input': '',
    'chrome_extension_message': None
}

# Function to clean and preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+|[^A-Za-z\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Function for binary cyberbullying detection
def binary_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction using the loaded pipeline
        prediction = model_pipeline.predict([preprocessed_text])

        # Check for offensive words
        with open('en.txt', 'r') as f:
            offensive_words = [line.strip() for line in f]

        offending_words = [word for word in preprocessed_text.split() if word in offensive_words]

        return prediction[0], offending_words
    except Exception as e:
        st.error(f"Error in binary_cyberbullying_detection: {e}")
        return None, None

# Function for multi-class cyberbullying detection
def multi_class_cyberbullying_detection(text):
    try:
        # Preprocess the input text
        preprocessed_text = preprocess_text(text)

        # Make prediction
        decision_function_values = sgd_classifier.decision_function([preprocessed_text])[0]

        # Get the predicted class index
        predicted_class_index = np.argmax(decision_function_values)

        # Get the predicted class label using the label encoder
        predicted_class_label = label_encoder.inverse_transform([predicted_class_index])[0]

        return predicted_class_label, decision_function_values
    except Exception as e:
        st.error(f"Error in multi_class_cyberbullying_detection: {e}")
        return None

# Function to classify user input
def classify_user_input(user_input):
    try:
        # Preprocess the user input
        preprocessed_text = preprocess_text(user_input)

        # Binary Cyberbullying Detection
        binary_prediction, offending_words = binary_cyberbullying_detection(preprocessed_text)

        # Multi-class Cyberbullying Detection
        multi_class_label, decision_function_values = multi_class_cyberbullying_detection(preprocessed_text)

        return {
            'Binary Prediction': binary_prediction,
            'Offending Words': offending_words,
            'Multi-class Prediction': multi_class_label,
            'Decision Function Values': decision_function_values
        }
    except Exception as e:
        st.error(f"Error in classify_user_input: {e}")
        return None

# Main function to start the Streamlit app
def main():
    st.title("Cyberbullying Detection App")
    st.image(logo, caption='Logo', use_column_width=True)

    user_input = st.text_area("Enter text for classification:", height=100)
    if st.button("Classify"):
        if user_input:
            classification_result = classify_user_input(user_input)
            st.subheader("Classification Result:")
            st.write(f"Binary Prediction: {classification_result['Binary Prediction']}")
            st.write(f"Offending Words: {classification_result['Offending Words']}")
            st.write(f"Multi-class Prediction: {classification_result['Multi-class Prediction']}")
            st.write(f"Decision Function Values: {classification_result['Decision Function Values']}")

if __name__ == '__main__':
    main()
