import streamlit as st
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import joblib

# Load the SGD classifier, TF-IDF vectorizer, and label encoder
sgd_classifier = joblib.load('sgd_classifier_model.joblib')
label_encoder = joblib.load('label_encoder.joblib')

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
        prediction = sgd_classifier.predict([preprocessed_text])

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

# Streamlit App
def main():
    st.title("Cyberbullying Detection App")

    # Button to capture highlighted text
    if st.button("Capture Highlighted Text"):
        # Use JavaScript to get the selected text on the webpage
        highlighted_text = st.text_input("Highlighted Text:")
        st.markdown("Highlight text on the webpage and click 'Capture Highlighted Text'.")

        # Classify button
        if st.button("Classify"):
            try:
                # Preprocess and classify the user input
                binary_prediction, offending_words = binary_cyberbullying_detection(highlighted_text)
                multi_class_label, decision_function_values = multi_class_cyberbullying_detection(highlighted_text)

                # Display results
                st.subheader("Results:")
                st.write("User Input:", highlighted_text)
                st.write("Binary Prediction:", binary_prediction)
                st.write("Offending Words:", offending_words)
                st.write("Multi-class Label:", multi_class_label)
                st.write("Decision Function Values:", decision_function_values)
            except Exception as e:
                st.error(f"Error in classification: {e}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
