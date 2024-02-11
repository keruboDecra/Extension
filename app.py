
import streamlit as st

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

# Function to extract text from highlighted input using JavaScript
def extract_highlighted_text():
    # Use JavaScript to get the selected text on the webpage
    highlighted_text = st.text_input("Highlight text on the webpage and click Classify")

    # Hypothetical function to extract the text from highlighted input
    # This is just a placeholder, adapt it to your needs
    extracted_text = highlighted_text.replace('<highlight>', '').replace('</highlight>', '')
    
    return extracted_text
def classify_highlighted_input(highlighted_input):
    try:
        # Extract the text from highlighted input
        user_input = extract_highlighted_text(highlighted_input)

        # Preprocess and classify the user input
        binary_prediction, offending_words = binary_cyberbullying_detection(user_input)
        multi_class_label, decision_function_values = multi_class_cyberbullying_detection(user_input)

        # Return results
        return {
            "user_input": user_input,
            "binary_prediction": binary_prediction,
            "offending_words": offending_words,
            "multi_class_label": multi_class_label,
            "decision_function_values": decision_function_values
        }
    except Exception as e:
        st.error(f"Error in classify_highlighted_input: {e}")
        print(f"Error in classify_highlighted_input: {e}")
        return None
# Streamlit App
def main():
    st.title("Cyberbullying Detection App")

    # Highlighted text input
    highlighted_input = extract_highlighted_text()

    # Classify button
    if st.button("Classify"):
        # Classify highlighted input
        classification_results = classify_highlighted_input(highlighted_input)

        if classification_results is not None:
            # Display results
            st.subheader("Results:")
            st.write("User Input:", classification_results["user_input"])
            st.write("Binary Prediction:", classification_results["binary_prediction"])
            st.write("Offending Words:", classification_results["offending_words"])
            st.write("Multi-class Label:", classification_results["multi_class_label"])
            st.write("Decision Function Values:", classification_results["decision_function_values"])
        else:
            st.error("Error in classification. Please try again.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
