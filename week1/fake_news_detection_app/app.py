import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load('model/fake_news_model.pkl')
vectorizer = joblib.load('vectorizer/tfidf_vectorizer.pkl')

st.set_page_config(page_title="📰 Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Enter a news article below, and the model will predict whether it is **REAL** or **FAKE**.")

# Input area
user_input = st.text_area("📝 Enter News Text:", height=250)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Transform input using the saved TF-IDF vectorizer
        input_vector = vectorizer.transform([user_input])

        # Predict using the saved model
        prediction = model.predict(input_vector)[0]

        # Map numeric label to string
        label = "REAL" if prediction == 1 else "FAKE"

        # Display result
        if label == "REAL":
            st.success("✅ This news appears to be REAL.")
        else:
            st.error("⚠️ This news appears to be FAKE.")
