import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = tf.keras.models.load_model("news_classifier.h5")

# Recompile the model (to suppress the warning)
model.compile(
    optimizer="adam", 
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
)

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

CLASS_NAMES = ["Business", "Tech", "Politics", "Sports", "Entertainment"]

def preprocess_text(text):
    MAXLEN = 120
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAXLEN, padding='post', truncating='post')
    return padded

def predict_news_category(text):
    processed_text = preprocess_text(text)
    prob = model.predict(processed_text)[0]
    predicted_class = CLASS_NAMES[np.argmax(prob)]
    confidence = np.max(prob)
    return predicted_class, confidence, prob

# Streamlit UI
st.title("News Category Classifier 📰")
user_input = st.text_area("Paste the news article here:", height=200)

if st.button("Classify"):
    if user_input:
        category, confidence, probs = predict_news_category(user_input)
        st.success(f"**Predicted Category:** {category}  \n**Confidence:** {confidence:.2f}")
        st.bar_chart({CLASS_NAMES[i]: float(probs[i]) for i in range(5)})
    else:
        st.warning("Please enter some text!")