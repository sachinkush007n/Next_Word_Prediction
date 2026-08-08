
import streamlit as st
import pickle
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = load_model("next_word_lstm.h5")


# Load tokenizer
with open("tokenizer.pickle", "rb") as file:
    tokenizer = pickle.load(file)


# Function to predict next word
def predict_next_word(model, tokenizer, text, max_length):

    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_length:
        token_list = token_list[-(max_length - 1):]

    token_list = pad_sequences(
        [token_list],
        maxlen=max_length - 1,
        padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)

    predicted_next_word_index = np.argmax(
        predicted,
        axis=1
    )[0]

    for word, index in tokenizer.word_index.items():
        if index == predicted_next_word_index:
            return word

    return None


# Streamlit App
st.title("Next Word Prediction using LSTM with Early Stopping")

input_text = st.text_input(
    "Enter the sequence of words",
    "Pizza, the delectable"
)


if st.button("Next Word"):

    max_seq_length = model.input_shape[1] + 1

    next_word = predict_next_word(
        model,
        tokenizer,
        input_text,
        max_seq_length
    )

    if next_word:
        st.success(f"Next word: {next_word}")
    else:
        st.warning("Unable to predict the next word.")

