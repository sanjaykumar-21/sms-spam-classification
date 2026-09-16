import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")


with open("model.pkl", "rb") as file:
    model = pickle.load(file)


with open("vectorizer.pkl", "rb") as file:
    txt_convert = pickle.load(file)


def clean(text):

    regex = r"[^a-zA-Z]"

    text = re.sub(regex, " ", text)
    text = text.lower()

    tokens = nltk.word_tokenize(text)

    stop_words = set(stopwords.words("english"))

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    lemmatizer = WordNetLemmatizer()

    filtered_lemmatizer = [
        lemmatizer.lemmatize(word)
        for word in filtered_tokens
    ]

    return " ".join(filtered_lemmatizer)


st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="centered"
)


st.title("📱 SMS Spam Classifier")

st.write(
    "Enter an SMS message below to predict whether it is "
    "Ham or Spam."
)


message = st.text_area(
    "Enter your SMS message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)


if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter an SMS message.")

    else:

        cleaned_message = clean(message)

        message_transformed = txt_convert.transform(
            [cleaned_message]
        )

        prediction = model.predict(
            message_transformed
        )[0]

        if prediction == 1:

            st.error("🚨 SPAM MESSAGE")

        else:

            st.success("✅ NORMAL MESSAGE")