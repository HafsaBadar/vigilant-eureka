import streamlit as st
import cohere
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the trained Logistic Regression model from the pickle file
with open('youtube_spam_model.pkl', 'rb') as f:
    model, tfidf = pickle.load(f)

# Initialize Cohere client
COHERE_API_KEY = 'pGbElaDbFkEHelKRVKyQG6QoTB14XF4iQ0iOMEqP'
co = cohere.Client(COHERE_API_KEY)

# Predict function using Cohere
def change_content(text):
    response = co.chat(message=text,
                   model="command-r-plus",
                   preamble="You are an expert in removing spam content from youtube comments\
                     and give non spam content my modifying the original one.")
    
    return response.text

# Streamlit app
st.title('YouTube Spam Comments Detector')

st.write('Enter comment to check if it is spam or not.')

# Input content
user_content = st.text_area("Enter comment")

if st.button('Check'):
    if user_content:
        user_content_tfidf = tfidf.transform([user_content])
        # Predict spam or not spam
        prediction = model.predict(user_content_tfidf)

        if prediction[0] == 1:
            st.write('The comment is identified as **spam**.')
            spam_removed_text = change_content(user_content)
            st.write(f'Spam removed comment is: {spam_removed_text}')
        else:
            st.write('This comment is: **not spam**.')
    else:
        st.write('Please enter content to check.')
