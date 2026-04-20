import os
import json
import datetime
import csv
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Fix SSL + nltk
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents
file_path = os.path.join(os.getcwd(), "intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Train model
vectorizer = TfidfVectorizer()
clf = LogisticRegression(max_iter=10000)

tags = []
patterns = []

for intent in intents:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

x = vectorizer.fit_transform(patterns)
clf.fit(x, tags)

# Chatbot function
def chatbot(text):
    x_test = vectorizer.transform([text])
    tag = clf.predict(x_test)[0]

    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand that."

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# CSV setup
if not os.path.exists("chat_log.csv"):
    with open("chat_log.csv", "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["User Input", "Chatbot Response", "Timestamp"])

# UI
st.title("🤖 NLP Chatbot")

menu = ["Home", "Conversation History", "About"]
choice = st.sidebar.selectbox("Menu", menu)

# ------------------ HOME ------------------
if choice == "Home":

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input (modern Streamlit way)
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get bot response
        response = chatbot(user_input)

        # Store bot response
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display immediately
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            st.write(response)

        # Save to CSV
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("chat_log.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([user_input, response, timestamp])

# ------------------ HISTORY ------------------
elif choice == "Conversation History":
    st.header("📜 Conversation History")

    if os.path.exists("chat_log.csv"):
        with open("chat_log.csv", "r", encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)

            for row in reader:
                st.write(f"**User:** {row[0]}")
                st.write(f"**Bot:** {row[1]}")
                st.write(f"_Time:_ {row[2]}")
                st.markdown("---")
    else:
        st.write("No history found.")

# ------------------ ABOUT ------------------
elif choice == "About":
        st.write("The goal of this project is to create a chatbot that can understand and respond to user input based on intents. The chatbot is built using Natural Language Processing (NLP) library and Logistic Regression, to extract the intents and entities from user input. The chatbot is built using Streamlit, a Python library for building interactive web applications.")

        st.subheader("Project Overview:")

        st.write("""
        The project is divided into two parts:
        1. NLP techniques and Logistic Regression algorithm is used to train the chatbot on labeled intents and entities.
        2. For building the Chatbot interface, Streamlit web framework is used to build a web-based chatbot interface. The interface allows users to input text and receive responses from the chatbot.
        """)

        st.subheader("Dataset:")

        st.write("""
        The dataset used in this project is a collection of labelled intents and entities. The data is stored in a list.
        - Intents: The intent of the user input (e.g. "greeting", "budget", "about")
        - Entities: The entities extracted from user input (e.g. "Hi", "How do I create a budget?", "What is your purpose?")
        - Text: The user input text.
        """)

        st.subheader("Streamlit Chatbot Interface:")

        st.write("The chatbot interface is built using Streamlit. The interface includes a text input box for users to input their text and a chat window to display the chatbot's responses. The interface uses the trained model to generate responses to user input.")

        st.subheader("Conclusion:")
    

        st.write("In this project, a chatbot is built that can understand and respond to user input based on intents. The chatbot was trained using NLP and Logistic Regression, and the interface was built using Streamlit. This project can be extended by adding more data, using more sophisticated NLP techniques, deep learning algorithms.")

if __name__ == '__main__':
    main()

