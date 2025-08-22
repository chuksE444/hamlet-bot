import streamlit as st
from chatbot import preprocess, chatbot

# Preprocess the file once
sentences, cleaned_sentences = preprocess("hamlet.txt")

# Streamlit app
st.title("Hamlet Chatbot")
st.write("Ask me anything about Hamlet!")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:")

if user_input:
    response = chatbot(user_input, sentences, cleaned_sentences)
    # Store conversation
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", response))

# Display conversation
for speaker, text in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")
