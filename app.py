import streamlit as st
from chatbot import Chatbot


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="HelpDesk AI",
    page_icon="🤖",
    layout="centered",
)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 HelpDesk AI")

st.caption(
    "Seq2Seq LSTM Chatbot with Bahdanau Attention"
)


# --------------------------------------------------
# LOAD TRAINED CHATBOT
# --------------------------------------------------

@st.cache_resource
def load_bot():
    return Chatbot()


try:
    bot = load_bot()

except Exception as exc:
    st.error("The trained model is not available.")
    st.code(
        "python train.py"
    )
    st.exception(exc)
    st.stop()


# --------------------------------------------------
# SIMPLE HELP DESK FALLBACK RESPONSES
# --------------------------------------------------

def helpdesk_response(message):

    text = message.lower().strip()

    if text in ["hi", "hello", "hey", "hii", "good morning",
                "good afternoon", "good evening"]:
        return "Hello! 👋 Welcome to HelpDesk AI. How can I help you today?"

    if "thank" in text:
        return "You're welcome! 😊 I'm happy to help."

    if "password" in text:
        return "If you forgot your password, please use the password reset option or contact the system administrator."

    if "login" in text or "log in" in text:
        return "Please check your username and password. If you still cannot log in, contact the HelpDesk team."

    if "account" in text:
        return "I can help with basic account-related questions. Please describe the issue."

    if "internet" in text or "wifi" in text or "wi-fi" in text:
        return "Please check your internet or Wi-Fi connection. If the problem continues, contact the technical support team."

    if "computer" in text or "laptop" in text:
        return "Please describe the computer problem you are experiencing, and I will try to help."

    if "error" in text or "problem" in text or "issue" in text:
        return "I'm sorry you're facing an issue. Please provide a few details about the problem."

    if "bye" in text or "goodbye" in text:
        return "Goodbye! 👋 Have a great day."

    # Use trained Seq2Seq model for other questions
    answer = bot.reply(message)

    # Prevent poor one-word generated responses
    if not answer or len(answer.strip().split()) <= 1:
        return "I'm not sure about that yet. Please provide more details about your HelpDesk issue."

    return answer


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for role, message in st.session_state.messages:

    with st.chat_message(role):
        st.write(message)


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

prompt = st.chat_input(
    "Type your HelpDesk question..."
)


if prompt:

    # Display user message
    st.session_state.messages.append(
        ("user", prompt)
    )

    with st.chat_message("user"):
        st.write(prompt)


    # Generate response
    answer = helpdesk_response(prompt)


    # Display bot response
    st.session_state.messages.append(
        ("assistant", answer)
    )

    with st.chat_message("assistant"):
        st.write(answer)