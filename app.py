import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Intent-Response Mapping
responses = {
    # Group 1: Order Status & Tracking
    "track_order": "I am redirecting you to our live tracking system. You can see your package's current location there.",
    "check_order_status": "Let me check that for you. I'm opening your order history to show the latest updates.",
    "track_refund": "I'm checking the status of your refund. I'll redirect you to the billing dashboard in a second.",

    # Group 2: Cancellations & Returns
    "cancel_order": "I understand you'd like to cancel. I'm opening the cancellation form; please state your reason there.",
    "return_order": "I'm sorry it didn't work out. I'm generating a return label for you right now.",
    "check_refund_status": "Your refund is being processed. I'm directing you to the status page for details.",

    # Group 3: Changes & Edits
    "edit_order": "I've paused the processing of your order. What would you like to change?",
    "change_shipping_address": "Sure! I'm opening the address update portal. Please make sure the new address is correct.",
    "change_order": "I'm ready to help you modify your order. Please select the items you'd like to change.",

    # Group 4: Account & Technical Support
    "recover_password": "No worries! I'm sending a password reset link to your registered email address.",
    "edit_profile": "I'm opening your profile settings so you can update your personal information.",
    "registration_problems": "I see you're having trouble signing up. I'm connecting you to our technical support team.",

    # Group 5: Payments & Invoices
    "payment_issue": "I've detected a payment error. I'm redirecting you to our secure payment gateway to retry.",
    "get_invoice": "Your invoice is ready! I'm opening the download link for your PDF invoice.",
    "check_payment_methods": "I'm showing you our accepted payment methods, including credit cards and digital wallets.",

    # Group 6: General Interaction
    "greeting": "Hello! I am your AI Support Assistant. How can I help you today?",
    "goodbye": "Thank you for reaching out! Have a wonderful day. Goodbye!",
    "thanks": "You're very welcome! Is there anything else I can assist you with?",

    # Default/Other
    "contact_customer_support": "I'm connecting you to a live representative for further assistance. Please wait a moment.",
    "complaint": "I've noted your complaint and I'm escalating it to our management team immediately."
}

@st.cache_resource
def load_assets():
    model = load_model("models/customer_model.keras")
    with open("models/tfidf.pkl", "rb") as f:
        tf = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label = pickle.load(f)
    return model, tf, label

# Load Model and Encoders
model, tf, label = load_assets()

# UI Setup
st.set_page_config(page_title="AI Customer Support", page_icon="🤖")
st.title("🤖 AI Customer Support Assistant")
st.markdown("Welcome! Type your message below to get help with your order.")
st.divider()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input and Prediction
if prompt := st.chat_input("How can I help you?"):
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prediction Logic
    transformed_text = tf.transform([prompt]).toarray()
    prediction = model.predict(transformed_text)
    result_index = np.argmax(prediction)
    intent = label.inverse_transform([result_index])[0]

    # Get Response from Dictionary
    bot_reply = responses.get(intent, "I understand your request. I'm looking into it right now.")

    # Assistant Message
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    # Save to History
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})