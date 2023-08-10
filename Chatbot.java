"""
Project: Chatbot that extracts my Whatsapp conversation with my brother and then trains a logistic regression model.
It then interacts with Whatsapp through a webhook and uses Flask to respond to messages.
Created by LAYANNE el ASSAAD
"""

import re
import requests
import numpy as np
import spacy
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from spacy.matcher import PhraseMatcher
from spacy import doc
from spacy.lang.en import English

# Step 1: Preprocess WhatsApp Conversations

# Load conversation
with open('kamelchat.txt', 'r', encoding='utf-8') as file:
    conversations = file.readlines()

preprocessed_conversations = []

timestamp_pattern = re.compile(r'\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2} [APap][Mm] -')

current_conversation = []
for line in conversations:
    if re.match(timestamp_pattern, line):
        if current_conversation:
            preprocessed_conversations.append(current_conversation)
        current_conversation = []
    else:
        current_conversation.append(line.strip())

if current_conversation:
    preprocessed_conversations.append(current_conversation)

# Clean and preprocess conversation
def preprocess_conversation(conversation):
    cleaned_conversation = []
    for message in conversation:
        cleaned_message = re.sub(r'<Media omitted>', '', message)
        cleaned_message = cleaned_message.lower()
        cleaned_conversation.append(cleaned_message)
    return cleaned_conversation

preprocessed_data = [preprocess_conversation(conversation) for conversation in preprocessed_conversations]

# Save preprocessed data to a new file
with open('preprocessed_kamelchat.txt', 'w', encoding='utf-8') as file:
    for conversation in preprocessed_data:
        file.write('\n'.join(conversation) + '\n\n')


# Step 2: Convert Preprocessed Data into ML Suitable

# Load the pre-trained language model
nlp = spacy.load("en_core_web_sm")

# Load and process preprocessed WhatsApp conversation data
with open('preprocessed_kamelchat.txt', 'r', encoding='utf-8') as file:
    conversations = file.read().split('\n\n')

# Process each message in the conversations and extract word embeddings
embeddings_by_conversation = []
for conversation in conversations:
    messages = conversation.split('\n')
    message_embeddings = []
    for message in messages:
        doc = nlp(message)
        message_embeddings.append(doc.vector)
    embeddings_by_conversation.append(message_embeddings)

# Step 3: Use Conversation Embeddings for Machine Learning

# Convert the list of embeddings into a NumPy array
embedding_matrix = np.array(embeddings_by_conversation)

# Assuming you have 'labels' corresponding to each conversation
# This is just an example, replace it with your actual labels
labels = [0, 1, 0, 1]  # Example labels

# Step 4: Train a Logistic Regression Model

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embedding_matrix, labels, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train.reshape(-1, X_train.shape[-1]), y_train)

# Make predictions on the test set
y_pred = model.predict(X_test.reshape(-1, X_test.shape[-1]))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", accuracy)


#Interacting with WhatsApp:
from flask import Flask, request, jsonify

app = Flask(__name__)

# Your webhook URL, e.g., https://yourserver.com/webhook
WEBHOOK_URL = "/webhook"

@app.route(WEBHOOK_URL, methods=["POST"])
def receive_message():
    data = request.json

    # Extract the message and sender from the received data
    message = data.get("message", "")
    sender = data.get("sender", "")

    # Process the message (implement your chatbot logic here)
    response = generate_response(message)

    # Send the response back to the user
    send_message(sender, response)

    return jsonify({"status": "success"})

def generate_response(message):
    # Implement your chatbot logic to generate a response
    # For simplicity, let's use a rule-based system

    # Convert the user message to lowercase for easier comparison
    cleaned_message = message.lower()

    # Define some sample rules and responses
    rules = {
        "hello": "Hi there! How can I assist you?",
        "how are you": "I'm just a chatbot, but I'm here to help!",
        "bye": "Goodbye! Have a great day!",
        # Add more rules and responses here
    }

    # Check if the cleaned message matches any rule
    response = rules.get(cleaned_message, "I'm sorry, I don't understand that.")

    return response


def send_message(recipient, message):
    # Replace these with your actual WhatsApp Business API credentials
    WHATSAPP_API_URL = "https://api.example.com/whatsapp/send"
    API_KEY = "your_api_key"
    
    # Construct the payload for sending the message
    payload = {
        "recipient": recipient,
        "message": message,
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    
    try:
        # Make a POST request to the WhatsApp API
        response = requests.post(WHATSAPP_API_URL, json=payload, headers=headers)
        response_data = response.json()
        
        if response.status_code == 200 and response_data.get("status") == "success":
            print("Message sent successfully")
        else:
            print("Message sending failed")
    
    except requests.exceptions.RequestException as e:
        print("An error occurred:", str(e))

# Example usage
recipient_number = "+1234567890"  # Replace with the actual recipient's number
message_to_send = "This is a sample response."

send_message(recipient_number, message_to_send)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
