import torch
from transformers import BertForSequenceClassification, BertTokenizer
import random

# Load the pre-trained BERT model and tokenizer
model_path = 'bert_model'  
tokenizer_path = 'bert_model'  
model = BertForSequenceClassification.from_pretrained('/content/drive/MyDrive/chatbot')
tokenizer = BertTokenizer.from_pretrained('/content/drive/MyDrive/chatbot')

# Define the intents and corresponding responses
intents_responses = {
    "Check Menu": [
        "Sure! Here is our menu.",
        "Take a look at our menu.",
        "What would you like to order?"
    ],
    "Check Payment Options": [
        "Please pay using this link.",
        "you can do payment in this link"
    ],
    "Farewell": [
        "Thank you for using our service. Have a great day!",
        "It was nice assisting you. Have a wonderful day!"
    ],
    "Greeting": [
        "Hello! Welcome to our restaurant reservation system.",
        "Hi there! How can I assist you today?"
    ],
    "Irrelevant": [
        "Sorry, I'm not sure how to help with that.",
        "I'm not trained to answer that. Can I assist you with something else?"
    ],
    "Make Reservation": [
         "When would you like to make the reservation?",
         "Sure! Let me know the date and time for the reservation."
    ],
    "Modify Reservation": [
        "Please provide the details of the reservation you want to modify.",
        "Sure! Let me know what changes you'd like to make to your reservation."
    ],
    "Place Order": [
        "What would you like to order?",
        "Sure! What can I get for you?"
    ]
}

prev_intent = None  # Track the previous intent

def generate_response(intent):
    if intent in intents_responses:
        return random.choice(intents_responses[intent])
    else:
        return random.choice(intents_responses["Irrelevant"])

def classify_intent(user_query):
    inputs = tokenizer(user_query, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predicted_label_idx = torch.argmax(outputs.logits).item()

    label_mapping = {
        0: 'Check Menu',
        1: 'Check Payment Options',
        2: 'Farewell',
        3: 'Greeting',
        4: 'Irrelevant',
        5: 'Make Reservation',
        6: 'Modify Reservation',
        7: 'Place Order'
    }
    predicted_label = label_mapping[predicted_label_idx]

    return predicted_label

def chatbot():
    global prev_intent

    print("Chatbot: Hello! Welcome to our restaurant reservation system.")
    user_query = input("User: ")  # Get user input

    while True:
        intent = classify_intent(user_query)

        if intent == "Farewell":
            print("Chatbot:", random.choice(intents_responses["Farewell"]))
            break
        elif prev_intent == intent:
            # Handle consecutive intents
            if intent == "Make Reservation" or intent == "Check Menu":
                print("Chatbot: What will be your mode of payment?")
                user_query = input("User: ")  # Get user input
                prev_intent = classify_intent(user_query)  # Update prev_intent
                continue

        response = generate_response(intent)
        print("Chatbot:", response)
        prev_intent = intent  # Update prev_intent
        user_query = input("User: ")  # Get user input

# Example usage
chatbot()

