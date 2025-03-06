import torch
from transformers import BertForSequenceClassification, BertTokenizer
import random

# Load the pre-trained BERT model and tokenizer
MODEL_PATH = "/content/drive/MyDrive/chatbot"

print("Loading model and tokenizer...")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
print("Model loaded successfully!")

# Define the intents and corresponding responses
intents_responses = {
    "Check Menu": [
        "Sure! Here is our menu.",
        "Take a look at our menu.",
        "What would you like to order?"
    ],
    "Check Payment Options": [
        "Please pay using this link.",
        "You can complete your payment here."
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

label_mapping = {
    0: "Check Menu",
    1: "Check Payment Options",
    2: "Farewell",
    3: "Greeting",
    4: "Irrelevant",
    5: "Make Reservation",
    6: "Modify Reservation",
    7: "Place Order"
}

prev_intent = None  # Track the previous intent

def classify_intent(user_query):
    """Classifies user intent using the trained BERT model."""
    inputs = tokenizer(user_query, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)

    predicted_label_idx = torch.argmax(outputs.logits).item()
    return label_mapping.get(predicted_label_idx, "Irrelevant")

def generate_response(intent):
    """Generates a chatbot response based on classified intent."""
    return random.choice(intents_responses.get(intent, intents_responses["Irrelevant"]))

def chatbot():
    """Runs the chatbot in a loop until the user says farewell."""
    global prev_intent

    print("Chatbot: Hello! Welcome to our restaurant reservation system.")
    
    while True:
        user_query = input("User: ")  # Get user input
        intent = classify_intent(user_query)

        if intent == "Farewell":
            print("Chatbot:", generate_response("Farewell"))
            break
        
        # Handle consecutive intents
        if prev_intent == intent and intent in ["Make Reservation", "Check Menu"]:
            print("Chatbot: What will be your mode of payment?")
            user_query = input("User: ")
            prev_intent = classify_intent(user_query)
            continue

        print("Chatbot:", generate_response(intent))
        prev_intent = intent  # Update previous intent

if __name__ == "__main__":
    chatbot()
