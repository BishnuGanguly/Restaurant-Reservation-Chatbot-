import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the trained model and tokenizer
model = BertForSequenceClassification.from_pretrained('/content/drive/MyDrive/chatbot')
tokenizer = BertTokenizer.from_pretrained('/content/drive/MyDrive/chatbot')

# Define the sentence to predict
sentence = "friday evening"

# Tokenize the sentence and convert to input IDs
inputs = tokenizer(sentence, return_tensors='pt')
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Forward pass through the model
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

# Get the predicted label
predicted_label_idx = torch.argmax(outputs.logits).item()

# Map the predicted label index to the corresponding label
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

print("Predicted label:", predicted_label)
