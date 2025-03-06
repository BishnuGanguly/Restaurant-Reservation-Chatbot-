import torch
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained('models/bert_model')
tokenizer = BertTokenizer.from_pretrained('models/bert_model')

def predict_intent(sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label_idx = torch.argmax(outputs.logits).item()

    label_mapping = {
        0: 'Check Menu', 1: 'Check Payment Options', 2: 'Farewell', 3: 'Greeting',
        4: 'Irrelevant', 5: 'Make Reservation', 6: 'Modify Reservation', 7: 'Place Order'
    }
    return label_mapping[predicted_label_idx]

# Example usage
sentence = "friday evening"
print("Predicted label:", predict_intent(sentence))
