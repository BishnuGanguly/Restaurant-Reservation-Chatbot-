!pip install transformers

import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
df = pd.read_csv('dataset.csv')

# Check for any null values and drop them if present
df = df.dropna()

# Define the BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(df['label'].unique()))

# Preprocess the data
max_length = 128
X = df['text'].values
y = df['label'].values

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the labels
y_encoded = label_encoder.fit_transform(y)

# Print the original labels and their corresponding encoded values
label_mapping = {label: encoded_label for label, encoded_label in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
print("Label Mapping:")
print(label_mapping)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.1, random_state=42)

# Tokenize inputs
encoded_data_train = tokenizer.batch_encode_plus(X_train,
                                                 add_special_tokens=True,
                                                 return_attention_mask=True,
                                                 padding=True,
                                                 truncation=True,
                                                 max_length=max_length,
                                                 return_tensors='pt')

encoded_data_val = tokenizer.batch_encode_plus(X_val,
                                               add_special_tokens=True,
                                               return_attention_mask=True,
                                               padding=True,
                                               truncation=True,
                                               max_length=max_length,
                                               return_tensors='pt')

# Convert tokenized inputs to tensors
input_ids_train = encoded_data_train['input_ids']
attention_masks_train = encoded_data_train['attention_mask']
labels_train = torch.tensor(y_train)

input_ids_val = encoded_data_val['input_ids']
attention_masks_val = encoded_data_val['attention_mask']
labels_val = torch.tensor(y_val)

# Create DataLoader for training and validation datasets
batch_size = 32
train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

val_data = TensorDataset(input_ids_val, attention_masks_val, labels_val)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Send model to device
model.to(device)

# Set optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

#2nd block
def evaluate_model(model, dataloader):
    model.eval()
    val_loss = 0
    val_correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            outputs = model(**inputs)
            loss = outputs.loss
            logits = outputs.logits
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(logits, 1)
            labels = inputs['labels']
            val_correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return val_loss / len(dataloader), val_correct / total


# Train the model
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'labels': batch[2]}
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
          # Evaluate the model
    val_loss, val_accuracy = evaluate_model(model, val_dataloader)
    print(f'Epoch {epoch + 1}:')
    print(f'Training Loss: {train_loss / len(train_dataloader)}')
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')
    print('---------------------------------------')

# Save the trained model
model.save_pretrained('bert_model', save_weights_only=True)
tokenizer.save_pretrained('bert_model')
