import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Load dataset
df = pd.read_csv('data/dataset.csv').dropna()

# Initialize tokenizer and model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(df['label'].unique()))

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['label'])

# Save label mapping
label_mapping = {label: encoded_label for label, encoded_label in zip(label_encoder.classes_, y_encoded)}
print("Label Mapping:", label_mapping)

# Split data
X_train, X_val, y_train, y_val = train_test_split(df['text'].values, y_encoded, test_size=0.1, random_state=42)

# Tokenize text
def tokenize_text(texts, tokenizer, max_length=128):
    return tokenizer.batch_encode_plus(
        texts, add_special_tokens=True, return_attention_mask=True,
        padding=True, truncation=True, max_length=max_length, return_tensors='pt'
    )

encoded_train = tokenize_text(X_train, tokenizer)
encoded_val = tokenize_text(X_val, tokenizer)

# Convert to tensors
train_data = TensorDataset(encoded_train['input_ids'], encoded_train['attention_mask'], torch.tensor(y_train))
val_data = TensorDataset(encoded_val['input_ids'], encoded_val['attention_mask'], torch.tensor(y_val))

# Create dataloaders
batch_size = 32
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
epochs = 5
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Training loop
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        train_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    print(f"Epoch {epoch+1}: Training Loss = {train_loss / len(train_dataloader)}")

# Save model and tokenizer
model.save_pretrained('models/bert_model')
tokenizer.save_pretrained('models/bert_model')

print("Model and tokenizer saved!")
