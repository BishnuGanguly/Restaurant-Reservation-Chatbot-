import torch

def evaluate_model(model, dataloader, device):
    model.eval()
    val_loss, val_correct, total = 0, 0, 0

    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
            loss, logits = outputs.loss, outputs.logits

            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            val_correct += (predicted == inputs['labels']).sum().item()
            total += inputs['labels'].size(0)

    return val_loss / len(dataloader), val_correct / total
