import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def test_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        y_trues, y_preds = [], []
        
        for inputs, labels in tqdm(dataloader):
            outputs = model(inputs)
            predicted = torch.argmax(outputs, 1)
            
            y_trues.extend(labels.detach().cpu().numpy())
            y_preds.extend(predicted.detach().cpu().numpy())
            
    acc = accuracy_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds, average='macro')
    
    return {
        'accuracy': accuracy_score(y_trues, y_preds),
        'f1': f1_score(y_trues, y_preds, average='macro'),
        'confusion_matrix': confusion_matrix(y_trues, y_preds),
        'classification_report': classification_report(y_trues, y_preds, zero_division=0)
    }