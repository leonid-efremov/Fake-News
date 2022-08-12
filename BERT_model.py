"""
Tools for setup and training/testing model
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#%% Convert to torch DataLoader
class torchDataset(Dataset):

    def __init__(self, X, y, to_bert=False, tokenizer=None, max_len=512):
        self.x_data, self.y_data = X, y
        self.to_bert = to_bert
        if self.to_bert:
            self.tokenizer = tokenizer
            self.max_len = max_len
    
    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        '''
        Data types:
        x_data - list
        y_data - list
        '''
        text, label = self.x_data[idx], self.y_data[idx]

        if self.to_bert:

            encoding = self.tokenizer.encode_plus(text, add_special_tokens=True,
                                                  max_length=self.max_len,
                                                  return_token_type_ids=False,
                                                  padding='max_length',
                                                  return_attention_mask=True,
                                                  return_tensors='pt',
                                                  truncation=True)

            return {'text': text,
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'targets': torch.tensor(label, dtype=torch.long)}
        else:
            return text, label

def to_torch(X, y, shuffle=False, batch_size=64, to_bert=False, tokenizer=None):
    """Convert data to torch.DataLoader format"""
    dataset = torchDataset(X, y, to_bert=to_bert, tokenizer=tokenizer) 
    return DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=True)



#%% Training and evaluating model
class BertClassifier:

    def __init__(self, bert, n_classes=2, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """Initialize neural network and setup layers"""

        self.model = bert
        self.out_features = self.model.bert.encoder.layer[1].output.dense.out_features
        self.model.classifier = nn.Sequential(nn.Dropout(0.1),
                                              nn.Linear(self.out_features, n_classes))
        self.device = device
        self.model.to(self.device)

    def train(self, train, valid, lr=1e-5, n_epochs=5, print_every=1):
        """
        Data types:
        train - torch.DataLoader
        valid - torch.DataLoader

        Training loop.
        Save model if needed.
        """

        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, 
                                                               verbose=True, factor=0.5)

        n_epochs = n_epochs
        self.loss_list = []

        for epoch in range(n_epochs):
            
            best_accuracy = 0

            # Training
            train_losses = []
            correct_predictions = 0  
            self.model.train()
    
            for data in train:

                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["targets"].to(self.device)

                output = self.model(input_ids, attention_mask)
                loss = criterion(output.logits, labels)
                train_losses.append(loss.item())

                preds = torch.argmax(output.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_acc = correct_predictions.double() / (len(train)*train.batch_size)
            train_loss = np.mean(train_losses)
            
            # Validation
            val_losses = []
            correct_predictions = 0 
            self.model.eval()

            for data in valid:                
                # inputs, labels = inputs.to(device), labels.to(device)
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["targets"].to(self.device)

                output = self.model(input_ids, attention_mask)
                loss = criterion(output.logits, labels)
                val_losses.append(loss.item())

                preds = torch.argmax(output.logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
        
            val_acc = correct_predictions.double() / (len(valid)*valid.batch_size)
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss) 
        
            self.loss_list.append(val_loss)

            if epoch % print_every == 0:
                print("Epoch: {}/{}".format(epoch + 1, n_epochs))
                print("Validation loss: {:.6f},".format(val_loss),
                      "Accuracy: {:.6f}".format(val_acc))

            if val_acc > best_accuracy:
                self.best_epoch = epoch
                torch.save(self.model.state_dict(), 'model.torch')
                best_accuracy = val_acc
        
        self.model.load_state_dict(torch.load('model.torch'))


    def predict(self, test, from_pretrained=True):
        """
        Data types:
        test - torch.DataLoader

        Predicting labels for test data.
        """  
        batch_size = test.batch_size  

        if from_pretrained:
            self.model.load_state_dict(torch.load('model.torch'))
        self.model.eval()

        y_pred = torch.zeros(batch_size*len(test))

        for i, data in enumerate(test):
            input_ids = data["input_ids"].to(self.device)
            attention_mask = data["attention_mask"].to(self.device)

            output = self.model(input_ids, attention_mask)

            # convert output probabilities to predicted class (0 or 1)
            pred = torch.argmax(output.logits, dim=1)

            y_pred[i*batch_size:(i+1)*batch_size] = pred[0]

        return y_pred.cpu().numpy()
        
    
