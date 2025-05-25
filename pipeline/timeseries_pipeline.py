import torch

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

class pipeline:
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()


    def train(self, train_loader, num_epochs=10):
        """
        El input es de tama;o ambiguo asi que solo es 1,L,D y el valor actual que es 1,1,D
        """
        self.model.train()
        for epoch in range(num_epochs):
            progress_bar = tqdm(train_loader)
            running_loss = 0.0
            total_loss = 0.0
            counter = 0
            accuracy = 0.0
            for (history,input, label) in progress_bar:
                self.optimizer.zero_grad()
                history = history.unsqueeze(0)
                input = input.unsqueeze(0)
                label = label.unsqueeze(0).reshape(-1)
                outputs = self.model(history, input)
                loss = self.criterion(outputs, label)
                # Add to the loss a l1 regularization
                l1_lambda = 0.01
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + l1_lambda * l1_norm
                total_loss += loss
                running_loss += loss.item()
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                accuracy += (predicted == label).float().sum()
                counter += 1
                progress_bar.set_postfix(loss=running_loss / (counter + 1), accuracy=accuracy / (counter + 1))
                if counter % 100 == 0:
                    total_loss /= 100
                    total_loss.backward()
                    self.optimizer.step()
                    total_loss = 0.0
                    # Save the model every 100 steps
                    if counter % 1000 == 0:
                        torch.save(self.model.state_dict(), f'model_epoch_{epoch}_step_{counter}.pth')

            
            total_loss = 0.0
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')