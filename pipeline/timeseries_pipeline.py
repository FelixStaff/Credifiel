import torch

import torch.nn as nn
import torch.optim as optim

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
            for (history, input, label) in train_loader:
                self.optimizer.zero_grad()
                history = history.unsqueeze(0)  
                input = input.unsqueeze(0)
                outputs = self.model(history, input)
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')