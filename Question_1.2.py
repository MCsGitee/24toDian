import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        
        self.W_xh = nn.Linear(input_size, hidden_size)  
        self.W_hh = nn.Linear(hidden_size, hidden_size) 
        self.W_hy = nn.Linear(hidden_size, output_size)  

    def forward(self, x, h_prev):
        h = torch.tanh(self.W_xh(x) + self.W_hh(h_prev))
        y = self.W_hy(h)
        return y, h

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)

input_size = 28      
hidden_size = 128    
output_size = 10     

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

model = RNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []

num = 10

for epoch in range(num):
    epoch_loss = 0
    for i, (images, labels) in enumerate(train_loader):
        
        batch_size = images.size(0)
        h_prev = model.init_hidden(batch_size)
        images = images.view(batch_size, 28, 28)

        for t in range(28): 
            x_t = images[:, t, :] 
            output, h_prev = model(x_t, h_prev)

        loss = criterion(output, labels)
        
        epoch_loss += loss.item()
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{num}] Average Loss: {avg_loss:.4f}')

    


plt.figure(figsize=(10, 5))
plt.plot(range(1, num + 1), train_losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

model.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        batch_size = images.size(0)
        h_prev = model.init_hidden(batch_size)

        images = images.view(batch_size, 28, 28)
        for t in range(28):
            x_t = images[:, t, :]
            output, h_prev = model(x_t, h_prev)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model: {100 * correct / total:.2f}%')
