import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import snntorch as snn
from snntorch import surrogate, utils
from snntorch import spikeplot as splt

import matplotlib.pyplot as plt

# Settings
batch_size = 128
num_steps = 25
learning_rate = 1e-3
num_epochs = 3

# Load MNIST dataset
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define a simple feedforward spiking network
class SNNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(512, 10)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        spk1_rec = []
        spk2_rec = []
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(num_steps):
            cur = self.flatten(x)
            cur = self.fc1(cur)
            spk1, mem1 = self.lif1(cur, mem1)

            cur = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur, mem2)

            spk1_rec.append(spk1)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)

# Init network
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SNNNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    net.train()
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        spk_out = net(data, num_steps=num_steps)
        output_sum = spk_out.sum(dim=0)  # Sum across time
        loss = loss_fn(output_sum, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

# Quick test
net.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        spk_out = net(data, num_steps=num_steps)
        output_sum = spk_out.sum(dim=0)
        preds = output_sum.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

print(f"Test Accuracy: {correct / total * 100:.2f}%")
