import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import time
print("PyTorch 버전:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("사용 가능한 GPU 개수:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("현재 GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("사용 장치:", device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                     download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                         shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False,
                                    download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                        shuffle=False, num_workers=2)
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

start_time = time.time()
train_subset = torch.utils.data.Subset(trainset, range(1000))
train_subset_loader = torch.utils.data.DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_subset_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 10 == 9:
            print(f'[{epoch + 1}, {i + 1}] 손실: {running_loss / 10:.3f}')
            running_loss = 0.0

end_time = time.time()
print(f"\n완료 시간: {end_time - start_time:.2f}초")

correct = 0
total = 0
test_subset = torch.utils.data.Subset(testset, range(100))
test_subset_loader = torch.utils.data.DataLoader(test_subset, batch_size=32, shuffle=False, num_workers=2)

with torch.no_grad():
    for data in test_subset_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'정확도: {100 * correct / total:.2f}%')
dataiter = iter(test_subset_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

outputs = net(images)
_, predicted = torch.max(outputs, 1)
images = images.cpu().numpy()

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i][0], cmap='gray')
    plt.title(f'예측: {predicted[i].item()}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('pytorch_predictions.png')