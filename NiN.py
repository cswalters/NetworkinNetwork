import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import time

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 10
num_classes = 10
batch_size = 10
learning_rate = 0.001

# CIFAR dataset
CIFAR_ALL = "/Users/chris/Documents/University/2018/Sem 2/Neural Networks/Datasets/CIFARData.nosync/"

train_dataset = torchvision.datasets.CIFAR10(root=CIFAR_ALL,
                                           train=True,
                                           transform=transforms.ToTensor())

test_dataset = torchvision.datasets.CIFAR10(root=CIFAR_ALL,
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(4 * 4 * 96, num_classes)
        #self.globalAve = nn.AvgPool2d(kernel_size=4)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        #out = self.globalAve(out)
        #out = out.reshape(out.size(0), -1)
        return out


model = ConvNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
last = time.time()

total_step = len(train_loader)
for epoch in range(num_epochs):

    # Train model
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time: {:.2f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), time.time() - last))
            last = time.time()

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images after epoch {}: {} %'.format(epoch, 100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')