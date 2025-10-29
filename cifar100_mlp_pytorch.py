import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

#transforms
transform = transforms.Compose(
[ transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))  #mean and standaerd deviation(RGB)
]
)

#load the data
train_data = datasets.CIFAR100(root='./dir',train=True,download=True,transform=transform)

test_data = datasets.CIFAR100(root='./dir',train=False,download=True,transform=transform)

#Dataloader
train_loader = DataLoader(train_data,batch_size=128,shuffle=True) #shuffle datas feeding will be shuffled

test_loader = DataLoader(test_data,batch_size=128,shuffle=False) #no need to shuffle , bcz at this stage it should predict

#Architecture
class cifar100_mlp(nn.Module):
    def __init__(self):
        super(cifar100_mlp,self).__init__()
        self.flatten = nn.Flatten()        
        self.fc1 = nn.Linear(32*32*3,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,100)

    def forward(self,x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

#object creation 
model = cifar100_mlp()
criterian = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

#train
#send the data, loss, back-propogation, optimizer
epoch_loss=0.0
for epoch in range(30):
    for image,label in train_loader:
        optimizer.zero_grad()
        output = model(image)
        loss = criterian(output,label)
        loss.backward()
        optimizer.step() 
        epoch_loss+=loss.item()
    print(f"epoch_loss: {epoch_loss}")

total=0.0
correct = 0.0
model.eval()
with torch.no_grad():
    for image,label in test_loader:
        output = model(image)
        _,predicted = torch.max(output,1)
        total += label.size(0)
        correct+=(predicted==label).sum().item()
print(f"accuracy:{(correct/total)*100}")

        
        
