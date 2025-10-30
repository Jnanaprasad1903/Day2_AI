import torch
import torch.nn as nn 
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print(f"using deivce: {device}")


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5),(0.5))
    ]
)

train_data = datasets.MNIST(root='./dir',train=True,download=True,transform=transform)
test_data = datasets.MNIST(root='./dir',train=False,download=False,transform=transform)

train_loader = DataLoader(train_data,batch_size=128,shuffle=True)
test_loader = DataLoader(test_data,batch_size=128,shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1)
            torch.ReLU()
            nn.Conv2d(16,32,3,stride=2,padding=1)
            torch.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32,16,3,stride=2,out_padding=1)
            torch.ReLU()
            nn.ConvTranspose2d(16,1,3,stride=2,output_padding=1)
            torch.tanh()
        )


        def forward(self,x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x



model = autoencoder()

criterian = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

def add_noise(img):
    noisy = torch.randn_like(img)*0.5
    noisy = img + noisy
    noise = torch.clamp(noisy,-1.,1.)

for epoch in range(1):
    for img,label in train_loader:
        noisy_img = add_noise(img)
        output = model(noisy_img)
        loss = criterian(output,label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    #print(f"epoch:{epoch+1},loss:{loss.item()}")

correct = 0.0
total = 0.0
model.eval()
with torch.no_grad():
    for img,label in test_loader:
        output = model(noisy_img)
        max,predicted = torch.max(output,1)
        correct+= (predicted==label).sum().item()
        total+= label.size(0)
    print(f"accuracy on test:{(correct/total)*100}")




