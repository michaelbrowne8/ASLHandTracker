import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt


'''
Neural Network Model
...needs work :(
70% accuracy? Good for 29 classes?
'''



labels = np.load('labels.npy') # 19436 labels
data = np.load('data.npy')     # 19436 preprocessed images 
                               # of x,y,z at 21 landmarks a hand


# dataNew = data[:, :, :2]
data = data.reshape(data.shape[0], 21*3) # including z
# data = dataNew.reshape(data.shape[0], 21*2) # not including z

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).type(torch.LongTensor)
        self.len = self.X.shape[0]
       
    def __getitem__(self, index):
        return self.X[index], self.y[index]
   
    def __len__(self):
        return self.len

class Model(nn.Module):
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(Model, self).__init__()
        layers = [input_dim] + hidden_dims + [output_dim]
        layer = []
        for i in range(len(layers)-1):
            layer.append(nn.Linear(layers[i], layers[i+1]))
            layer.append(nn.ReLU())
        layer = layer[:-1]
        self.seq = nn.Sequential(*layer)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.seq(x)
        x = torch.sigmoid(x)
        # return nn.functional.softmax(x, dim=1)
        return x

batch_size = 64

train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

input_dim = 21 * 3 # including z
# input_dim = 21 * 2 # not including z
hidden_dims = [int(264 * 1.5), 200]
output_dim = 29

model = Model(input_dim, hidden_dims, output_dim)

lr = 0.1
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

num_epochs = 100

correct = 0
total = 0

iters, losses = [], []
# training
n = 0 
for epoch in range(num_epochs):
    for xs, ts in train_dataloader:
        if len(ts) != batch_size:
            continue
        
        zs = model(xs)
        loss = criterion(zs, ts)
        loss.backward() 
        optimizer.step() 
        optimizer.zero_grad() 
        
        iters.append(n)
        losses.append(float(loss)/batch_size) 
        n += 1

plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, lr))
plt.plot(iters, losses, label="Train")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.show()


with torch.no_grad():
    for xs, ts in test_dataloader:
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] 
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])

print(f'Accuracy of the network on the {total} test instances: {100 * correct // total}%')