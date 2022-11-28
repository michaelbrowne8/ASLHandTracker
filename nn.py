import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from handDetector import handDetector as hd
import cv2
import time
from sklearn.neural_network import MLPClassifier as MLPC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.mixture import GaussianMixture

labels = np.load('labelsNew.npy') # 19436 labels
data = np.load('dataNew.npy').astype(float)    # 19436 preprocessed images 
                               # of x,y,z at 21 landmarks a hand

def preprocess(lmlist):
  max = lmlist.max(axis=0)
  min = lmlist.min(axis=0)
  newList = np.zeros_like(lmlist)

  if max[0] - min[0] == 0: max[0] += 0.0001
  if max[1] - min[1] == 0: max[1] += 0.0001
  if max[2] - min[2] == 0: max[2] += 0.0001
  newList[:, 0] = (lmlist[:, 0] - min[0]) / (1.0 * max[0] - min[0])
  newList[:, 1] = (lmlist[:, 1] - min[1]) / (1.0 * max[1] - min[1])
  newList[:, 2] = (lmlist[:, 2] - min[2]) / (1.0 * max[2] - min[2])
  return newList

for i in range(len(data)):
    data[i] = preprocess(data[i])

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
lr = 0.15
num_epochs = 200

input_dim = 21 * 3 # including z
# input_dim = 21 * 2 # not including z
# hidden_dims = [int(264 * 1.6), 200]
hidden_dims = [100]
output_dim = 29
model = Model(input_dim, hidden_dims, output_dim)

train_data = Data(X_train, y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_data = Data(X_test, y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)


def train(model, batch_size=64, num_epochs=100, criterion=criterion, optimizer=optimizer, lr=0.15, \
        train_dataloader=train_dataloader, test_dataloader=test_dataloader, plot=True, test=True):

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

    if plot:
        plt.title("Training Curve (batch_size={}, lr={})".format(batch_size, lr))
        plt.plot(iters, losses, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

    if test:
        with torch.no_grad():
            for xs, ts in test_dataloader:
                zs = model(xs)
                pred = zs.max(1, keepdim=True)[1] 
                correct += pred.eq(ts.view_as(pred)).sum().item()
                total += int(ts.shape[0])

        print(f'Accuracy of the network on the {total} test instances: {100 * correct // total}%')

    return model

translate = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
            "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
    ]



# model = train(model)

#model = MLPC(hidden_layer_sizes=(100, 70, 50, 20, 10), batch_size=batch_size, max_iter=num_epochs)
# model = LR()
model = GaussianMixture(len(translate))
model.fit(data, labels)
score = model.score(data, labels)

print('\n\n')
print(f'{score=}')
print('\n\n')
input()

pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    # frame = frame[:, (w-h)//2:w-(w-h)//2, :]
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter = "nothing"
    if lmlist:
        # lmlist = np.array(lmlist, dtype=float)

        lmlist = preprocess(np.array(lmlist, dtype=float))
        if lmlist.shape != (21, 3):
            continue
        lmlist = lmlist.reshape(1, 21*3)
        # lmlist = lmlist[:, :2] # no z
        # lmlist = torch.from_numpy(lmlist.astype(np.float32))
        # zs = model(lmlist[None, :, :])
        
        # pred = zs.max(1, keepdim=True)[1] 
        pred = model.predict(lmlist)[0]
        # pred = max(zs)
        print(pred)
        letter = translate[pred]
    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(frame, letter, (h-20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break