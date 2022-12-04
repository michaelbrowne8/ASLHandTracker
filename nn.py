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
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.model_selection import cross_val_score
'''

Ok so I overcomplicated it, one layer works

File's a little messy, so here's whats going on.

    - dataNew.npy is unprocessed, so I process it all using preprocess
    - for some of the labels, there just isn't enough info on them
        (like c or o) so those just become "nothing"
        * scroll down to line 60 or so for new alphabet
    - train test split up the info, saving 20% for testing
    - creating the model (feel free to change around to see if it works better)
    - giving the model its parameters (batch size, lr, ...)
    - training and testing

At the very bottom of the file is the hand detector using the webcam
This one we can actually make into a separate file so that it can just be 
    imported and ran
But it should show the tracked hand and the letter it thinks it is

Press 'Q' to quit

'''

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

'''

NEW ALPHABET

doesn't include C, D, O, P, Q, del, space due to lack of data

'''
newTranslate = ["A", "B", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", 
             "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "nothing"
    ]

translate = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", 
            "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "del", "nothing", "space"
    ]

removed = [2, 3, 14, 15, 16, 26, 28]

# for r in removed:
#     labels = np.where(labels == r, 27, labels)

# for i in range(2, 14):
#     labels = np.where(labels == i, i - 2, labels)

# for i in range(17, 26):
#     labels = np.where(labels == i, i - 5, labels)

# labels = np.where(labels == 27, len(newTranslate)-1, labels)
'''
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
num_epochs = 150

input_dim = 21 * 3 # including z
# input_dim = 21 * 2 # not including z
# hidden_dims = [500, 300, 100]
hidden_dims = [150]
# output_dim = len(newTranslate)
output_dim = len(translate)

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

'''
# params = {'hidden_layer_sizes': [(175,), (172,), (177,)]}
# grids = GSCV(MLPC(max_iter=500), params, cv=10)
# grids.fit(data, labels)
# print(f'{grids.best_params_=}')
# model = train(model)
# model = MLPC(hidden_layer_sizes=grids.best_params_["hidden_layer_sizes"])
model = MLPC(hidden_layer_sizes=(175,))
model.fit(data, labels)
print(model.score(data, labels))
print(np.mean(cross_val_score(model, data, labels)))
input()
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = hd(maxHands=1)
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape
    frame = detector.findHands(frame)
    lmlist = detector.findPosition(frame)
    letter = "nothing"
    if lmlist:
        lmlist = preprocess(np.array(lmlist, dtype=float))
        # if lmlist.shape != (21, 2): #not including z
        if lmlist.shape != (21, 3): #including z
            continue

        lmlist = lmlist.reshape(1, 21*3) # including z
        # lmlist = lmlist.reshape(1, 21*2) # not including z
        # zs = model(lmlist[None, :, :])
        # pred = zs.max(1, keepdim=True)[1] 
        pred = model.predict(lmlist)[0]
        # letter = newTranslate[pred]
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