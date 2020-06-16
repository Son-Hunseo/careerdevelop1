#Linear Regression without PyTorch
import matplotlib.pyplot as plt
import numpy as np
import random

def gradientDescent (x, y, theta, alpha, m, numlterations):
    xTrans = x.transpose()

    plt.ion()

    for i in range(0, numlterations):
         hypothesis = np.dot(x, theta)
         loss = hypothesis - y
         cost = np.sum(loss **2) / (2*m)
         gradient = np.dot (xTrans, loss ) / m
         theta = theta - alpha * gradient

         if i % 10000 == 0:
             print("Iteration %d | Cost: %f" % (i, cost))

         if i % 100000 == 0:
             plt.cla()
             plt.scatter(x[:,1], y)
             plt.show()
             plt.pause(0.2)
    plt.ioff()
    return theta

def genData(numPoints, slope, bias, variance):
    x = np.zeros(shape=(numPoints,2))
    y = np.zeros(shape=numPoints)

    for i in range(0, numPoints):
        x[i][0] = 1
        x[i][1] = i

        y[i] = (i*slope + bias) + random.uniform(0, 1) * variance
    return x, y

x, y = genData(100, 0.5, 50, 5)
m, n = np.shape(x)
print(('Dimensionality of data: X: %d, %d Y: %d,1')%(m,n,y.shape[0]))
numIterations = 100000
alpha = 0.0001
np.random.seed(0)
theta = np.random.rand(2)
theta = gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)

#Variables, small functions, and linear regression in PyTorch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

#오류 해결
#relu, sigmoid, tanh 는 더이상 torch.nn.functional에서 사용하지 않고 torch에서 바로사용
#softplus 는 torch.nn. functional에 있다.

x_act = torch.linspace(-5, 5, 200)
x_act = Variable(x_act)
x_act_np = x_act.data.numpy()


y_act_relu = torch.relu(x_act).data.numpy()
y_act_sigmoid = torch.sigmoid(x_act).data.numpy()
y_act_tanh = torch.tanh(x_act).data.numpy()
y_act_softplus = F.softplus(x_act).data.numpy()


plt.figure(1, figsize=(8,6))
plt.subplot(221)
plt.plot(x_act_np, y_act_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_act_np, y_act_sigmoid, c='red', label = 'sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(223)
plt.plot(x_act_np, y_act_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(224)
plt.plot(x_act_np, y_act_softplus, c='red' , label = 'softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')


plt.show()


# 여기 까지는 그냥 그래프 개형


from torch import optim

x_torch = torch.from_numpy(x[:,1]).float().unsqueeze(1)
y_torch = torch.from_numpy(y).float().unsqueeze(1)

model = torch.nn.Sequential()
model_linear = torch.nn.Linear(1, 1, bias=True)
model.add_module("linear", model_linear)


loss = torch.nn.MSELoss(reduction= 'mean')
# size_average 도 없어질거란다 reduction = 'mean'을  씀으로서 해결
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.0)

batch_size = 100

num_iterations = 100000


for i in range(num_iterations):
    cost = 0.
    num_batches = len(x_torch) // batch_size
    for k in range(num_batches):

        start, end = k* batch_size, (k+1)* batch_size
        x_batch = x_torch[start:end]
        y_batch = y_torch[start:end]

        x_var = Variable(x_batch, requires_grad=False)
        y_var = Variable(y_batch, requires_grad=False)

        optimizer.zero_grad()


        fx = model.forward(x_var.view(len(x_var), 1))
        output = loss.forward(fx, y_var)

        output.backward()

        optimizer.step()

        cost += output.data# 오류가 났음 output.data[0] 를 output.data로 바꾸니 해결

    if i % (num_iterations/10) == 0:
        print("Epoch = %d, cost = %s" %(i + 1, cost / num_batches))

print('\nLearned parameters:')
w = next(model.parameters()).data
print("Weight = %.2f" %(w.numpy()))

print('Bias = %d' % (model_linear.bias))

# Building a Neural Network for Regression

torch.manual_seed(1)

x_torch = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y_torch = x_torch.pow(2) + 0.2*torch.rand(x_torch.size())

print(x_torch.size())
print(x_torch.size())

x_torch, y_torch = Variable(x_torch), Variable(y_torch)

plt.scatter(x_torch.data.numpy(), y_torch.data.numpy())
plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x)) #F.relu(torch.nn.functional.relu 지원 안함 torch.relu 로 수정)
        x = self.predict(x)
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_func = torch.nn.MSELoss()

plt.ion()

num_iterations = 1000
for t in range(num_iterations):
    prediction = net(x_torch)

    loss = loss_func(prediction, y_torch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % (num_iterations/10) == 0:
        print("Epoch = %d, loss = %s" % (t + 1, loss.data.numpy()))

        plt.cla()
        plt.scatter(x_torch.data.numpy(), y_torch.data.numpy())
        plt.plot(x_torch.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=% 4f' % loss.data, fontdict={'size':20, 'color': 'red'})#loss.data[] -> loss.data
        plt.show()
        plt.pause(0.2)

    plt.ioff()

# Building a Neural Network for Classification

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal (-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s= 100, lw=0, cmap='spring')
plt.show()

class Net(torch.nn.Module) :
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.hidden(x)) # F.relu --> torch.relu
        x = self.out(x)
        return x

net = Net(n_feature=2, n_hidden=10, n_output=2)
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

plt.ion()

for t in range(100):
    out = net(x)
    loss = loss_func(out, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t % 10 == 0 or t in [3, 6]:

        plt.cla()
        _, prediction = torch.max(F.softmax(out), 1)
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s = 100, lw=0, cmap='spring')
        accuracy = sum(pred_y == target_y)/200.
        plt.text(1.5, -4, 'Accuracy=%.2f' %accuracy, fontdict={'size':20, 'color': 'red'})
        plt.show()
        plt.pause(0.1)

    plt.ioff()
