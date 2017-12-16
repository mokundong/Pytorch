
# coding: utf-8

# In[1]:


import torch
import numpy as np
import pandas as pd
from torch.autograd import Variable
from torch import nn,optim
from torch.autograd import variable
from torch.utils.data import dataloader
from torchvision import datasets,transforms


# In[2]:


#定义三层网络
class sipleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1 = nn.Linear(in_dim,n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2,n_hidden_3)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# In[3]:


#添加激活函数
class Activation_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(NeuralNetwork,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# In[4]:


#添加标准化--收敛加速方法
class Batch_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Batch_Net,self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2,out_dim))
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# In[5]:


batch_size = 64
learning_rate = 1e-2
num_epoches = 20


# In[6]:


data_tf = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])


# In[ ]:


#下载训练数据集
train_dataset = datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset = datasets.MNIST(root='./data',train=False,transform=data_tf)
train_loader = Dataloader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = Dataloader(test_dataset,batch_size=batch_size,shuffle=False)

