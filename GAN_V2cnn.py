# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 15:02:48 2018

@author: Weixia
"""
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.utils import save_image
import torch.nn as nn
import time
import os

class DiscriminationNet(nn.Module):
    def __init__(self):
        super().__init__() #编码器
        conv1=nn.Sequential() #输入数据为 28*28  
        conv1.add_module('conv1',nn.Conv2d(1,32,5,stride=1,padding=2))#输出数据batch,32,28,28
        conv1.add_module('leakyrelu1',nn.LeakyReLU(0.2,True))
        conv1.add_module('Avg_pool1',nn.AvgPool2d(2,stride=2)) #batch,32,14,14
        self.conv1=conv1
        conv2=nn.Sequential()
        conv2.add_module('conv2',nn.Conv2d(32,64,5,stride=1,padding=2))#batch,64,14，14
        conv2.add_module('leakyrelu2',nn.LeakyReLU(0.2,True))
        conv2.add_module('max_pool2',nn.AvgPool2d(2,stride=2)) #batch，64，7，7   64*7*7=3136
        self.conv2=conv2
        
        fc=nn.Sequential()
        fc.add_module('fc1',nn.Linear(64*7*7,1024)) 
        fc.add_module('Leakyrelu3',nn.LeakyReLU(0.2,True))
        fc.add_module('fc2',nn.Linear(1024,1))
        fc.add_module('Sigmoid',nn.Sigmoid())
        self.fc=fc
        
    def forward(self,x):
        """
        x:batch,width,height,channel=1
        """
        
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)  #batch*N列，
        x=self.fc(x)
        return  x
    
class GeneratorNet(nn.Module):#解码器
    def __init__(self,input_size,num_feature):
        super().__init__()
        fc=nn.Sequential()       
        fc.add_module('fc',nn.Linear(input_size,num_feature)) #batch ，56*56*1=3136
        self.fc=fc
        
        BR=nn.Sequential()
        BR.add_module('bn',nn.BatchNorm2d(1)) 
        BR.add_module('relu',nn.ReLU(True))
        self.BR=BR
        
        DOWNSAMPLE1=nn.Sequential()
        DOWNSAMPLE1.add_module('conv1',nn.Conv2d(1,50,3,stride=1,padding=1)) #batch，50，56，56
        DOWNSAMPLE1.add_module('bn2',nn.BatchNorm2d(50,eps=0.001))
        DOWNSAMPLE1.add_module('relu2',nn.ReLU(True))
        self.DOWNSAMPLE1=DOWNSAMPLE1
        
        DOWNSAMPLE2=nn.Sequential()
        DOWNSAMPLE2.add_module('conv2',nn.Conv2d(50,25,3,stride=1,padding=1))#batch，25， 56，56
        DOWNSAMPLE2.add_module('bn3',nn.BatchNorm2d(25,eps=0.001))
        DOWNSAMPLE2.add_module('relu3',nn.ReLU(True))
        self.DOWNSAMPLE2=DOWNSAMPLE2
        
        DOWNSAMPLE3=nn.Sequential()
        DOWNSAMPLE3.add_module('conv3',nn.Conv2d(25,1,2,stride=2,)) #batch 1，28，28
        DOWNSAMPLE3.add_module('Tanh',nn.Tanh())
        self.DOWNSAMPLE3=DOWNSAMPLE3
        
    def forward(self,x):
        x=self.fc(x)
        x=x.view(x.size(0),1,56,56) #batch ，1，56，56
        x=self.BR(x)
        x=self.DOWNSAMPLE1(x)
        x=self.DOWNSAMPLE2(x)
        x=self.DOWNSAMPLE3(x)
        return x
        
if not os.path.exists('./imgv2'):
    os.mkdir('./imgv2')

def to_img(x):
    out =0.5*(x+1)
    out =out.clamp(0,1)
    out =out.view(-1,1,28,28)
    return out
    
# 定义超参数
batch_size=64
num_epoches=100
z_dimension=100 #噪声维度

download_start_time=time.time()
data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
#加载数据
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf,)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)   #加载数据
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
download_end_time=time.time()
print("共计下载数据及加载数据花费时间：%f"%(download_end_time-download_start_time))

# 创建判断网络
D=DiscriminationNet()
D.train()
if torch.cuda.is_available():
    D=D.cuda()
else:
    D=D
print(D)

# 创建生成网络
G=GeneratorNet(input_size=z_dimension,num_feature=3136)
G.train()
if torch.cuda.is_available():
    G=G.cuda()
else:
    G=G
print(G)
criterion=nn.BCELoss()
D_optimizer=optim.Adam(D.parameters(),lr=0.0003)
G_optimizer=optim.Adam(G.parameters(),lr=0.0003)

for epoch in range(num_epoches):
    start=time.time()
    print('Current epoch ={}'.format(epoch))
    train_generator_loss=0
    train_discrimination_loss=0
    for i ,(img,_)in enumerate(train_loader,1):#利用enumerate取出一个可迭代对象的内容
        num_img=img.size(0) # train_loader 是个四维张量（batch_size, channel,x,y）
        #num_img=batch_size
#        img=img.view(num_img,1)
#        img=img.view(num_img,-1)  # img.view与img.reshape功能类似
        if torch.cuda.is_available():
            real_img=Variable(img).cuda()
            real_label=Variable(torch.ones(num_img,1)).cuda()
            fake_label=Variable(torch.zeros(num_img,1)).cuda()

        else:
            real_img=Variable(img)
            real_label=Variable(torch.ones(num_img,1))
            fake_label=Variable(torch.zeros(num_img,1))

        #训练判断网络
        # 计算real_img上的loss    
        real_out=D(real_img)
        D_loss_real=criterion(real_out,real_label) 
        real_scores=real_out

        # 计算fake_img上的loss
        if torch.cuda.is_available():
            z=Variable(torch.randn(num_img,z_dimension)).cuda()
        else:
            z=Variable(torch.randn(num_img,z_dimension)) 
        fake_img=G(z)
        fake_out=D(fake_img)
        D_loss_fake=criterion(fake_out,fake_label) 
        fake_scores=fake_out
        
        #判断网络总误差
        D_loss=D_loss_real+D_loss_fake
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()
        train_discrimination_loss+=D_loss.item()
        
        
        #训练生成网络
        # 计算fake_img的loss
        if torch.cuda.is_available():
            z=Variable(torch.randn(num_img,z_dimension)).cuda()
        else:
            z=Variable(torch.randn(num_img,z_dimension))
        fake_img=G(z)
        out_put=D(fake_img)
        G_loss=criterion(out_put,real_label) 
        #生成网络误差
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
        train_generator_loss+=G_loss.item()   #item是从标量中获取python数字
        
        if(i+1)%100==0:#每100张显示一次
            print('Epoch[{}/{}],D_loss:{:.6f},G_loss:{:.6f},D real:{:.6f},D fake:{:.6f}'
                  .format(epoch+1,num_epoches,D_loss.item(),G_loss.item(),real_scores.data.mean(),
                          fake_scores.data.mean()))
    
    train_discrimination_loss=train_discrimination_loss/len(train_dataset)
    train_generator_loss=train_generator_loss/len(train_dataset)
    print('Epoch[{}/{}],D_AVE_loss:{:.6f},G_AVE_loss:{:.6f}'
                  .format(epoch+1,num_epoches,train_discrimination_loss,train_generator_loss))    

    if epoch==0:
        real_images =to_img(real_img.cpu().data)
        save_image(real_images,'./imgv2/real_iamges.png')
        
    fake_images=to_img(fake_img.cpu().data)
    save_image(fake_images,'./imgv2/fake_image-{}.png'.format(epoch+1))

torch.save(G.state_dict(),'./generator.pth')   #保存模型
torch.save(D.state_dict(),'./discriminator.pth')        










        
        
            
