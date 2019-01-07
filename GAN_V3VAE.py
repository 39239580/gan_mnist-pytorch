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
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self): #编码器
        super().__init__()
        
        self.fc1=nn.Linear(784,400)
        self.fc21=nn.Linear(400,20)
        self.fc22=nn.Linear(400,20)
        self.fc3=nn.Linear(20,400)
        self.fc4=nn.Linear(400,784)
    
    def encode(self,x): #编码器
        h1=F.relu(self.fc1(x))
        return self.fc21(h1),self.fc22(h1)
    
    def reparametrize(self,mu,logvar):
        std=logvar.mul(0.5).exp_()
        eps=torch.FloatTensor(std.size()).normal_() #正态分布
        if torch.cuda.is_available():
            eps=Variable(eps).cuda()
        else:
            eps=Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self,z): #解码器
        h3=F.relu(self.fc3(z))
        return  F.tanh(self.fc4(h3))
    
    def forward(self,x):
        mu,logvar=self.encode(x) #编码得到均值与log sigma^2
        z=self.reparametrize(mu,logvar) #重新参数化成正态分布
        return self.decode(z),mu,logvar #解码同时输出均值方差
            
if not os.path.exists('./imgvae'):
    os.mkdir('./imgvae')

def to_img(x):
    out =0.5*(x+1)
    out =out.clamp(0,1) #等价于torch.clamp(out,0,1),将像素点，都限制在0到1的范围内
    out =out.view(-1,1,28,28)
    return out

reconstruction_function=nn.MSELoss(size_average=False) #交叉熵
def loss_function(recon_x,x,mu,logvar,size_average):
    """
    recon_x: 生成的图片
    x:原始图片
    mu,均值
    logvar，log方差
    """

    MSE=reconstruction_function(recon_x,x) #MSE
    # loss=0.5*sum(1+log(sigma^2)-mu^2-sigma^2)
    KLD_element=mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD=torch.sum(KLD_element).mul_(-0.5)
    #KL divergence   Kl散度
    loss=MSE+KLD
    lenth=x.size(0) #batchsize长度
    if size_average == True:
        final_loss=loss/lenth
    else:
        final_loss=loss
    return final_loss
    #logvar为 log sigma^2  mu.pow(2)为mu^2
    #logvar.exp()为 e^(log sigma^2)=sigma^2
    #mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)  为
    #(mu^2+sigma^2)*(-1)+1+log sigma^2
    #loss=0.5*sum(1+log(sigma^2)-mu^2-sigma^2)
        
# 定义超参数
batch_size=64
num_epoches=100

download_start_time=time.time()
data_tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
#加载数据
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_dataset=datasets.MNIST(root='./data',train=False,transform=data_tf,)
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)   #加载数据
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
download_end_time=time.time()
print("共计下载数据及加载数据花费时间：%f"%(download_end_time-download_start_time))

# 创建网络
NET=VAE()
NET.train()
if torch.cuda.is_available():
    NET=NET.cuda()
else:
    NET=NET
print(NET)

#criterion=nn.BCELoss()  #交叉熵
optimizer=optim.Adam(NET.parameters(),lr=0.0003)
for epoch in range(num_epoches):
    start=time.time()
    print('Current epoch ={}'.format(epoch))
    train_generator_loss=0
    train_discrimination_loss=0
    for i ,(img,_)in enumerate(train_loader):#利用enumerate取出一个可迭代对象的内容
        num_img=img.size(0) # 获取图片的大小，获取行数
        img=img.view(num_img,-1)  # img.view与img.reshape功能类似,
        if torch.cuda.is_available():
            im=Variable(img).cuda()

        else:
            im=Variable(img)

        #训练网络
        # 计算loss    
        recon_im,mu,logvar=NET(im)
        loss=loss_function(recon_im,im,mu,logvar,size_average=True) 
        optimizer.zero_grad()
        loss.backward() #反响传播
        optimizer.step()
     
        
        if(i+1)%100==0:#每100张显示一次
            print('Epoch[{}/{}],loss:{:.6f}'.format(epoch+1,num_epoches,loss.item()))
    print('Epoch[{}/{}],loss:{:.6f}'.format(epoch+1,num_epoches,loss.item()))    

    if epoch==0:
        real_images = to_img(im.cpu().data) #转成图片
        save_image(real_images,'./imgvae/real_iamges.png')
        
    fake_images=to_img(recon_im.cpu().data)
    save_image(fake_images,'./imgvae/genaration_image-{}.png'.format(epoch+1))

torch.save(NET.state_dict(),'./generator.pth')   #保存模型
     
