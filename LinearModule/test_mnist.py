# import LinearModule.simpleNet as Net
from LinearModule.simpleNet import simpleNet as Net
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
from torchvision import  datasets,transforms

batch_size=64

#数据预处理
#ToTensro将图片标准化，范围0---1，Normalize两个参数,第一个均值，第二个方差
data_tf=transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])]
)

#读取数据集
train_dataset=datasets.MNIST(root='./data',train=True,transform=data_tf,download=True)
test_data=datasets.MNIST(root='./data',transform=data_tf,download=True)

#shuffle表示每次迭代数据的时候是否将数据打乱
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_data,batch_size=batch_size,shuffle=True)


model=torch.load('../save_logs/Linear_simple.ckpt')
