import torch.nn as nn

#添加激活函数
class Activation_Net(nn.Module):
    def __init__(self,in_dim,n_hidden1,n_hidden_2,out_dim):
        super(Activation_Net,self).__init__()
        self.layer1=nn.Sequential(
            nn.Linear(in_dim,n_hidden1),
            nn.ReLU(True),
        )
        self.layer2=nn.Sequential(
            nn.Linear(n_hidden1,n_hidden_2),
            nn.ReLU(True)
        )
        #最后一层不能添加激活函数
        self.layer3=nn.Sequential(n_hidden_2,out_dim)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        return  x