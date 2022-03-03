import torch.nn as nn


class simpleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(simpleNet,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,out_dim)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)

        return x