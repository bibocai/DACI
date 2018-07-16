import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TreeLstmCell(nn.Module):
	def __init__(self,input_size,hidden_size):
		super(TreeLstmCell,self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.w_ih = nn.Parameter(torch.Tensor(3*hidden_size,input_size))
		self.w_hh = nn.Parameter(torch.Tensor(3*hidden_size,hidden_size))
		self.b_ih = nn.Parameter(torch.Tensor(3*hidden_size))
		self.b_hh = nn.Parameter(torch.Tensor(3*hidden_size))

		self.wl_ih = nn.Parameter(torch.Tensor(hidden_size,input_size))
		self.wl_hh = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
		self.bl_ih = nn.Parameter(torch.Tensor(hidden_size))
		self.bl_hh = nn.Parameter(torch.Tensor(hidden_size))

		self.wr_ih = nn.Parameter(torch.Tensor(hidden_size,input_size))
		self.wr_hh = nn.Parameter(torch.Tensor(hidden_size,hidden_size))
		self.br_ih = nn.Parameter(torch.Tensor(hidden_size))
		self.br_hh = nn.Parameter(torch.Tensor(hidden_size))

		self.w = nn.Parameter(torch.Tensor(2,hidden_size))
		self.b = nn.Parameter(torch.Tensor(2))

		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1.0/math.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv,stdv)

	def forward(self,input,hlx,clx,hrx,crx):
		hx = hlx + hrx
		g = torch.mv(self.w_ih,input)+self.b_ih+torch.mv(self.w_hh,hx)+self.b_hh
		gl = torch.mv(self.wl_ih,input)+self.bl_ih+torch.mv(self.wl_hh,hlx)+self.bl_hh
		gr = torch.mv(self.wr_ih,input)+self.br_ih+torch.mv(self.wr_hh,hrx)+self.br_hh

		i,c,o = g.chunk(3)

		ingate = F.sigmoid(i)
		cellgate = F.tanh(c)
		outgate = F.sigmoid(o)
		forgetgatel = F.sigmoid(gl)
		forgetgater = F.sigmoid(gr)

		cy = (ingate*cellgate) + (forgetgatel*clx) + (forgetgater*crx)
		hy = outgate*F.tanh(cy)

		return hy,cy


if __name__=='__main__':
    model=TreeLstmCell(10,10)
    a=torch.autograd.Variable(torch.randn(10))
    b=torch.autograd.Variable(torch.randn(10))
    c=torch.autograd.Variable(torch.randn(10))
    d=torch.autograd.Variable(torch.randn(10))
    e=torch.autograd.Variable(torch.randn(10))
    print model(a,b,c,d,e)
