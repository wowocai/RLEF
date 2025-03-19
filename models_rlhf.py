import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers#2 or 3
        self.lstm = nn.LSTM(input_size-5, hidden_size, num_layers, batch_first=True)
        # self.fc1 = nn.Linear(5, hidden_size)
        self.fc2 = nn.Linear(hidden_size+5, 1)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = torch.FloatTensor(x).cuda()
        device = x.device
        # 初始化LSTM的隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers,x.size(0), self.hidden_size).to(device)
        # print(h0.shape)
        # print(c0.shape)

        # LSTM层
        indices = torch.tensor([0,1,2,3,4,5,6,7]).cuda()
        # print(x.shape)
        m = [x[0,0,i].item() for i in range(8,13)]
        # print(m)
        x = x.index_select(2,indices)
        # print(x.shape)
        # x = self.dropout(self.relu(x))
        out, _ = self.lstm(x, (h0, c0))

        out = out[:, 40:, :]  # 只获取最后10时间步的输出

        out5 = torch.zeros([1,10,5]).cuda()
        for i in range(5):
            out5[0,:,i] = m[i]
        # # print(out.shape)
        out  = torch.cat((out,out5),dim = -1)
        # print(out.shape)

        #dropout
        out = self.dropout(out)
        # 全连接层
        output = self.fc2(out).cuda()
    
        return output.reshape(1,10)
