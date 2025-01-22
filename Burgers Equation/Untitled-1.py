# %%
import torch
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn 
import torch.nn.functional as F
import math

torch.manual_seed(0)

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mu = torch.tensor(0.01)

# %%
class Forwardnett(nn.Module):
    def __init__(self):
        super(Forwardnett,self).__init__()
        self.L1 = nn.Linear(102,100)
        self.L2 = nn.Linear(100,100)
        self.L3 = nn.Linear(100,100)
        self.L4 = nn.Linear(100,100)
        self.L5 = nn.Linear(100,100)
        
    def forward(self,y_0,x_loc_and_time):
        input_final = torch.cat((y_0,x_loc_and_time),-1)

        b = F.tanh(self.L1(input_final))
        b = F.tanh(self.L2(b))
        b = F.tanh(self.L3(b))
        b = F.tanh(self.L4(b))
        b = self.L5(b)

        return b 

# %%
model = Forwardnett().to(device)
database = pd.read_csv('sin_pix.csv',index_col=0).dropna().to_numpy(dtype='float32')

# %%
class Data(Dataset):
    def __init__(self,transform=None):
        self.initial_conditions = torch.from_numpy(database[:,0:100])#.requires_grad_(True)
        self.x_location = torch.from_numpy(database[:,[100]])#.requires_grad_(True)
        self.time_vale = torch.from_numpy(database[:,[101]])#.requires_grad_(True)
        self.true_y_value = torch.from_numpy(database[:,[102]])#.requires_grad_(True)
        self.n_samples = database.shape[0]

    def __getitem__(self, index):
        return self.initial_conditions[index] , self.x_location[index] , self.time_vale[index] , self.true_y_value[index]
    
    def __len__(self):
        return self.n_samples
    
dataset_data = Data()

# %%
import torch.utils
import torch.utils.data


train_size = int(0.7*dataset_data.__len__())
test_size = dataset_data.__len__()-train_size

batch_size = 500

Burger_train_data , Burger_test_data = torch.utils.data.random_split(Data(),[train_size,test_size])

train_loader = DataLoader(dataset=Burger_train_data,batch_size=batch_size,shuffle=True)

#data_iter = iter(train_loader)
#data = data_iter.__next__()
#Init_val , x_loc, time, y_value = data

# %%
num_epoch = 200
total_samples = len(train_loader)
n_iterations = math.ceil(num_epoch*total_samples/(batch_size))
print(total_samples,n_iterations)
learning_rate = 0.01

criterion = nn.MSELoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.00001)

loss_rec = []

# %%
model.train()

for epoch in range(num_epoch):
    for i , (input_init_conditions,input_x_loc,input_time,Actual_y) in enumerate(train_loader):
        input1 = input_init_conditions
        input1 = input1.to(device)

        input2 = torch.cat((input_x_loc,input_time),-1)
        input2 = input2.to(device)

        Actual_y = Actual_y.to(device)

        Outputs = model(input1,input2)

        #input2_BC1 = torch.cat((torch.zeros(input_time.size(0),1),input_time),-1).to(device)
        #target_BC1 = torch.zeros(input_time.size(0))
        #predicted_BC1 = model(input1,input2_BC1)
        #loss_BC1 = torch.mean((predicted_BC1-target_BC1)**2)

        #input2_BC2 = torch.cat((torch.ones(input_time.size(0),1),input_time),-1).to(device)
        #target_BC2 = torch.zeros(input_time.size(0))
        #predicted_BC2 = model(input1,input2_BC2)
        #loss_BC2 = torch.mean((predicted_BC2-target_BC2)**2)

        #Physics_loss = 

        #Physics_loss = torch.mean()
        
        loss = criterion(Outputs,Actual_y) # + 1000*(loss_BC1 + loss_BC2)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()    

        loss_rec.append(loss.item())

        if (i+1) % 10 ==0:
            print(f'Epoch [{epoch+1}/{num_epoch}] , Step [{i+1}/{total_samples}] , Loss: {loss.item():.16f}')


