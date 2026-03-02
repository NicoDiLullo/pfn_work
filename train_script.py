import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from tqdm import tqdm
#from tqdm.notebook import tqdm


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return torch.mean(
            torch.abs((y_true - y_pred) / (y_true + self.eps))
        )

    
    
class ParticleLevelLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ParticleLevelLinear, self).__init__()
        self.weights = nn.Parameter((torch.rand(input_dim, output_dim) * 2 / input_dim**0.5) - 1/input_dim**0.5)
        self.bias = nn.Parameter((torch.rand(output_dim) * 2 / input_dim**0.5) - 1/input_dim**0.5)

    def forward(self, x):
        return torch.einsum('abc,cd->abd', x, self.weights) + self.bias

class PFN(nn.Module):
    def __init__(self, input_dim, L, phi_hidden_dim, f_hidden_dim):
        super(PFN, self).__init__()
        self.phi_fc1 = ParticleLevelLinear(input_dim, phi_hidden_dim)
        self.phi_fc2 = ParticleLevelLinear(phi_hidden_dim, phi_hidden_dim)
        self.phi_fc3 = ParticleLevelLinear(phi_hidden_dim, L)
        self.f_fc1 = nn.Linear(L, f_hidden_dim)
        self.f_fc2 = nn.Linear(f_hidden_dim, f_hidden_dim)
        self.f_fc3 = nn.Linear(f_hidden_dim, f_hidden_dim)
        self.f_fc4 = nn.Linear(f_hidden_dim, 1)
        print("PFN initialized with input_dim:", input_dim, "L:", L, "phi_hidden_dim:", phi_hidden_dim, "f_hidden_dim:", f_hidden_dim, "output_dim:", 1)
        print("Total number of parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        x = F.relu(self.phi_fc1(x))
        x = F.relu(self.phi_fc2(x))
        x = self.phi_fc3(x)  # (N, P, L)
        x = (x).sum(dim=1)  # Sum over particles, masking out zero padded ones
        x = F.relu(self.f_fc1(x))
        x = F.relu(self.f_fc2(x))
        x = F.relu(self.f_fc3(x))
        x = self.f_fc4(x)
        return x[:, 0]


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

def train_func(model, optimizer, loss_fn, trainset, valset, batchsize, save_path, num_epochs=100, early_stopping=None):
    trainloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)


    best_epoch = 0
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_val_loss = 0.0
        num_train_batches = len(trainloader)
        num_val_batches = len(valloader)
        train_loop = tqdm(trainloader, total=len(trainloader), desc=f"Epoch {epoch} Training")
        
        for data in train_loop:
            x_batch, y_batch = data
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / num_train_batches
        train_losses.append(train_loss)
        model.eval()
        
        val_loop = tqdm(valloader, total=len(valloader), desc=f"Epoch {epoch} Validation")
        for data in val_loop:
            x_val, y_val = data
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            val_output = model(x_val)
            val_loss = loss_fn(val_output, y_val)
            running_val_loss += val_loss.item()
        val_loss = running_val_loss / num_val_batches
        val_losses.append(val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch} with Val MAPE: {best_loss:.6f}")
        if early_stopping and epoch - best_epoch >= early_stopping:
            print(f'Early stopping at epoch {epoch+1}')
            break
    model.load_state_dict(best_model)
    return train_losses, val_losses


##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################
##########################################################################################################################################

X_train = np.load('/users/rrjain/PFN_models/pfn_pytorch/train_features.npy')
X_val = np.load('/users/rrjain/PFN_models/pfn_pytorch/val_features.npy')

Y_train = np.load('/users/rrjain/PFN_models/pfn_pytorch/train_targets.npy')
Y_val = np.load('/users/rrjain/PFN_models/pfn_pytorch/val_targets.npy')

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
Y_val = torch.tensor(Y_val, dtype=torch.float32)


train_dataset = TensorDataset(X_train, Y_train)
val_dataset = TensorDataset(X_val, Y_val)

model = PFN(4, 256, 800, 800)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = MAPELoss()

pfn_train_losses, pfn_val_losses = train_func(model, optimizer, loss_fn, train_dataset, val_dataset, 
                                              save_path = 'pfn_pytorch_50pruned.pth', 
                                              num_epochs=500, early_stopping=50, batchsize = 1024)


np.save('train_loss.npy', pfn_train_losses)
np.save('val_loss.npy', pfn_val_losses)
