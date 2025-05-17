import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import svd
import tensornetwork as tn 

class TensorLayer(nn.Module):
    def __init__(self, input_dim, bond_dim, output_dim):
        super(TensorLayer, self).__init__()
        self.input_dim = input_dim
        self.bond_dim = bond_dim
        self.output_dim = output_dim
        
        # Inicialización de los tensores
        self.tensor1 = nn.Parameter(torch.randn(input_dim, bond_dim))
        self.tensor2 = nn.Parameter(torch.randn(bond_dim, output_dim))
        
        # Normalización inicial
        with torch.no_grad():
            self.tensor1.data = self.tensor1.data / torch.norm(self.tensor1.data)
            self.tensor2.data = self.tensor2.data / torch.norm(self.tensor2.data)
    
    def forward(self, x):
        # Contraction del tensor
        batch_size = x.size(0)
        # Reshape para la multiplicación de matrices
        x = x.view(batch_size, self.input_dim, -1)
        # Primera multiplicación
        x = torch.matmul(x.transpose(1, 2), self.tensor1)
        # Segunda multiplicación
        x = torch.matmul(x, self.tensor2)
        return x

class TensorEEGNet(nn.Module):
    def __init__(self, F1=8, D=2, F2=16, dropout_rate=0.25, num_classes=2, bond_dim=4):
        super(TensorEEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.bond_dim = bond_dim

        # Capas convolucionales
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        self.conv2 = nn.Conv2d(F1, F1 * D, (2, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.conv3 = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F1 * D)
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm4 = nn.BatchNorm2d(F2)

        # Capa de red tensorial
        self.tensor_layer = TensorLayer(F2, bond_dim, F2)
        
        # Capa fully connected
        self.fc1 = nn.Linear(F2 * bond_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Procesamiento convolucional
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 4))
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = F.elu(x)
        x = F.avg_pool2d(x, (1, 8))
        x = self.dropout(x)
        
        # Reshape para la capa tensorial
        batch_size = x.size(0)
        x = x.view(batch_size, self.F2, -1)
        
        # Aplicar la capa tensorial
        x = self.tensor_layer(x)
        
        # Reshape para la capa fully connected
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        
        return x

def train_tensor_model(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test_tensor_model(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')

def main():
    # Configuración del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hiperparámetros
    batch_size = 64
    learning_rate = 1e-3
    epochs = 10

    # Datos sintéticos (reemplazar con datos reales)
    X_train = torch.randn(1000, 1, 2, 128)
    y_train = torch.randint(0, 2, (1000,))
    X_test = torch.randn(200, 1, 2, 128)
    y_test = torch.randint(0, 2, (200,))
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Crear y entrenar el modelo
    model = TensorEEGNet(num_classes=2, bond_dim=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train_tensor_model(model, device, train_loader, optimizer, criterion, epoch)
        test_tensor_model(model, device, test_loader, criterion)

if __name__ == "__main__":
    main() 