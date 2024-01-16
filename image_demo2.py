
'''
 Before Augmentation
'''
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import datetime
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import cv2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
import torchvision.models as models
data = 'dataset2'
classes = {idx:cls_name for idx, cls_name in enumerate(os.listdir(data))} # enumerate là hàm để tạo thêm idx cho từng class
classes
x =[]
y =[]
for cls_id, cls_name in classes.items(): # duyệt qua ind và name của các class trong all ảnh
    cls_dir = os.path.join(data,cls_name)
    for img_filename in os.listdir(cls_dir):
        img_path = os.path.join(cls_dir,img_filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # ảnh xám(h,w,1)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # ảnh màu (h,w,3)
                x.append(img)
                y.append(cls_id)
            else:
                pass  
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=42)
class MosquitoDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])
# dành cho ảnh màu 
# train_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
# test_transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

train_dataset = MosquitoDataset(x=x_train, y=y_train, transform=train_transform)
test_dataset = MosquitoDataset(x=x_test, y=y_train, transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=52, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
train_imgs, train_labels = next(iter(train_loader))
train_img = train_imgs[0].permute(1, 2, 0).numpy()  # Chuyển về định dạng numpy
plt.imshow(train_img)
plt.show()
# mô tả CNNModel
class CNNModel(nn.Module):
    def __init__(self, n_classes=3, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size =2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate)  # Dropout hạn chế overfitting 
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(16, 32,kernel_size =2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate)  
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size =2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate) 
        )
        self.conv_layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size =2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(dropout_rate) 
        )

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)  

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128,n_classes ),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.global_avgpool(x)
        out = self.classifier(x)
        return out
''' rút gọn CNNModel
class CNNModel(nn.Module):
    def __init__(self, num_labels, dropout_rate=0.5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_labels)
        self.layers = [nn.MaxPool2d(2), nn.ReLU(),nn.Dropout(dropout_rate)]
    def forward(self, x):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4]:
            x = self.layers[2](self.layers[1](self.layers[0](F.relu(layer(x)))))
        x = self.global_pool(x)
        x = self.fc(x.view(x.size(0), -1))
        return x 
'''
model = CNNModel(3)
model.to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)  
decayRate = 0.96
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
def evaluate(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    running_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
    accuracy = 100 * running_correct / total
    test_loss = test_loss / len(test_loader)
    return test_loss, accuracy
# Tạo một callback để lưu mô hình checkpoint sau mỗi epoch
def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    return torch.save(checkpoint, 'model_checkpoint.pth')
t0 = datetime.datetime.now() 
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
max_epoch = 30
early_stopping_counter = 0
best_test_loss = float('inf')
for epoch in range(max_epoch):
    model.train()
    running_loss = 0.0
    running_correct = 0   
    total = 0             
    for i, (inputs, labels) in enumerate(train_loader, 0):
        inputs, labels = inputs.to(device), labels.to(device) 
        optimizer.zero_grad() # cập nhật giá trị gradient = 0 tránh việc cộng dồn gradient
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()  
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        running_correct += (predicted == labels).sum().item()
    epoch_accuracy = 100 * running_correct / total
    epoch_loss = running_loss / (i + 1)
    test_loss, test_accuracy = evaluate(model, test_loader, criterion)
    print(f"Epoch [{epoch + 1}/{max_epoch}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, test Loss: {test_loss:.4f}, test Accuracy: {test_accuracy:.2f}%")
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    lr_scheduler.step()
    # Lưu checkpoint sau mỗi epoch
    save_checkpoint(epoch, model, optimizer, loss.item())
    # Kiểm tra early stopping
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= 10:
        print("Early stopping triggered.")
        break
t1 = datetime.datetime.now() # tính time thực thi model
print(t1-t0)
# the best accuracy
best_epoch = np.argmax(test_accuracies)
print(f"\nBest epoch: {best_epoch + 1} with test accuracy: {test_accuracies[best_epoch]:.2f}%")
plt.plot(train_losses, label='train_losses')
plt.plot(val_losses, label='val_losses')
plt.legend()

# Định nghĩa lại kiến trúc mô hình
model = CNNModel(3)
model.to(device)
criterion = nn.CrossEntropyLoss() 
checkpoint = torch.load('model_checkpoint.pth') # Tải checkpoint
model.load_state_dict(checkpoint['model_state_dict'])# Load trạng thái của mô hình
optimizer = optim.Adam(model.parameters(), lr=1e-3)  
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval() # Đặt mô hình vào chế độ đánh giá
# Test trên dữ liệu kiểm thử
test_loss, test_accuracy = evaluate(model, test_loader, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
# Sử dụng mô hình để dự đoán hoặc tiếp tục huấn luyện

#=>> chưa thu thập được ảnh nhiều =>> cần data Augmentation để tạo thêm các phiên bản mới có tính đa dạng, giúp mô hình học được các đặc trưng tổng quát hơn và tránh overfitting
''' 
mô hình sau khi sử dụng data Augmentation
'''

class MosquitoDataset(Dataset):
      def __init__(self, x, y, transform=None):
            self.x = x
            self.y = y
