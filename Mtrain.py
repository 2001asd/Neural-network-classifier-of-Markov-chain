import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from preprocessing import preprocess_data
import numpy as np
import json

# DNN模型定义
class DNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(DNNModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size//2, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)

# 按工单状态划分训练和验证
def split_data_by_label(X, y, test_size=0.2):
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        if len(label_indices) == 0:
            continue
            
        num_val = max(1, int(len(label_indices) * test_size))
        train_indices = label_indices[:-num_val]
        val_indices = label_indices[-num_val:]
        
        X_train.extend(X[train_indices])
        y_train.extend(y[train_indices])
        X_val.extend(X[val_indices])
        y_val.extend(y[val_indices])
    
    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val)

def main():
    file_path = '输入训练样本json路径'
    processed_data, encoded_labels, le, encoders, imputer, scaler = preprocess_data(file_path)
    if processed_data is None:
        return

    num_classes = len(le.classes_)
    input_size = processed_data.shape[1] + num_classes  # 原特征 + 前状态独热编码维度

    # 准备带前状态特征的数据
    X, y = [], []
    for i in range(len(processed_data)):
        if i == 0:
            prev_state_onehot = np.zeros(num_classes)  # 首帧无前状态，用全0
        else:
            prev_state = encoded_labels[i-1]
            prev_state_onehot = np.zeros(num_classes)
            prev_state_onehot[prev_state] = 1  # 前状态独热编码
        
        # 拼接特征：原特征 + 前状态独热编码
        features = np.concatenate([processed_data[i], prev_state_onehot])
        X.append(features)
        y.append(encoded_labels[i])
    
    X = np.array(X)
    y = np.array(y)

    # 划分数据集
    X_train, y_train, X_val, y_val = split_data_by_label(X, y, test_size=0.2)

    # 转换为张量
    X_train = torch.tensor(X_train, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    y_train = torch.tensor(y_train, dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')
    X_val = torch.tensor(X_val, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    y_val = torch.tensor(y_val, dtype=torch.long).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 模型参数
    hidden_size = 128
    model = DNNModel(input_size, hidden_size, num_classes).to('cuda' if torch.cuda.is_available() else 'cpu')

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    # 训练参数
    num_epochs = 500
    best_val_accuracy = 0.0
    model_save_path = 'dnn_model.pth'
    saved = False

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (val_predicted == y_val).sum().item() / len(y_val)

            # 保存最佳模型
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                torch.save({
                   'model_state_dict': model.state_dict(),
                    'encoders': encoders,
                    'label_encoder': le,
                    'imputer': imputer,
                    'scaler': scaler,
                    'input_size': input_size,
                    'hidden_size': hidden_size,
                    'num_classes': num_classes,
                }, model_save_path)
                saved = True

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Accuracy: {val_accuracy * 100:.2f}%')

    if saved:
        print(f"模型已保存到: {model_save_path}")
    print(f'Best Validation Accuracy: {best_val_accuracy * 100:.2f}%')

if __name__ == "__main__":
    main()
