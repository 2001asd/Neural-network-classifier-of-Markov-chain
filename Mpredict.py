import torch
import torch.nn as nn
import numpy as np
import json
from preprocessing import preprocess_data
from tqdm import tqdm

# DNN模型定义（与训练脚本保持一致）
class DNNModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

def load_model(model_path: str):
    """加载训练好的模型和相关预处理参数"""
    # 设置weights_only=True提高安全性
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # 初始化模型
    model = DNNModel(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 加载预处理参数
    encoders = checkpoint['encoders']
    label_encoder = checkpoint['label_encoder']
    imputer = checkpoint['imputer']
    scaler = checkpoint['scaler']
    
    return model, encoders, label_encoder, imputer, scaler, checkpoint['num_classes']

def predict(model: DNNModel, data_path: str, encoders: dict, label_encoder, 
            imputer, scaler, num_classes: int):
    """使用训练好的模型进行预测，序列预测逻辑"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 预处理数据（不包含标签）
    processed_data, _, _, _, _, _ = preprocess_data(
        data_path, 
        encoders=encoders, 
        label_encoder=label_encoder,
        imputer=imputer,
        scaler=scaler,
        is_training=False
    )
    
    predicted_indices = []
    prev_state_onehot = np.zeros(num_classes)  # 初始状态为全0（首帧无前状态）
    
    for i in range(len(processed_data)):
        # 拼接特征：当前特征 + 前状态独热编码
        features = np.concatenate([processed_data[i], prev_state_onehot])
        
        # 转换为张量
        X_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(X_tensor)
            _, current_pred = torch.max(outputs, 1)
            current_pred = current_pred.item()
        
        predicted_indices.append(current_pred)
        
        # 更新前状态独热编码（用当前预测结果生成下一帧的前状态特征）
        prev_state_onehot = np.zeros(num_classes)
        prev_state_onehot[current_pred] = 1
    
    # 转换为标签
    predicted_labels = label_encoder.inverse_transform(np.array(predicted_indices))
    
    return predicted_labels, np.array(predicted_indices)

def main():
    model_path = 'dnn_model.pth'  # 训练保存的模型路径
    test_data_path = '输入要预测的josn路径'  # 测试数据路径
    
    # 加载模型和参数
    model, encoders, label_encoder, imputer, scaler, num_classes = load_model(model_path)

    # 预测
    predicted_labels, predicted_indices = predict(
        model, 
        test_data_path, 
        encoders, 
        label_encoder, 
        imputer, 
        scaler,
        num_classes=num_classes
    )
    
    # 生成结果列表
    result_list = []
    for index, label in enumerate(predicted_labels):
        result_list.append({
            "image_index": index,
            "工单状态": label
        })
    
    # 保存预测结果
    result_path = 'prediction_results.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)
    
    print(f"预测结果已保存到: {result_path}")

if __name__ == "__main__":
    main()
