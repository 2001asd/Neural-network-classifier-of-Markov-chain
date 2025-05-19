# 工单状态预测系统
## 一、项目概述
本项目基于深度学习构建工单状态预测模型，包含数据预处理、模型训练及预测功能。通过处理JSON格式的传感器数据，提取环境类型、物体状态等特征，使用深度神经网络（DNN）实现工单状态的序列预测。
## 二、目录结构
project/
├─ preprocessing.py # 数据预处理脚本
├─ Mtrain.py # 模型训练脚本
├─ Mpredict.py # 模型预测脚本
├─ dnn_model.pth # 训练好的模型文件（需手动添加）
├─ README.md # 项目说明
└─ requirements.txt # 依赖清单
## 三、环境依赖
### 3.1 安装依赖
```bash
pip install -r requirements.txt
## 四、数据准备
预测数据：结构与训练数据类似，但无需包含工单状态字段。
##五、模型训练
# 运行训练脚本
python Mtrain.py
六、网络结构
plaintext
输入层 (input_size)
├── 全连接层 (hidden_size) + ReLU + Dropout(0.2)
├── 全连接层 (hidden_size/2) + ReLU + Dropout(0.1)
└── 输出层 (num_classes)
输入维度：原始特征 + 前一状态独热编码（共input_size维）。
输出维度：工单状态类别数（通过 Softmax 输出概率）。
