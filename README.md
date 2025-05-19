# 一、项目概述
本项目基于深度学习构建工单状态预测模型，包含数据预处理、模型训练及预测功能。通过处理 JSON 格式的传感器数据，提取环境类型、物体状态等特征，使用深度神经网络（DNN）实现工单状态的序列预测。
# 二、目录结构
project/  
├─ preprocessing.py    # 数据预处理脚本  
├─ Mtrain.py           # 模型训练脚本  
├─ Mpredict.py         # 模型预测脚本  
├─ dnn_model.pth       # 训练好的模型文件（需手动添加）  
├─ README.md           # 项目说明  
└─ requirements.txt    # 依赖清单  
# 三、环境依赖
## 3.1 安装依赖
pip install -r requirements.txt  
# 四、数据准备
训练数据：按照py文件说明
预测数据：结构与训练数据类似，但无需包含工单状态字段。
# 五、运行脚本  
python Mtrain.py  
python Mpredict.py  

