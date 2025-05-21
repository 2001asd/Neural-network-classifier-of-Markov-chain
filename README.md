# 一、项目概述
本项目基于马尔科夫链模型和深度学习构建工单状态预测模型，包含数据预处理、模型训练及预测功能。通过处理 JSON 格式的传感器数据，提取环境类型、物体状态等当前帧的特征，并利用马尔科夫链性质，即把上一帧的工单状态引入当作本帧额外特征结合，使用深度神经网络（DNN）实现工单状态的序列预测。
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
训练数据：按照train.json文件说明，即可运行训练脚本，而trainupgrade.json是为后续想要进一步处理不同工单发生在同一时刻做数据准备。
预测数据：结构与训练数据类似，但无需包含工单状态字段。
# 五、运行脚本  
python Mtrain.py  
python Mpredict.py  

