import json
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


def preprocess_data(file_path, encoders=None, label_encoder=None, imputer=None, scaler=None, is_training=True):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = file.read()
        data = json.loads(json_data)
    except FileNotFoundError:
        print(f"错误: 未找到文件 {file_path}。")
        return None, None, None, None, None, None
    except json.JSONDecodeError:
        print("错误: 无法解析 JSON 数据。")
        return None, None, None, None, None, None

    # 数据探查
    print("\n数据探查:")
    print(f"总记录数: {len(data)}")
    
    # 检查当前车速字段的数据类型分布
    speed_types = {}
    for item in data:
        if "当前车速" in item:
            value = item["当前车速"]
            value_type = type(value).__name__
            speed_types[value_type] = speed_types.get(value_type, 0) + 1
    print(f"当前车速字段的数据类型分布: {speed_types}")

    # 定义需要处理的字段
    category_fields = ["环境类型", "物体类型", "行为状态", "相对位置"]
    numeric_fields = ["当前车速"]
    boolean_fields = [
        "牵引连接状态", "托板平台有车", "反光锥存在",
        "伸缩臂动作", "轮胎托架动作"
    ]

    # 提取所有类别型字段的值
    category_values = {field: [] for field in category_fields}
    all_obj_features = []  # 存储所有环境物体特征
    
    for index, item in enumerate(tqdm(data, desc="正在提取类别型字段的值...")):
        # 处理环境类型
        try:
            category_values["环境类型"].append(item["环境类型"])
        except KeyError:
            print(f"在第 {index} 条记录中，字段 环境类型 缺失，记录内容为: {item}")
            category_values["环境类型"].append(None)

        # 处理摄像头数据中的类别字段
        try:
            camera_data = item["摄像头数据"]
            for camera_id, camera in camera_data.items():
                if "环境物体" in camera:
                    for obj in camera["环境物体"]:
                        obj_features = {}
                        for field in ["物体类型", "行为状态", "相对位置"]:
                            try:
                                value = obj[field]
                                category_values[field].append(value)
                                obj_features[field] = value
                            except KeyError:
                                # 检查英文形式的字段名
                                if field == '相对位置' and 'relative_position' in obj:
                                    value = obj['relative_position']
                                    category_values[field].append(value)
                                    obj_features[field] = value
                                else:
                                    print(f"在第 {index} 条记录的摄像头 {camera_id} 数据中，字段 {field} 缺失")
                                    category_values[field].append(None)
                                    obj_features[field] = None
                        all_obj_features.append(obj_features)
        except KeyError:
            print(f"在第 {index} 条记录中，摄像头数据字段缺失，记录内容为: {item}")

    # 处理数值型字段，添加类型检查
    numeric_values = []
    for item in tqdm(data, desc="正在处理数值型字段..."):
        try:
            value = item[numeric_fields[0]]
            # 确保值是数值类型或可转换为数值
            if isinstance(value, (int, float)):
                numeric_values.append([value])
            else:
                try:
                    numeric_values.append([float(value)])
                except (ValueError, TypeError):
                    print(f"警告: 记录 {item} 的字段 {numeric_fields[0]} 包含非数值值 '{value}'，将其视为缺失值")
                    numeric_values.append([np.nan])
        except KeyError:
            print(f"警告: 记录 {item} 缺少数值型字段 {numeric_fields[0]}，将其视为缺失值")
            numeric_values.append([np.nan])

    numeric_values = np.array(numeric_values)

    # 归一化数值型字段
    if scaler is None:
        scaler = StandardScaler()
        numeric_values = scaler.fit_transform(numeric_values)
    else:
        numeric_values = scaler.transform(numeric_values)

    # 处理布尔型字段
    all_bool_lengths = []
    boolean_values = []
    for item in tqdm(data, desc="正在处理布尔型字段..."):
        bool_vals = []
        try:
            camera_data = item["摄像头数据"]
            for camera_id, camera in camera_data.items():
                # 处理牵引检测
                if "牵引检测" in camera:
                    for field in boolean_fields[:1]:
                        try:
                            bool_vals.append(int(camera["牵引检测"][field]))
                        except KeyError:
                            print(f"在记录的摄像头 {camera_id} 牵引检测中，布尔型字段 {field} 缺失")
                            bool_vals.append(0)
                else:
                    bool_vals.extend([0] * len(boolean_fields[:1]))
                
                # 处理驮载检测
                if "驮载检测" in camera:
                    for field in boolean_fields[1:3]:
                        try:
                            bool_vals.append(int(camera["驮载检测"][field]))
                        except KeyError:
                            print(f"在记录的摄像头 {camera_id} 驮载检测中，布尔型字段 {field} 缺失")
                            bool_vals.append(0)
                else:
                    bool_vals.extend([0] * len(boolean_fields[1:3]))
                
                # 处理臂架状态
                if "臂架状态" in camera:
                    for field in boolean_fields[3:]:
                        try:
                            bool_vals.append(int(camera["臂架状态"][field]))
                        except KeyError:
                            print(f"在记录的摄像头 {camera_id} 臂架状态中，布尔型字段 {field} 缺失")
                            bool_vals.append(0)
                else:
                    bool_vals.extend([0] * len(boolean_fields[3:]))
            
            boolean_values.append(bool_vals)
            all_bool_lengths.append(len(bool_vals))
        except KeyError:
            print(f"在记录中，摄像头数据字段缺失，记录内容为: {item}")
            boolean_values.append([0] * (len(boolean_fields) * len(camera_data)))
            all_bool_lengths.append(len(boolean_fields) * len(camera_data))

    # 确保所有样本的布尔特征长度一致
    max_bool_length = max(all_bool_lengths) if all_bool_lengths else 0
    for i in range(len(boolean_values)):
        while len(boolean_values[i]) < max_bool_length:
            boolean_values[i].append(0)

    # 如果是训练模式，创建编码器；否则使用传入的编码器
    if is_training or encoders is None:
        encoders = {field: OneHotEncoder(handle_unknown='ignore') for field in category_fields}
        for field in category_fields:
            values = np.array(category_values[field]).reshape(-1, 1)
            encoders[field].fit(values)

        # 数值型字段的imputer
        imputer = SimpleImputer(strategy='mean')
        if len(numeric_values) > 0:
            imputer.fit(numeric_values)

        # 标签编码器
        le = LabelEncoder()
        labels = [item.get("工单状态", None) for item in data if "工单状态" in item]
        if len(labels) > 0:
            le.fit(labels)
    else:
        le = label_encoder

    # 使用编码器转换数据
    encoded_category = {}
    for field in category_fields:
        values = np.array(category_values[field]).reshape(-1, 1)
        encoded_category[field] = encoders[field].transform(values).toarray()

    # 统计训练时的最大环境物体数量
    if is_training:
        max_obj_count = 0
        for item in data:
            try:
                camera_data = item["摄像头数据"]
                obj_count = 0
                for camera in camera_data.values():
                    if "环境物体" in camera:
                        obj_count += len(camera["环境物体"])
                if obj_count > max_obj_count:
                    max_obj_count = obj_count
            except KeyError:
                pass
    else:
        max_obj_count = encoders.get('max_obj_count', 0)  # 从编码器中获取训练时的最大环境物体数量

    # 合并特征
    processed_data = []
    obj_index = 0
    for i, item in enumerate(tqdm(data, desc="正在合并特征...")):
        features = list(numeric_values[i]) + list(boolean_values[i])

        # 添加环境类型
        if i < len(encoded_category["环境类型"]):
            features.extend(encoded_category["环境类型"][i])

        # 处理环境物体
        current_obj_count = 0
        try:
            camera_data = item["摄像头数据"]
            for camera_id, camera in camera_data.items():
                if "环境物体" in camera:
                    for obj in camera["环境物体"]:
                        for field in ["物体类型", "行为状态", "相对位置"]:
                            if obj_index < len(encoded_category[field]):
                                features.extend(encoded_category[field][obj_index])
                            else:
                                features.extend([0] * len(encoders[field].categories_[0]))
                        obj_index += 1
                        current_obj_count += 1
        except KeyError:
            print(f"在记录中，摄像头数据字段缺失，记录内容为: {item}")

        # 填充缺失的环境物体特征，使用训练时的最大数量
        while current_obj_count < max_obj_count:
            for field in ["物体类型", "行为状态", "相对位置"]:
                features.extend([0] * len(encoders[field].categories_[0]))
            current_obj_count += 1

        processed_data.append(features)

    # 转换为 NumPy 数组
    processed_data = np.array(processed_data)

    # 提取工单状态标签
    labels = [item.get("工单状态", None) for item in data if "工单状态" in item]
    if len(labels) > 0 and le is not None:
        encoded_labels = le.transform([l if l is not None else "未知" for l in labels])
    else:
        encoded_labels = np.zeros(len(data))

    if is_training:
        encoders['max_obj_count'] = max_obj_count  # 保存到编码器中

    return processed_data, encoded_labels, le, encoders, imputer, scaler 
