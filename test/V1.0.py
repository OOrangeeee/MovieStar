#!/usr/bin/env python
# coding: utf-8

# 导入第三方库
import torch  # pytorch框架
import pandas as pd  # 数据集读入库
import matplotlib.pyplot as plt  # 绘制图表
import torch.nn as nn  # 模型中的网络层使用
from torch import optim  # 优化器使用
from torch.utils.data import DataLoader, TensorDataset  # 数据集处理
from sklearn.model_selection import train_test_split  # 数据划分

# 读取文件
movies_df = pd.read_csv("./movies.csv")
ratings_df = pd.read_csv("./ratings.csv")
tags_df = pd.read_csv("./tags.csv")

# movies_df  # 查看数据

movies_df.head(5)

merged_movies_rating_df = pd.merge(
    ratings_df, movies_df, on="movieId", how="inner"
)  # 将需要的数据进行整合到一份表中

merged_movies_rating_df["genres"].unique()  # 查看genres列中的唯一值

# 将"genres"列分割成多个列
genres_split = merged_movies_rating_df["genres"].str.get_dummies(
    "|"
)  # 使用get_dummies这个方法将genres这一列的数据分割出来形成新的特征列
# 将新生成的列与原始数据合并
df = pd.concat([merged_movies_rating_df, genres_split], axis=1)
# 删除generes列
df.drop(columns=["genres"], inplace=True)

# df  # 查看处理好的数据

# df["title"]  # 观察标题列

df["title"] = df["title"].astype("category").cat.codes  # 为电影标题创建唯一的整数编码

# df  # 查看数据

# df.info()  # 查看每个特征列的缺失值

# 根据用户ID和电影ID进行分组，使用少数服从多数的原则
unique_votes = df.groupby(["userId", "movieId"])["rating"].mean().round().reset_index()
# 删除同一movie_id的重复数据，只保留一个
df_unique = df.drop_duplicates(subset="movieId", keep="first")
# 合并唯一投票结果到原始数据集
df = pd.merge(
    df_unique,
    unique_votes,
    on=["userId", "movieId"],
    how="left",
    suffixes=("", "_unique"),
)
# 删除处理完后不需要的数据信息
df.drop(columns=["userId"])
df.drop(columns=["timestamp"])
df.drop(columns=["rating"])

# df.shape  # 观察数据量和特征列数量

# 数据划分选取特征列和目标列
features = df.iloc[:, df.columns != "rating_unique"].values
target = df.iloc[:, df.columns == "rating_unique"].values

# 划分第一个训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.3, random_state=42
)

# 划分第二个训练集和测试集
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(
    features, target, test_size=0.3, random_state=2021
)

# 划分第三个训练集和测试集
X_train_3, X_test_3, Y_train_3, Y_test_3 = train_test_split(
    features, target, test_size=0.3, random_state=2000
)

# 数据转换，转换成为能够使用torch训练类型和验证的类型
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
X_train_2_tensor = torch.tensor(X_train_2, dtype=torch.float32)
Y_train_2_tensor = torch.tensor(Y_train_2, dtype=torch.long)
X_test_2_tensor = torch.tensor(X_test_2, dtype=torch.float32)
Y_test_2_tensor = torch.tensor(Y_test_2, dtype=torch.long)
X_train_3_tensor = torch.tensor(X_train_3, dtype=torch.float32)
Y_train_3_tensor = torch.tensor(Y_train_3, dtype=torch.long)
X_test_3_tensor = torch.tensor(X_test_3, dtype=torch.float32)
Y_test_3_tensor = torch.tensor(Y_test_3, dtype=torch.long)

# 数据转换为批次进行训练
train_data = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
train_data_2 = TensorDataset(X_train_2_tensor, Y_train_2_tensor)
train_loader_2 = DataLoader(train_data_2, batch_size=128, shuffle=True)
train_data_3 = TensorDataset(X_train_3_tensor, Y_train_3_tensor)
train_loader_3 = DataLoader(train_data_3, batch_size=128, shuffle=True)


# 多层感知机模型参数
# 创建多层感知机模型，神经网络
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * ((input_size - 2) // 2), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 6)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.fc4(x)
        x = nn.functional.relu(x)
        x = self.fc5(x)
        x = nn.functional.relu(x)
        x = self.fc6(x)
        return x


# X_train.shape[1]  # 特征列的数量

model = MLP(X_train.shape[1])  # 输入量为特征列的数量
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 学习率定义和优化器定义
# 模型训练
epochs = 50  # 设置训练次数
model_acc = 0.0  # 用于保存模型最好的准确率
train_loss_data_1 = []  # 记录第一次的训练的损失
test_loss_data_1 = []  # 记录第一次的测试的损失
for epoch in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X.unsqueeze(1))
        loss = criterion(outputs, batch_y.squeeze())  # 计算损失值
        loss.backward()
        optimizer.step()
        train_loss_data_1.append(loss.item())
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")  # 输出训练次数和当前损失

    # 模型评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.unsqueeze(1))  # 模型进行预测
        test_loss = criterion(test_outputs, Y_test_tensor.squeeze())  # 计算测试损失值
        _, predicted_labels = torch.max(test_outputs, 1)  # 选择6个标签中概率最高
        accuracy = (predicted_labels == Y_test_tensor).float().mean().item()

    print(f"Test Accuracy: {accuracy * 100:.2f}%")  # 输出准确率
    if model_acc < accuracy * 100:
        model_acc = accuracy * 100  # 保留最好的准确率值
    test_loss_data_1.append(test_loss.item())  # 保存测试损失值，用于可视化
    print(f"Test Loss: {test_loss.item()}")  # 输出测试损失值

model2 = MLP(X_train_2.shape[1])
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 损失函数初始化
optimizer = optim.Adam(model2.parameters(), lr=0.001)  # 学习率定义和优化器定义

# 模型训练
epochs = 50  # 训练次数
train_loss_data_2 = []  # 记录第二次训练的损失
test_loss_data_2 = []  # 记录第二次的测试的损失
model2_acc = 0.0  # 用于保存第二次模型最好的准确率
for epoch in range(epochs):
    for batch_X, batch_y in train_loader_2:
        optimizer.zero_grad()
        outputs = model2(batch_X.unsqueeze(1))
        loss = criterion(outputs, batch_y.squeeze())  # 计算损失
        loss.backward()
        optimizer.step()
        train_loss_data_2.append(loss.item())  # 记录训练损失值，用于可视化
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")  # 输出训练次数和当前损失

    # 模型评估
    model2.eval()
    with torch.no_grad():
        test_outputs = model2(X_test_2_tensor.unsqueeze(1))  # 模型进行预测
        test_loss = criterion(test_outputs, Y_test_2_tensor.squeeze())  # 损失计算
        _, predicted_labels = torch.max(test_outputs, 1)  # 选择6个标签中概率最高
        accuracy = (predicted_labels == Y_test_2_tensor).float().mean().item()  # 计算准确率

    print(f"Test Accuracy: {accuracy * 100:.2f}%")  # 输出准确率
    if model2_acc < accuracy * 100:
        model2_acc = accuracy * 100
    test_loss_data_2.append(test_loss.item())  # 记录测试损失值，用于可视化
    print(f"Test Loss: {test_loss.item()}")

model3 = MLP(X_train_3.shape[1])
# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model3.parameters(), lr=0.001)  # 学习率定义和优化器定义

# 模型训练
epochs = 50  # 训练次数
model3_acc = 0.0  # 第三个模型最好的准确率
train_loss_data_3 = []  # 记录第三次的训练的损失
test_loss_data_3 = []  # 记录第三次的测试的损失
for epoch in range(epochs):
    for batch_X, batch_y in train_loader_3:
        optimizer.zero_grad()
        outputs = model3(batch_X.unsqueeze(1))
        loss = criterion(outputs, batch_y.squeeze())  # 计算损失值
        loss.backward()
        optimizer.step()
        train_loss_data_3.append(loss.item())  # 记录损失值，用于可视化
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")  # 输出训练次数和当前损失

    # 模型评估
    model3.eval()
    with torch.no_grad():
        test_outputs = model3(X_test_3_tensor.unsqueeze(1))
        test_loss = criterion(test_outputs, Y_test_3_tensor.squeeze())
        _, predicted_labels = torch.max(test_outputs, 1)
        accuracy = (predicted_labels == Y_test_3_tensor).float().mean().item()

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    if model3_acc < accuracy * 100:
        model3_acc = accuracy * 100
    test_loss_data_3.append(test_loss.item())
    print(f"Test Loss: {test_loss.item()}")

avg_acc = (model_acc + model2_acc + model3_acc) / 3
print(f"平均准确率:{avg_acc:.2f}")

# Y_test.shape

# 将 PyTorch 张量转为 NumPy 数组并去除维度为 1 的维度
# 绘制图表
plt.plot(range(50), train_loss_data_1)
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.title("First train_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(50), test_loss_data_1)
plt.xlabel("epoch")
plt.ylabel("test_loss")
plt.title("First Test_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(50), train_loss_data_2)
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.title("Second train_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(50), test_loss_data_2)
plt.xlabel("epoch")
plt.ylabel("test_loss")
plt.title("Second test_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(50), train_loss_data_3)
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.title("Thrid train_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(50), test_loss_data_3)
plt.xlabel("epoch")
plt.ylabel("test_loss")
plt.title("Third test_Loss")
plt.legend()
plt.show()
