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


def round_half_int(x):
    return round(x * 2) / 2


def compute_accuracy(predictions, targets, threshold=0.49):
    # param predictions: 预测值
    # param targets: 真实值
    # param threshold: 判定为正确预测的阈值
    correct = torch.abs(predictions - targets) <= threshold
    return correct.float().mean()


def round_to_nearest_half(tensor):
    return torch.round(tensor * 2) / 2


# 读取文件
movies_df = pd.read_csv("./movies.csv")
ratings_df = pd.read_csv("./ratings.csv")
tags_df = pd.read_csv("./tags.csv")

merged_movies_rating_df = pd.merge(
    ratings_df, movies_df, on="movieId", how="inner"
)  # 将需要的数据进行整合到一份表中

merged_movies_rating_df["genres"].unique()  # 查看genres列中的唯一值

# 将"genres"列分割成多个列
genres_split = merged_movies_rating_df["genres"].str.get_dummies(
    "|"
)  # 使用get_dummies这个方法将genres这一列的数据分割出来形成新的特征列
# 将标签分割开来

# 将新生成的列与原始数据合并
df = pd.concat([merged_movies_rating_df, genres_split], axis=1)

df.drop(columns=["genres"], inplace=True)

df["title"] = df["title"].astype("category").cat.codes  # 为电影标题创建唯一的整数编码

# 同一电影的打分取平均值
rating_df = df.groupby(["movieId"])["rating"].mean().apply(round_half_int).reset_index()

# 删除同一movie_id的重复数据，只保留一个
df_unique = df.drop_duplicates(subset="movieId", keep="first")

df_unique = df_unique.sort_values(by="movieId").reset_index()

df_unique.drop(columns=["rating"], inplace=True)

df_unique = pd.merge(df_unique, rating_df, on="movieId")

df_unique.drop(columns=["index"], inplace=True)

# 合并唯一投票结果到原始数据集
df = df_unique

df = df.sort_values(by=["userId", "movieId"]).reset_index()
df = df.drop(columns="index")

df = df.drop(columns=["userId"])
df = df.drop(columns=["timestamp"])

# 数据划分选取特征列和目标列
features = df.iloc[:, df.columns != "rating"].values
target = df.iloc[:, df.columns == "rating"].values

# 划分第一个训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.1, random_state=99
)

# 划分第二个训练集和测试集
X_train_2, X_test_2, Y_train_2, Y_test_2 = train_test_split(
    features, target, test_size=0.1, random_state=2021
)

# 划分第三个训练集和测试集
X_train_3, X_test_3, Y_train_3, Y_test_3 = train_test_split(
    features, target, test_size=0.1, random_state=2000
)

# 数据转换，转换成为能够使用torch训练类型和验证的类型
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.float32)
X_train_2_tensor = torch.tensor(X_train_2, dtype=torch.float32)
Y_train_2_tensor = torch.tensor(Y_train_2, dtype=torch.float32)
X_test_2_tensor = torch.tensor(X_test_2, dtype=torch.float32)
Y_test_2_tensor = torch.tensor(Y_test_2, dtype=torch.float32)
X_train_3_tensor = torch.tensor(X_train_3, dtype=torch.float32)
Y_train_3_tensor = torch.tensor(Y_train_3, dtype=torch.float32)
X_test_3_tensor = torch.tensor(X_test_3, dtype=torch.float32)
Y_test_3_tensor = torch.tensor(Y_test_3, dtype=torch.float32)

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
        self.fc6 = nn.Linear(16, 1)

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


model = MLP(X_train.shape[1])  # 输入量为特征列的数量

# 使用均方误差损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 50
model_acc = 0.0
train_loss_data_1 = []
test_loss_data_1 = []

for epoch in range(epochs):
    total_loss = 0  # 初始化累积损失值
    total_batches = 0  # 初始化批次计数
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X.unsqueeze(1))
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    average_loss = total_loss / total_batches
    train_loss_data_1.append(average_loss)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {average_loss}")

    # 模型评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.unsqueeze(1))
        rounded_outputs = torch.tensor([round_to_nearest_half(x) for x in test_outputs])
        test_loss = criterion(test_outputs, Y_test_tensor)  # 计算测试损失
        test_loss_data_1.append(test_loss.item())
        test_accuracy = compute_accuracy(rounded_outputs, Y_test_tensor)
        print(f"Test Loss: {test_loss.item()}")

    # 打印准确率
    print(f"Test Accuracy: {test_accuracy.item() * 100:.2f}%")
    if model_acc < test_accuracy * 100:
        model_acc = test_accuracy * 100
    test_loss_data_1.append(test_loss.item())
print(f"Test Accuracy max: {model_acc:.2f}%")

model2 = MLP(X_train_2.shape[1])  # 输入量为特征列的数量

# 使用均方误差损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(model2.parameters(), lr=0.003)

model2_acc = 0.0  # 用于保存第二次模型最好的准确率
train_loss_data_2 = []  # 记录第二次训练的损失
test_loss_data_2 = []  # 记录第二次的测试的损失

for epoch in range(epochs):
    total_loss = 0  # 初始化累积损失值
    total_batches = 0  # 初始化批次计数
    for batch_X, batch_y in train_loader_2:
        optimizer.zero_grad()
        outputs = model2(batch_X.unsqueeze(1))
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    average_loss = total_loss / total_batches
    train_loss_data_2.append(average_loss)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {average_loss}")

    # 模型评估
    model2.eval()
    with torch.no_grad():
        test_outputs = model2(X_test_2_tensor.unsqueeze(1))
        rounded_outputs = torch.tensor([round_to_nearest_half(x) for x in test_outputs])
        test_loss = criterion(test_outputs, Y_test_tensor)  # 计算测试损失
        test_loss_data_2.append(test_loss.item())
        test_accuracy = compute_accuracy(rounded_outputs, Y_test_2_tensor)
        print(f"Test Loss: {test_loss.item()}")

    # 打印准确率
    print(f"Test Accuracy: {test_accuracy.item() * 100:.2f}%")
    if model2_acc < test_accuracy * 100:
        model2_acc = test_accuracy * 100
    test_loss_data_2.append(test_loss.item())
print(f"Test Accuracy max: {model2_acc:.2f}%")

model3 = MLP(X_train_3.shape[1])  # 输入量为特征列的数量

# 使用均方误差损失函数
criterion = nn.MSELoss()
optimizer = optim.Adam(model3.parameters(), lr=0.003)

model3_acc = 0.0  # 用于保存第二次模型最好的准确率
train_loss_data_3 = []  # 记录第二次训练的损失
test_loss_data_3 = []  # 记录第二次的测试的损失

for epoch in range(epochs):
    total_loss = 0  # 初始化累积损失值
    total_batches = 0  # 初始化批次计数
    for batch_X, batch_y in train_loader_3:
        optimizer.zero_grad()
        outputs = model3(batch_X.unsqueeze(1))
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_batches += 1
    average_loss = total_loss / total_batches
    train_loss_data_3.append(average_loss)
    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {average_loss}")

    # 模型评估
    model3.eval()
    with torch.no_grad():
        test_outputs = model3(X_test_3_tensor.unsqueeze(1))
        rounded_outputs = torch.tensor([round_to_nearest_half(x) for x in test_outputs])
        test_loss = criterion(test_outputs, Y_test_tensor)  # 计算测试损失
        test_loss_data_3.append(test_loss.item())
        test_accuracy = compute_accuracy(rounded_outputs, Y_test_3_tensor)
        print(f"Test Loss: {test_loss.item()}")

    # 打印准确率
    print(f"Test Accuracy: {test_accuracy.item() * 100:.2f}%")
    if model3_acc < test_accuracy * 100:
        model3_acc = test_accuracy * 100
    test_loss_data_3.append(test_loss.item())
print(f"Test Accuracy max: {model3_acc:.2f}%")

avg_acc = (model_acc + model2_acc + model3_acc) / 3
print(f"平均准确率:{avg_acc:.2f}%")

# 绘制图表
plt.plot(range(epochs), train_loss_data_1, label="Training Loss")  # 添加 label 参数
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.title("First train_Loss")
plt.legend()  # 现在这里将显示"Training Loss"
plt.show()

# 绘制图表
plt.plot(range(epochs * 2), test_loss_data_1, label="Training Loss")
plt.xlabel("epoch")
plt.ylabel("test_loss")
plt.title("First Test_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(epochs), train_loss_data_2, label="Training Loss")
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.title("Second train_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(epochs * 2), test_loss_data_2, label="Training Loss")
plt.xlabel("epoch")
plt.ylabel("test_loss")
plt.title("Second test_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(epochs), train_loss_data_3, label="Training Loss")
plt.xlabel("epoch")
plt.ylabel("train_loss")
plt.title("Thrid train_Loss")
plt.legend()
plt.show()

# 绘制图表
plt.plot(range(epochs * 2), test_loss_data_3, label="Training Loss")
plt.xlabel("epoch")
plt.ylabel("test_loss")
plt.title("Third test_Loss")
plt.legend()
plt.show()
