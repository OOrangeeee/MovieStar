import pandas as pd  # 数据集读入库
import numpy as np
import train_test as tt  # 划分训练集/测试集
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error,r2_score

# 导入数据文件
df = pd.read_csv("D:\project\MovieStar\create_data\df_data.csv")

# 删除第2列数据
df.drop(['title'],inplace=True,axis=1)

# 划分训练集测试集
x_train, x_test, y_train, y_test = tt.train_test(df, 0.2, 42)

#定义评估函数
def evaluation(model):
    model.fit(x_train, y_train.ravel())
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    score_train = model.score(x_train, y_train)
    score_test = model.score(x_test, y_test)
    print('  训练集Accuracy: ', score_train)
    print('  测试集Accuracy: ', score_test)

# ①随机森林回归
from sklearn.ensemble import RandomForestRegressor
model1=RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=6, random_state=42)
evaluation(model1)
# 相对好一点

# ②弹性网回归
from sklearn.linear_model import ElasticNet
model2 = ElasticNet(alpha=0.05, l1_ratio=0.5)
evaluation(model2)
# 拟合度太低，直接舍弃

# ③K近邻
from sklearn.neighbors import KNeighborsRegressor
model3 = KNeighborsRegressor(n_neighbors=10)
evaluation(model3)
# 速度慢，拟合度也低

# ④决策树
from sklearn.tree import DecisionTreeRegressor
model4 = DecisionTreeRegressor(random_state=42)
evaluation(model4)

# ⑤梯度提升
from sklearn.ensemble import GradientBoostingRegressor
model5=GradientBoostingRegressor(n_estimators=500,random_state=123)
evaluation(model5)
# 训练集很低，测试集和随机森林差不多

# ⑥极端梯度提升
from xgboost.sklearn import XGBRegressor
model6 = XGBRegressor(objective='reg:squarederror', n_estimators=1000, max_depth=6, min_child_weight=1,
                      random_state=42, learning_rate=0.3)
evaluation(model6)
# 未调参情况下有略高的准度

# ⑦轻量梯度提升
from lightgbm import LGBMRegressor
model7 = LGBMRegressor(n_estimators=1000,boosting_type='gbdt',objective='regression', learning_rate=0.5,
                       min_split_gain=0,min_child_weight=1,random_state=0)
evaluation(model7)










