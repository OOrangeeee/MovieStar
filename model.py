from Accuracy import CustomAccuracy


# NCF

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Flatten,
    Dense,
    Concatenate,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2


class NCFModel:
    """
    NCF模型处理电影评分
    """

    def __init__(
        self, movie_id_max, user_id_max, tag_max, genres_dim, embedding_size=50
    ):
        """
        NCF模型构造函数
        :param movie_id_max: 电影ID的最大值
        :param user_id_max: 用户ID的最大值
        :param tag_max: 标签的最大值
        :param genres_dim: 特征列数量
        :param embedding_size: 训练维度数量，越大效果越好但是容易过拟合
        """
        self.movie_id_max = movie_id_max
        self.user_id_max = user_id_max
        self.tag_max = tag_max
        self.genres_dim = genres_dim
        self.embedding_size = embedding_size
        self.model = self._create_model()

    def _create_model(self):
        """
        构造模型
        :return:返回模型
        """

        # 输入层
        # 输入电影ID，用户ID，标签对应的值，特征列
        # 前三个输出都是一维数组，最后的是20维数组，因为有二十种不同的特征
        movie_id_input = Input(shape=(1,), name="movie_id_input")
        user_id_input = Input(shape=(1,), name="user_id_input")
        tag_input = Input(shape=(1,), name="tag_input")
        genres_input = Input(shape=(self.genres_dim,), name="genres_input")

        # 嵌入层
        # 对前三个数据做嵌入，最后一个数据无需嵌入
        movie_embedding = Embedding(
            output_dim=self.embedding_size,
            input_dim=self.movie_id_max,
            name="movie_embedding",
        )(movie_id_input)
        user_embedding = Embedding(
            output_dim=self.embedding_size,
            input_dim=self.user_id_max,
            name="user_embedding",
        )(user_id_input)
        tag_embedding = Embedding(
            output_dim=self.embedding_size, input_dim=self.tag_max, name="tag_embedding"
        )(tag_input)

        # 平展层
        # 将嵌入的结果展平为一维数组
        movie_vec = Flatten(name="flatten_movie")(movie_embedding)
        user_vec = Flatten(name="flatten_user")(user_embedding)
        tag_vec = Flatten(name="flatten_tag")(tag_embedding)
        genres_vec = genres_input

        # 合并
        # 合并为最终结果
        concat = Concatenate()([movie_vec, user_vec, tag_vec, genres_vec])
        print(concat.shape)

        # MLP
        mlp = Dense(128, activation="relu")(concat)
        mlp = Dropout(0.5)(mlp)
        mlp = Dense(64, activation="relu")(mlp)
        mlp = Dropout(0.5)(mlp)
        # mlp = Dense(32, activation="relu")(mlp)
        # mlp = Dropout(0.3)(mlp)

        # 预测评分
        rating_prediction = Dense(1, activation="sigmoid")(mlp)

        # 构建模型
        model = Model(
            inputs=[
                movie_id_input,
                user_id_input,
                tag_input,
                genres_input,
            ],
            outputs=rating_prediction,
        )
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=[CustomAccuracy()],
        )
        return model

    def train(self, train_data, train_labels, callback, epochs=10, batch_size=32):
        """
        模型训练函数
        :param train_data: 训练数据集
        :param train_labels: 训练目标集
        :param callback: 用于记录训练过程
        :param epochs: 训练次数
        :param batch_size: 单次训练数据量
        :return:无
        """
        self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callback],
        )

    def evaluate(self, test_data, test_labels):
        """
        检验测试集
        :param test_data: 测试数据集
        :param test_labels: 测试目标集
        :return: 返回准确率和损失
        """
        return self.model.evaluate(test_data, test_labels)
