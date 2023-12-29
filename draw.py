# 最后编辑：
# 晋晨曦 2023.12.20 7:08
# qq：2950171570
# email：Jin0714@outlook.com  回复随缘
import matplotlib.pyplot as plt


def draw_show_save(epochs, data, xlabel, ylabel, title):
    plt.plot(range(epochs), data)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    filepath = "./create_data/result/src/" + title + ".png"
    plt.savefig(filepath, format="png")
    plt.show()
    plt.close()
