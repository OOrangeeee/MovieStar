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
