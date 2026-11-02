import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(values, x_labels, y_labels, x_name, y_name, title="", fmt=".3f", ax=None):
    ax = ax if ax is not None else plt.gca()
    sns.heatmap(values, annot=True, fmt=fmt, xticklabels=x_labels,
                yticklabels=y_labels, cmap="viridis", ax=ax)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    return ax


def plot_confusion(cm, class_labels=None, title="", ax=None):
    ax = ax if ax is not None else plt.gca()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title)
    return ax