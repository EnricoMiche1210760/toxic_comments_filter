import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def show_number(plot, barplot):
    for i in range(len(barplot)):
        height = barplot[i].get_height()
        plot.text(barplot[i].get_x() + barplot[i].get_width()/2, height, height, ha='center', va='bottom')

def plot_bar(columns, rows, figsize=(8,5), title:str=None, xlabel:str=None, ylabel:str=None, color='blue', xticks_settings=None, yticks_settings=None):
    plt.figure(figsize=figsize)
    barplot = plt.bar(columns, rows, color=color)
    plt.yticks(**yticks_settings)
    plt.xticks(**xticks_settings)
    if title is not None:
        plt.title(title, fontsize=15)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=10)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=10)
    show_number(plt, barplot)
    plt.show();


def plot_stats(dataframe, categories, data):
    axes = []
    fig, axes = plt.subplots(2, 3, figsize=(15,10))

    len_sentences = np.array([len(str(sentence)) for sentence in data])

    for i, col in enumerate(categories):
        len_sentences_col = len_sentences[dataframe[dataframe[col] == 1].index]
        ax = axes[i//3, i%3]
        ax.hist(len_sentences_col, bins=50, color='skyblue')
        ax.set_title(col)
        ax.text(.99, 1, f"Mean: {round(len_sentences_col.mean(), 2)}", transform=ax.transAxes, ha='right', va='top')
        ax.text(.99, .96, f"Median: {round(np.median(len_sentences_col), 2)}", transform=ax.transAxes, ha='right', va='top')
        ax.text(.99, .92, f"Std-deviation: {round(len_sentences_col.std(), 2)}", transform=ax.transAxes, ha='right', va='top')
        ax.text(.99, .88, f"95-percentile: {round(np.percentile(len_sentences_col, 95), 2)}", transform=ax.transAxes, ha='right', va='top', color='red')
        ax.axvline(x=np.percentile(len_sentences_col, 95), color='red', linewidth=.95)

    fig.tight_layout()
    plt.show();

def print_confusion_matrix(y_test, y_pred, columns):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    print(columns)
    for i, col in enumerate(columns):
        ax = axes[i//3, i%3]
        sns.heatmap(confusion_matrix(y_test.T[i], y_pred.T[i]), annot=True, fmt='d', cmap="Greens", ax=ax)
        ax.set_title(f"Confusion matrix for {col}")
        ax.set_xlabel("Predicted severity")
        ax.set_ylabel("Severity")
        ax.xaxis.set_ticklabels([f'Pred non {col}', f'Pred {col}'])
        ax.yaxis.set_ticklabels([f'Non {col}', f'{col}'])

    plt.subplots_adjust(hspace=0.3)
    plt.show();

def plot_training_stats(train_loss, val_loss, train_acc, val_acc):
    fig, axes = plt.subplots(1, 2, figsize=(15,10))
    axes[0].plot(train_loss, label='train')
    axes[0].plot(val_loss, label='validation')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(train_acc, label='train')
    axes[1].plot(val_acc, label='validation')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    plt.show();