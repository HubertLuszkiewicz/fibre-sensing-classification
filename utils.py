from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

def show_histogram(data, title, xlabel, ylabel, bins=50):
    plt.figure(figsize=(10, 5))
    sns.histplot(data, bins=bins, kde=True, color='C0')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def combine_features_and_labels(features_df, labels_df):
    return pd.concat([features_df, labels_df], axis=1)


def print_classification_report(y_true, y_pred, class_mapping):
    print(
        classification_report(
            y_true, 
            y_pred, 
            target_names=[class_mapping[i] for i in range(len(class_mapping))]
        )
    )


def show_confusion_matrix(y_val, y_pred, class_mapping):
    cm = confusion_matrix(y_val, y_pred)

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=[class_mapping[i] for i in range(len(class_mapping))], 
        yticklabels=[class_mapping[i] for i in range(len(class_mapping))]
    )
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()