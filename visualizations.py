import matplotlib.pyplot as plt
import seaborn as sns
def plot_confusion_matrix(cm ,class_names, save_to_file = False):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')


    if save_to_file:
        plt.savefig('outputs/confusion_matrix.png')
    plt.show()