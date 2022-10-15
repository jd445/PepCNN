import sklearn
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from torch import nn
import pandas as pd


def writeToCSVBYPandas(fileName, system_table):
    # df = pd.concat([word_list, species_code_list], axis=1)
    df = pd.DataFrame(list(system_table.items()))
    labels = ['metrix', 'num']
    # DataFrame存储为csv格式文件，index表示是否显示行名,默认是
    df.to_csv(fileName, header=labels, sep=',', index=False, encoding="utf_8_sig")


def compute_metrics(model, model_name, save_path_acc, test_loader, plot_roc_curve=False):
    model_name = model_name + '.csv'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    val_loss = 0
    val_correct = 0
    criterion = nn.CrossEntropyLoss()
    score_list = torch.Tensor([]).to(device)
    pred_list = torch.Tensor([]).to(device).long()
    target_list = torch.Tensor([]).to(device).long()

    for batch in test_loader:
        # A batch consists of image data and corresponding labels.
        imgs, target = batch
        # imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            output = model(imgs.to(device))

        # Log loss
        val_loss += criterion(output.to(device), target.to(device)).item()

        # Calculate the number of correctly classified examples
        pred = output.argmax(dim=1, keepdim=True)
        val_correct += pred.eq(target.long().view_as(pred).to(device)
                               ).sum().item()

        # Bookkeeping
        score_list = torch.cat(
            [score_list, nn.Softmax(dim=1)(output)[:, 1].squeeze()])
        pred_list = torch.cat([pred_list, pred.squeeze()])
        target_list = torch.cat(
            [target_list.to(device), target.squeeze().to(device)])

    classification_metrics = classification_report(
        target_list.tolist(), pred_list.tolist(), output_dict=True)

    # sensitivity is the recall of the positive class
    sensitivity = classification_metrics['0']['recall']

    # specificity is the recall of the negative class
    specificity = classification_metrics['1']['recall']

    # accuracy
    accuracy = classification_metrics['accuracy']

    # confusion matrix
    conf_matrix = confusion_matrix(target_list.tolist(), pred_list.tolist())

    # roc score
    roc_score = roc_auc_score(target_list.tolist(), score_list.tolist())

    # plot the roc curve
    if plot_roc_curve:
        fpr, tpr, _ = roc_curve(target_list.tolist(), score_list.tolist())
        plt.plot(fpr, tpr, label="Area under ROC = {:.4f}".format(roc_score))
        plt.legend(loc='best')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()
    label = target_list.tolist()
    score = score_list.tolist()
    pred = pred_list.tolist()
    all = [label, score, pred]

    all = list(map(list, zip(*all)))

    columns = ['target_list', 'score_list', 'pred_list']
    df = pd.DataFrame(columns=columns, data=all)
    df.to_csv(model_name, encoding='utf-8')
    # put together values
    metrics_dict = {"Accuracy": accuracy,
                    "Sensitivity": sensitivity,
                    "Specificity": specificity,
                    "Roc_score": roc_score,
                    "Confusion Matrix": conf_matrix,
                    "Validation Loss": val_loss / len(test_loader),
                    'MCC : ': sklearn.metrics.matthews_corrcoef(pred_list.cpu(), target_list.cpu())
                    }
    writeToCSVBYPandas(save_path_acc+'.txt', metrics_dict)

    print('MCC : ', sklearn.metrics.matthews_corrcoef(pred_list.cpu(), target_list.cpu()))
    print('specificity : ', specificity)
    print('sensitivity : ', sensitivity)
    print('accuracy : ', accuracy)

    return metrics_dict
