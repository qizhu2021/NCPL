import numpy as np
import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score


def get_test_data(loader, net):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features, outputs = net(inputs)
            if start_test:
                all_features = features.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_features, all_output, all_label, predict


def get_metrics_sev_class(logits, y_true, y_predict):
    print(y_true[0])
    print(y_predict[0])
    f1 = f1_score(y_true, y_predict, average='weighted')
    recall = recall_score(y_true, y_predict, average='weighted')
    precision = precision_score(y_true, y_predict, average='weighted')
    accuracy, kappa, report, sensitivity, specificity, roc_auc = get_metrics(logits, y_true, y_predict)
    return accuracy, kappa, report, sensitivity, specificity, roc_auc, f1, recall, precision


def get_metrics(logits, y_true, y_predict):
    class_num = len(np.unique(y_true))

    cm = confusion_matrix(y_true, y_predict)
    accuracy = torch.sum(torch.squeeze(y_predict).float() == y_true).item() / float(y_true.size()[0])
    kappa = cohen_kappa_score(y_true, y_predict, weights="quadratic")
    report = classification_report(y_true, y_predict, target_names=['Grade ' + str(i) for i in range(class_num)])
    # Binary classification
    sensitivity = specificity = roc_auc = 0
    if class_num == 2:
        sensitivity = cm[0, 0] / (cm[0, 0] + cm[1, 0])
        specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        roc_auc = roc_auc_score(y_true=y_true, y_score=torch.nn.Softmax(dim=1)(logits.cpu())[:, 1])

    return accuracy * 100, kappa, report, sensitivity, specificity, roc_auc
