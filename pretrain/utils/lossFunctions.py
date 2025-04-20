from torch.nn.functional import sigmoid
import torch.nn as nn
import torch



def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    # label2 = label.float()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    prediction = sigmoid(prediction)
    # print(label.shape)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(),weight = mask, reduce=False)
    # weight = mask
    # return torch.sum(cost)
    # loss = F.binary_cross_entropy(prediction, label2, mask, size_average=True)
    return torch.sum(cost) /(num_positive+num_negative)


def BinaryFocalLoss(inputs, targets):
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    BCE_loss = criterion(inputs, targets)
    pt = torch.exp(-BCE_loss)
    F_loss = (1-pt)**2 * BCE_loss
    return F_loss.mean()
