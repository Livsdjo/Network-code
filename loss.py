import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

margin = 1
def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None,
                                     reduce=False, reduction='elementwise_mean', pos_weight=None):

    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)

    if pos_weight is None:
        ce_loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    else:
        log_weight = 1 + (pos_weight - 1) * target
        ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp()).log())
        # print(1010101)
        # print(ce_loss.shape)
    
    #    if math.isnan(ce_loss.mean()):
    #       ce_loss = input - input * target + log_weight * (max_val + ((-max_val).exp() + (-input - max_val).exp() + 0.1).log())
    #    print(input.grad)
    #    return torch.Tensor([0]) 
            
    if weight is not None:
        ce_loss = ce_loss * weight

    # print(weight)

    if reduction == False:
        return ce_loss
    elif reduction == 'elementwise_mean':
        return ce_loss.mean()
    else:
        return ce_loss.sum()

class Loss_classi(nn.Module):
    def __init__(self):
        super(Loss_classi, self).__init__()

    def loss_classi(self, output, label):
        # print("26183683")
        # print(output.shape, label.shape)
        #print(label)
        pos = torch.sum(label)
        pos_num = F.relu(pos - 1) + 1
        total = torch.numel(label)
        neg_num = F.relu(total - pos - 1) + 1
        pos_w = neg_num // pos_num
        # print("ahdashdaui", pos_w)
        pos_w = None
        classi_loss = binary_cross_entropy_with_logits(output, label, pos_weight=pos_w, reduce=True)
        
        return classi_loss

    def forward(self, output, label, loss_type="1"):
        # ############
        # loss type:
        # 方式 1 : NM-net原文中的loss 有点奇葩
        #
        # 方式 2 ：正统的分类均衡loss
        #
        # ############
        if loss_type == "1":
            loss = self.loss_classi(output, label)
        """
        elif loss_type == "2":  
            # OAnet代码 不要动了  改起来太麻烦了
            
            is_pos = label
            is_neg = (label == 0.).type(label.type()).unsqueeze(0)
            c = is_pos - is_neg

            loss = torch.tensor(0.0).cuda()
            for value in output:
                if 0:
                    logits = output
                else:
                    logits = value

                classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())

                classif_losses = classif_losses.squeeze(0)
                is_neg = is_neg.squeeze(0)
                num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
                if is_neg.dim() == 1:
                    is_neg = is_neg.unsqueeze(0)
                # print(222222, is_neg.shape)
                num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
                classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
                classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)

                classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)

                loss += classif_loss
        elif loss_type == "3":
            
            #  这个版本是用于NM-net Cne 多模态网络 参数化网络等
            
            is_pos = label
            is_neg = (label == 0.).type(label.type()).unsqueeze(0)
            c = is_pos - is_neg
            logits = output

            classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())

            classif_losses = classif_losses.squeeze(0)
            is_neg = is_neg.squeeze(0)
            num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
            if is_neg.dim() == 1:
                is_neg = is_neg.unsqueeze(0)
            # print(222222, is_neg.shape)
            num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
            classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
            classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)

            classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)

            loss = classif_loss
        else:
            print("Error >>>>>>>>")
        """

        return loss



    """
    gt_geod_d = y_in[:, :, 0]
    is_pos = (gt_geod_d < self.obj_geod_th).type(logits.type())
    is_neg = (gt_geod_d >= self.obj_geod_th).type(logits.type())
    c = is_pos - is_neg
    classif_losses = -torch.log(torch.sigmoid(c * logits) + np.finfo(float).eps.item())
    # balance
    num_pos = torch.relu(torch.sum(is_pos, dim=1) - 1.0) + 1.0
    num_neg = torch.relu(torch.sum(is_neg, dim=1) - 1.0) + 1.0
    classif_loss_p = torch.sum(classif_losses * is_pos, dim=1)
    classif_loss_n = torch.sum(classif_losses * is_neg, dim=1)
    classif_loss = torch.mean(classif_loss_p * 0.5 / num_pos + classif_loss_n * 0.5 / num_neg)


    precision = torch.mean(
        torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
        torch.sum((logits > 0).type(is_pos.type()) * (is_pos + is_neg), dim=1)
    )
    recall = torch.mean(
        torch.sum((logits > 0).type(is_pos.type()) * is_pos, dim=1) /
        torch.sum(is_pos, dim=1)
    )
    """


