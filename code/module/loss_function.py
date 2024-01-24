import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

import torch.nn.functional as F



def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)

    # --------------------- dangwei li TIP20 ---------------------
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)


    # --------------------- AAAI ---------------------
    # pos_weights = torch.sqrt(1 / (2 * ratio.sqrt())) * targets
    # neg_weights = torch.sqrt(1 / (2 * (1 - ratio.sqrt()))) * (1 - targets)
    # weights = pos_weights + neg_weights

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights




class adjLoss(nn.Module):

    def __init__(self, num_classes):
        super(adjLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, pre_adj, label):
        target_adj = Variable(self.get_adj(label))
        result = pre_adj - target_adj
        result = result.abs()
        flag = (result < 1).float()
        result = flag * result ** 2 + (1 - flag) * (result - 0.5)
        return result.mean()

    def get_adj(self, label):
        adj = torch.zeros((self.num_classes, self.num_classes)).cuda()
        for item in label:
            for i in range(self.num_classes):
                if item[i] == 1:
                    for j in range(i + 1, self.num_classes):
                        if item[j] == 1:
                            adj[i, j] += 1
                            adj[j, i] += 1
        count = label.sum(dim=0).reshape((self.num_classes, 1))
        adj = adj / (count + 1e-7)
        adj = adj + torch.eye(self.num_classes).cuda()
        return adj


class WeightLoss(nn.Module):
    def __init__(self, weight):
        nn.Module.__init__(self)
        self.weight = weight

    def forward(self, pre, label):
        epsilon = 1e-7
        weight = self.weight
        output = torch.sigmoid(pre)
        weighted_b_ce = - 1 / (2 * weight) * label * torch.log(output + epsilon) - (1.0 - label) / (
                2 * (1 - weight)) * torch.log(1.0 - output + epsilon)
        loss = torch.mean(weighted_b_ce)
        return loss


class WeightMSELoss(nn.Module):
    def __init__(self, weight):
        nn.Module.__init__(self)
        self.weight = weight

    def forward(self, pre, label):
        epsilon = 1e-7
        weight = self.weight
        output = torch.sigmoid(pre)
        weighted_b_ce = - 1 / (2 * weight) * torch.sqrt((output - label)**2) - 1 / (
                2 * (1 - weight)) * torch.sqrt((output - (1 - label))**2)
        loss = torch.mean(weighted_b_ce)
        return loss



class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss

class smooth_BCELoss(nn.Module):

    def __init__(self, sample_weight=None, size_sum=True, scale=None, tb_writer=None):
        super(smooth_BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = None
        self.loss_fn1 = torch.nn.MSELoss(reduction='none')

    def forward(self, logits, targets):
        #print(targets)
        #logits = logits[0]

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (1 - targets)

        loss_m = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        #loss_m = self.loss_fn1(logits.float(), targets.float())

        targets_mask = torch.where(targets.detach().cpu() > 0.5, torch.ones(1), torch.zeros(1))
        if self.sample_weight is not None:
            sample_weight = ratio2weight(targets_mask, self.sample_weight)

            loss_m = (loss_m * sample_weight.cuda())

        # losses = loss_m.sum(1).mean() if self.size_sum else loss_m.mean()
        #loss_m = loss_m * mask.cuda()
        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return loss


if __name__ == '__main__':
    a = np.random.rand(5, 5)
    b = np.random.rand(16, 5)
    criteration = adjLoss(5)
    criteration.cuda()
    a = torch.tensor(a)
    a.cuda()
    s = criteration(a, b)
    print(s)