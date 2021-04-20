from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable

def get_optim_and_scheduler(network, epochs, lr, train_all, nesterov=False):
    if train_all:
        params = network.parameters()
    else:
        params = network.get_params(lr)
    optimizer = optim.SGD(params, weight_decay=.0005, momentum=.9, nesterov=nesterov, lr=lr)
    #optimizer = optim.Adam(params, lr=lr)
    step_size = int(epochs * .8)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
    print("Step size: %d" % step_size)
    return optimizer, scheduler

class FocalLoss(nn.Module):
    '''
    Args:
        alpha(1D Tensor, Variable) : the scalar factor for this criterion
        gamma(float, double) : gamma>0; reduces the relatvie loss for well-classified examples(p>.5),
                            putting more focus on hard, misclassiﬁed examples
        size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    '''
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num,1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
    
    def forward(self, inputs, targets):
        N,C = inputs.shape[0], inputs.shape[1]
        P = F.softmax(inputs, dim=1)
        class_mask = inputs.data.new(N,C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1,1)
        class_mask.scatter_(1,ids,1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1).clamp(1e-32)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average: 
            loss = batch_loss.mean()
        else: 
            loss = batch_loss.sum()

        return loss
