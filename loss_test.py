import torch
import torch.nn as nn
from torch.nn.functional import logsigmoid

Sigmoid_fun = nn.Sigmoid()
BCELoss_fun = nn.BCELoss()
BCELossLog_fun = nn.BCEWithLogitsLoss()
#                       O    O   X    X   O   O
predit = torch.tensor([-3., -4., 5., -5., 4., 9.])
target = torch.tensor(([0.,  0., 0.,  1., 1., 1.]))

predit_sig = Sigmoid_fun(predit)
print(f"Sigmoid(Prediction) = {predit_sig}")
print(BCELoss_fun(predit_sig, target))
print(BCELossLog_fun(predit, target))

# GAC's implementation
bce = -(     target  * logsigmoid( predit)) +\
      -((1 - target) * logsigmoid(-predit))
print(bce.mean())

# My implementation 
my_bce = -(     target  * torch.log(  predit_sig)) +\
         -((1 - target) * torch.log(1-predit_sig))
print(my_bce.mean())


focal_weight = torch.where(torch.eq(target, 1.), 1. - predit_sig, predit_sig)
print(focal_weight)

# probs = torch.sigmoid(cls_pred) #[B, N, 1]
# focal_weight = torch.where(torch.eq(targets, 1.), 1. - probs, probs)
# focal_weight = torch.pow(focal_weight, gamma)

# # Binary Cross1
# bce = -(     targets  * logsigmoid( cls_pred)) * balance_weights +\
#         -((1 - targets) * logsigmoid(-cls_pred)) #[B, N, 1]
# cls_loss = focal_weight * bce