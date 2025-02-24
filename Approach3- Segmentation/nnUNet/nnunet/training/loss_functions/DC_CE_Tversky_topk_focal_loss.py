
import torch
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.training.loss_functions.focal_loss import FocalLoss
from nnunet.training.loss_functions.focal_loss import FocalLoss
from torch import nn
from nnunet.training.loss_functions.TverskyLoss import TverskyLoss
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss,SoftDiceLossSquared
import numpy as np

class DC_CE_Tversky_topk_focal_loss(nn.Module):

    def __init__(self, soft_dice_kwargs, topk_kwargs, focal_kwargs, ce_kwargs,tversky_kwargs,aggregate="sum",square_dice=False):
        super(DC_CE_Tversky_topk_focal_loss, self).__init__()
        self.aggregate = aggregate
        self.topk = TopKLoss(**topk_kwargs)
        self.focal = FocalLoss(apply_nonlin=softmax_helper, **focal_kwargs)
        self.ce= RobustCrossEntropyLoss(**ce_kwargs)
        self.tversky=TverskyLoss(**tversky_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        topk_loss = self.topk(net_output, target)
        focal_loss = self.focal(net_output, target)
        ce_loss = self.ce(net_output, target)
        tversky_loss = self.tversky(net_output, target)

        if self.aggregate == "sum":
            loss1 = dc_loss
            loss2 = ce_loss
            loss3 = topk_loss
            loss4 = tversky_loss
            loss5 = focal_loss
            loss6 = dc_loss + ce_loss
            loss7 = dc_loss + topk_loss
            loss8 = dc_loss + focal_loss
            loss9 = tversky_loss + ce_loss
            loss10 = tversky_loss + topk_loss
            loss11 = tversky_loss + focal_loss
            loss12 = dc_loss + ce_loss + focal_loss
            loss13 = dc_loss + ce_loss + topk_loss
            loss14 = dc_loss + tversky_loss + ce_loss
            loss15 = dc_loss + tversky_loss + topk_loss
            loss16 = dc_loss + tversky_loss + focal_loss
            # Generate a random integer  1=<a<=20
            a = np.random.randint(1, 17)
            # switch case on a

            if a == 1:
                result = loss1
                #print("dc_loss %f" % loss1)
            elif a == 2:
                result = loss2
                #print("topk_loss %f" % loss2)
            elif a == 3:
                result = loss3
                #print("focal_loss %f" % loss3)
            elif a == 4:
                result = loss4
                #print("ce_loss %f" % loss4)
            elif a == 5:
                result = loss5
                #print("tversky_loss %f" % loss5)
            elif a == 6:
                result = loss6
                #print("dc_loss + ce_loss %f" % loss6)
            elif a == 7:
                result = loss7
                #print("dc_loss + topk_loss %f" % loss7)
            elif a == 8:
                result = loss8
                #print("ce_loss + topk_loss %f" % loss8)
            elif a == 9:
                result = loss9
                #print("dc_loss + tversky_loss %f" % loss9)
            elif a == 10:
                result = loss10
            elif a == 11:
                result = loss11
            elif a == 12:
                result = loss12
            elif a == 13:
                result = loss13
            elif a == 14:
                result = loss14
            elif a == 15:
                result = loss15
            elif a == 16:
                result = loss16
            
                
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result


#loss1 = dc_loss     loss2 = topk_loss      loss3 = focal_loss         loss4 = ce_loss         loss5 = tversky_loss         loss6 = dc_loss + ce_loss        loss7 = dc_loss + topk_loss        loss8 = ce_loss + topk_loss       loss9 = dc_loss + tversky_loss          loss10 = ce_loss + tversky_loss


#result = 0.6 * dc_loss + 0.1 * topk_loss + 0.1 * focal_loss +  0.1 * ce_loss + 0.1 * tversky_loss
#result = 0.5 * dc_loss + 0.25 * topk_loss + 0.25 * ce_loss

##############################
#Weighted Dice Loss from paper
#DSC = 1 - dc_loss
#beta = 0.1
#term1 = -1 * DSC
#term2 =  pow((1-DSC),beta)
#result = term1 * term2
#############################

#Randomly Selection of Loss
