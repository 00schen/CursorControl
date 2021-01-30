import numpy as np
import torch as th

def mixup(feat,target,alpha=.5):
    lambd = np.random.beta(alpha, alpha, feat.size(0))
    lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
    lambd = feat.new(lambd)
    shuffle = th.randperm(feat.size(0)).to(feat.device)
    feat1,target1 = feat[shuffle], target[shuffle]

    mix_feat = (feat * lambd.view(lambd.size(0),1) + feat1 * (1-lambd).view(lambd.size(0),1))
    mix_target = (target * lambd.view(lambd.size(0),1) + target1 * (1-lambd).view(lambd.size(0),1))
    return mix_feat,mix_target