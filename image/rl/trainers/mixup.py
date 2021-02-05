import numpy as np
import torch as th

def mixup(feats,target,alpha=5):
    lambd = np.random.beta(alpha, alpha, target.size(0))
    lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
    lambd = target.new(lambd)
    shuffle = th.randperm(target.size(0)).to(target.device)

    target1 = target[shuffle]
    mix_target = (target * lambd.view(lambd.size(0),1) + target1 * (1-lambd).view(lambd.size(0),1))
    
    mix_feats = []
    for feat in feats:
        feat1 = feat[shuffle]
        mix_feat = (feat * lambd.view(lambd.size(0),1) + feat1 * (1-lambd).view(lambd.size(0),1))
        mix_feats.append(mix_feat)
    return mix_feats,mix_target