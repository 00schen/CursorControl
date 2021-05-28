"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
from rlkit.torch.networks.basic import (
    Clamp, ConcatTuple, Detach, Flatten, FlattenEach, Split, Reshape,
)
from rlkit.torch.networks.cnn import BasicCNN, CNN, MergedCNN, CNNPolicy
from rlkit.torch.networks.dcnn import DCNN, TwoHeadDCNN
from rlkit.torch.networks.feat_point_mlp import FeatPointMlp
from rlkit.torch.networks.image_state import ImageStatePolicy, ImageStateQ
from rlkit.torch.networks.linear_transform import LinearTransform
from rlkit.torch.networks.normalization import LayerNorm
from rlkit.torch.networks.mlp import (
    Mlp, ConcatMlp, ConcatMlpPolicy, MlpPolicy, TanhMlpPolicy, MlpGazePolicy, VAE,
    MlpQf,
    MlpQfWithObsProcessor,
    ConcatMultiHeadedMlp,
    QrMlp,
    QrGazeMlp,
)
from rlkit.torch.networks.rnn import (ConcatRNN,ConcatRNNPolicy)
from rlkit.torch.networks.pretrained_cnn import PretrainedCNN
from rlkit.torch.networks.two_headed_mlp import TwoHeadMlp
from rlkit.torch.networks.encoder_policies import VQGazePolicy, VAEGazePolicy, TransferEncoderPolicy

__all__ = [
    'Clamp',
    'ConcatMlp',
    'ConcatMultiHeadedMlp',
    'ConcatTuple',
    'BasicCNN',
    'CNN',
    'CNNPolicy',
    'DCNN',
    'Detach',
    'FeatPointMlp',
    'Flatten',
    'FlattenEach',
    'LayerNorm',
    'LinearTransform',
    'ImageStatePolicy',
    'ImageStateQ',
    'MergedCNN',
    'Mlp',
    'PretrainedCNN',
    'Reshape',
    'Split',
    'TwoHeadDCNN',
    'TwoHeadMlp',
    'QrMlp',
    'QrGazeMlp',
    'VQGazePolicy',
    'VAEGazePolicy',
    'VAE'
]

