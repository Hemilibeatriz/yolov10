import dill
import torch
import numpy as np
from numpy._core.multiarray import scalar
from ultralytics import YOLOv10, settings
from ultralytics.nn.modules.head import v10Detect
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.nn.tasks import YOLOv10DetectionModel
from ultralytics.utils.loss import v10DetectLoss, v8DetectionLoss, BboxLoss
from ultralytics.nn.modules.block import C2f, Bottleneck, SCDown, SPPF, PSA, Attention, C2fCIB, CIB, RepVGGDW, DFL
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.activation import SiLU
from torch.nn.modules.linear import Identity
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.nn.modules.container import Sequential, ModuleList


# Permitir deserialização segura de objetos usados no modelo
torch.serialization.add_safe_globals([
    dill._dill._load_type,
    np.dtype, np.float64, np.dtypes.Float64DType,
    scalar,
    v10Detect, 
    Conv, Concat,
    IterableSimpleNamespace,
    TaskAlignedAssigner,
    YOLOv10DetectionModel,
    v10DetectLoss, v8DetectionLoss, BboxLoss,
    C2f, Bottleneck, SCDown, SPPF, PSA, Attention, C2fCIB, CIB, RepVGGDW, DFL,
    Conv2d, 
    SiLU,
    Identity,
    MaxPool2d,
    Upsample,
    BatchNorm2d,
    BCEWithLogitsLoss,
    Sequential, ModuleList,
])
