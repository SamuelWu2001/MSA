import numpy as np
import torch
from monai.metrics import DiceMetric
from monai.transforms import Activations
from monai.losses import DiceLoss

# 示例的预测数据和标签数据
pred_data = np.array([[
    [[0, 1, 0, 1, 0],
     [1, 1, 1, 1, 1],
     [0, 1, 0, 1, 0],
     [1, 0, 1, 0, 1],
     [0, 1, 0, 1, 0]]
], [
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]
]])

label_data = np.array([[
    [[0, 1, 0, 1, 0],
     [1, 1, 0, 1, 1],
     [0, 0, 1, 1, 0],
     [1, 1, 0, 1, 1],
     [0, 1, 0, 1, 0]]
], [
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]]
]])

# 转换为 PyTorch 的 Tensor
pred_tensor = torch.tensor(pred_data, dtype=torch.float32)
label_tensor = torch.tensor(label_data, dtype=torch.float32)

# 初始化 DiceMetric
dice_metric = DiceMetric(include_background=True, reduction="mean")

# 计算 Dice 分数
activation = Activations(sigmoid=True)(pred_tensor)  # 应用 sigmoid 激活
dice_score = dice_metric(pred_tensor, label_tensor)

print("Dice Score:", dice_score)
dice_loss = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
loss = dice_loss(pred_tensor, label_tensor)

print("Dice Loss:", loss.item())
