import torch
import numpy as np
from torch import nn

class SelectionPolicy(nn.Module):
    """
    通用动态选择策略模块
    :param threshold: 误差阈值
    """

    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.use_pred_records = []  # 记录策略触发情况

    def forward(self, pred, true, model, input_seq, scaler, device):
        """
        :param pred: 预测值 (batch_size, output_dim)
        :param true: 真实值 (batch_size, features)
        :param model: 基础模型
        :param input_seq: 当前输入序列 (batch_size, seq_len, features)
        :param scaler: 标准化器
        :param device: 设备 (CPU/GPU)
        :return: 更新后的输入序列 (batch_size, seq_len, features)
        """
        # 转换为 NumPy 数组
        pred_np = pred.detach().cpu().numpy()
        true_np = true.detach().cpu().numpy()
        true_np = scaler.inverse_transform(true_np)  # 对真实值进行反标准化

        # 仅计算目标特征的误差
        target_error = np.abs(pred_np[:, -1] - true_np[:, -1])  # 最后一列是目标特征
        use_pred_mask = target_error < self.threshold

        # 记录策略触发情况
        self.use_pred_records.append(use_pred_mask)

        # 生成新的输入序列
        next_inputs = []
        for i in range(pred_np.shape[0]):
            if use_pred_mask[i]:
                # 复制真实值，仅替换目标特征为预测值
                modified_input = true_np[i].copy()
                modified_input[-1] = pred_np[i, -1]  # 最后一列是目标
            else:
                modified_input = true_np[i]  # 完全使用真实值
            next_inputs.append(modified_input)

        # 标准化后返回
        next_input_norm = scaler.transform(np.array(next_inputs))
        next_tensor = torch.FloatTensor(next_input_norm).to(device)
        return torch.cat([input_seq[:, 1:, :], next_tensor.unsqueeze(1)], dim=1)  # 保持 [batch, seq_len, features]


class modelPolicy(nn.Module):
    def __init__(self, base_model, threshold):

        super(modelPolicy, self).__init__()
        self.base_model =base_model
        self.policy =SelectionPolicy(threshold)  # 动态选择策略模块

    def forward(self, x, labels=None, scaler=None, device=None, select=False):
        if select:
            # 动态预测模式
            batch_size, seq_len, features = x.shape
            preds = []  # 存储预测结果
            current_seq = x.clone()  # 初始化输入序列

            for step in range(labels.shape[1]):  # 按时间步循环
                # 预测下一个时间点
                pred = self.base_model(current_seq)  # [batch_size, pre_len, output_dim]
                last_pred = pred[:, -1, :]  # 取最后一个预测点 [batch_size, output_dim]

                # 反标准化预测值（仅目标特征）
                pred_denorm = scaler.inverse_transform(
                    last_pred.detach().cpu().numpy()
                )[:, -1].reshape(-1, 1)  # 仅取目标列

                # 使用选择策略更新输入序列
                current_seq = self.policy(
                    torch.FloatTensor(pred_denorm).to(device),  # 转换为 torch.Tensor
                    labels[:, step, :],  # 当前时间步的完整真实值
                    self.base_model,
                    current_seq,
                    scaler,
                    device
                )
                preds.append(last_pred.detach().cpu().numpy())  # 收集预测结果

            # 将 preds 转换为 [batch_size, time_steps, 1]
            preds = np.stack(preds, axis=1)  # [batch_size, time_steps, 1]
            return torch.FloatTensor(preds).to(device)  # 转换为 torch.Tensor
        else:
            return self.base_model(x)

