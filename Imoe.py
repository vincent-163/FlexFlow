import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTMoE(nn.Module):
    def __init__(self, num_experts, hidden_size, num_classes):
        super(GPTMoE, self).__init__()

        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # GPT Encoder层
        self.encoder_layers = nn.ModuleList([GPTEncoder(hidden_size) for _ in range(12)])

        # Mixture of Experts层
        self.expert_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_experts)])
        self.gate_layer = nn.Linear(hidden_size, num_experts)

        # 输出层
        self.output_layer = nn.Linear(hidden_size, num_classes)

        # 初始化KV Cache
        self.key_cache = None
        self.value_cache = None

    def forward(self, x):
        # GPT Encoder层
        encoder_output = x
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output)

        # Mixture of Experts
        expert_outputs = [expert_layer(encoder_output) for expert_layer in self.expert_layers]
        gate_scores = self.gate_layer(encoder_output)
        gate_probs = torch.softmax(gate_scores, dim=-1)

        # 加权平均输出
        moe_output = sum(expert_output * gate_probs[:, :, i].unsqueeze(-1) for i, expert_output in enumerate(expert_outputs))

        # 输出层
        output = self.output_layer(moe_output)

        return output


class GPTEncoder(nn.Module):
    def __init__(self, hidden_size):
        super(GPTEncoder, self).__init__()

        self.hidden_size = hidden_size

        # 自注意力机制层
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads=4)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        # 前馈神经网络层
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.ReLU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        # 自注意力机制
        attended, _ = self.self_attention(x, x, x)

        # 残差连接和层归一化
        attended = self.layer_norm1(attended + x)

        # 前馈神经网络
        ff_output = self.feed_forward(attended)

        # 残差连接和层归一化
       
