import torch
import torch.nn as nn
from model.Common.MHA import MultiHeadAttentionLayer, SingleHeadAttention
from model.Common.config import ConflictModelConfig

class ConflictAttentionModel(nn.Module):
    def __init__(self, config: ConflictModelConfig):
        super(ConflictAttentionModel, self).__init__()
        # 不能使用dropout
        self.city_agent_att = nn.ModuleList([
            MultiHeadAttentionLayer(config.embed_dim,
                                    config.embed_dim,
                                    config.conflict_deal_hidden_dim,
                                    config.conflict_deal_num_heads,
                                    'layer')
            for _ in range(config.conflict_deal_num_layers)
        ])
        self.agents = SingleHeadAttention(config.embed_dim)

    def forward(self, agent_embed, city_embed, acts):
        """
        Args:
            agent_embed:   [B,A,E] 智能体特征
            city_embed:    [B,N,E] 城市特征
            acts: [B,A] 动作
        Returns:
            final_cities: [B,A] 最终分配结果
            conflict_mask: [B,N] 初始冲突标记
        """
        B, A, E = agent_embed.shape

        # 2. 生成初始冲突掩码 ----------------------------------------------------
        # 扩展维度用于广播比较
        acts_exp1 = acts.unsqueeze(2)  # [B,5,1]
        acts_exp2 = acts.unsqueeze(1)  # [B,1,5]
        # 生成布尔型冲突矩阵
        conflict_matrix = (acts_exp1 == acts_exp2).bool()  # [B,5,5]
        identity_matrix = torch.eye(A, device=acts.device).unsqueeze(0).bool()  # [1, A, A]
        conflict_matrix = torch.where(acts_exp1 == 0, identity_matrix, conflict_matrix)
        conflict_matrix = ~conflict_matrix
        # expand_conflict_mask = conflict_matrix.unsqueeze(1).expand(B, self.num_heads, A, A).reshape(B*self.num_heads, A, A)

        # 3. 提取候选城市特征 -----------------------------------------------------
        selected_cities = torch.gather(
            city_embed,
            1,
            acts.unsqueeze(-1).expand(-1, -1, E)
        )  # [B,5,E]

        # 4. 注意力重新分配 ------------------------------------------------------
        # Q: 候选城市特征 [B,5,E]
        # K/V: 智能体特征 [B,5,E]
        cac = selected_cities
        for att in self.city_agent_att:
            cac = att(cac, agent_embed, conflict_matrix)

        agents_logits = self.agents(cac, agent_embed, conflict_matrix)

        del conflict_matrix, identity_matrix, acts_exp1, acts_exp2

        return agents_logits
