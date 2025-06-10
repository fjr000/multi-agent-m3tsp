import torch
import torch.nn as nn

from model.AttnModel.Model import Model as BaseModel
from model.AttnModel.Net import SingleHeadAttention


class Config(object):
    embed_dim = 128
    dropout = 0

    city_encoder_hidden_dim = 128
    city_encoder_num_layers = 3
    city_encoder_num_heads = 8

    action_decoder_hidden_dim = 128
    action_decoder_num_layers = 2
    action_decoder_num_heads = 8

    conflict_deal_hidden_dim = 128
    conflict_deal_num_layers = 1
    conflict_deal_num_heads = 8

    action_hidden_dim = 128
    action_num_layers = 1
    action_num_heads = 8



class ConflictDeal:
    def __call__(self, *args, **kwargs):
        agent_embed, city_embed, acts = args

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

        k = agent_embed
        q = cac
        score = torch.bmm(q, k.transpose(-2, -1).contiguous())
        score = score.masked_fill(conflict_matrix, -torch.inf)

        del conflict_matrix, identity_matrix, acts_exp1, acts_exp2
        return score


class Model(BaseModel):
    def __init__(self, config: Config):
        super(Model, self).__init__(Config)
        self.conflict_deal = ConflictDeal()


    def forward(self, agent, mask, info=None, eval=False):
        info.update({
            "dones":None
        })



        # if self.step == 0:
        #     self.actions_model.agent_decoder.init_rnn_state(agent.size(0), agent.size(1), agent.device)

        dones = None if info is None else info.get("dones", None)
        batch_mask = None if dones is None else ~dones
        city_mask = None if info is None else info.get("mask", None)
        info.update({
            "batch_mask": batch_mask
        })

        if batch_mask is not None:
            agent = agent[batch_mask]
            city_mask = city_mask[batch_mask] if city_mask is not None else city_mask
            mask = mask[batch_mask] if mask is not None else mask

        self.actions_model.update_city_mean(agent.size(1), city_mask, batch_mask)

        mode = "greedy" if info is None else info.get("mode", "greedy")
        actions_logits, _ = self.actions_model(agent, mask, info)
        acts = None
        if mode == "greedy":

            acts = actions_logits.argmax(dim=-1)
        elif mode == "sample":
            probs = torch.softmax(actions_logits.view(-1, actions_logits.size(-1)), dim=-1)
            acts = torch.multinomial(probs, num_samples=1).squeeze(-1)
            acts = acts.view(actions_logits.size(0), actions_logits.size(1))
            # acts = torch.distributions.Categorical(logits=actions_logits).sample()

        else:
            raise NotImplementedError

        use_conflict_model = False if info is None else info.get("use_conflict_model", False)
        acts_no_conflict = None
        if use_conflict_model:
            act2agent_logits = self.conflict_deal( self.actions_model.agent_decoder.action.glimpse_Q,self.actions_model.agent_decoder.action.glimpse_K.transpose(-2, -1).contiguous(), acts)
            agents = act2agent_logits.argmax(dim=-1)
            pos = torch.arange(actions_logits.size(1), device=agents.device).unsqueeze(0)
            masks = torch.logical_or(agents == pos, acts == 0)
            acts_no_conflict = torch.where(masks, acts, -1)
            del pos, agents, masks

        self.step += 1

        if batch_mask is not None:
            B = batch_mask.size(0)
            A = actions_logits.size(1)
            N = actions_logits.size(2)

            final_acts = torch.zeros((B, A), dtype=torch.int64, device=actions_logits.device)
            final_acts[batch_mask] = acts
            final_acts_no_conflict = torch.zeros((B, A), dtype=torch.int64, device=actions_logits.device)
            final_acts_no_conflict[batch_mask] = acts_no_conflict

            if eval:
                return None, final_acts, final_acts_no_conflict

            final_actions_logits = torch.full((B, A, N),
                                              fill_value=-torch.inf,
                                              dtype=torch.float32,
                                              device=actions_logits.device)
            final_actions_logits[:, :, 0] = 1.0  # 模拟选择仓库
            final_actions_logits[batch_mask] = actions_logits

            actions_logits = final_actions_logits
            acts = final_acts
        return actions_logits, acts, acts_no_conflict

