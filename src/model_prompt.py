import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import RGCNConv
from utils import SelfAttentionLayer,SelfAttentionCF
import random
import numpy as np

class KGPrompt(nn.Module):
    def __init__(
        self, hidden_size, token_hidden_size, n_head, n_layer, n_block,
        n_entity, num_relations, num_bases, edge_index, edge_type,
        entity_kg_n_entity,entity_kg_num_relations,entity_kg_edge_index, entity_kg_edge_type,
        n_prefix_rec=None, n_prefix_conv=None
    ):
        super(KGPrompt, self).__init__()
        self.hidden_size = hidden_size
        self.n_head = n_head
        self.head_dim = hidden_size // n_head
        self.n_layer = n_layer
        self.n_block = n_block
        self.n_prefix_rec = n_prefix_rec
        self.n_prefix_conv = n_prefix_conv

        entity_hidden_size = hidden_size // 2
        user_hidden_size = hidden_size // 2
        # RGCN提取知识图谱
        self.kg_encoder = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=num_relations,
                                   num_bases=num_bases)

        self.kg_encoder_EntityKG = RGCNConv(entity_hidden_size, entity_hidden_size, num_relations=entity_kg_num_relations,
                            num_bases=num_bases)
        # empty开辟一块内存[n*entity,entity_hidden_size]，nn.parameter转化为可训练的tensor
        # node_embeds size(entity在kg中的数量31162*382)
        self.node_embeds = nn.Parameter(torch.empty(n_entity, entity_hidden_size))

        # 标准差
        stdv = math.sqrt(6.0 / (self.node_embeds.size(-2) + self.node_embeds.size(-1)))
        self.node_embeds.data.uniform_(-stdv, stdv)
        self.edge_index = nn.Parameter(edge_index, requires_grad=False)
        self.edge_type = nn.Parameter(edge_type, requires_grad=False)

        self.entity_kg_node_embeds = nn.Parameter(torch.empty(entity_kg_n_entity, entity_hidden_size))        

        entity_kg_stdv = math.sqrt(6.0 / (self.entity_kg_node_embeds.size(-2) + self.entity_kg_node_embeds.size(-1)))
        self.entity_kg_node_embeds.data.uniform_(-entity_kg_stdv, entity_kg_stdv)
        self.entity_kg_edge_index = nn.Parameter(entity_kg_edge_index, requires_grad=False)
        self.entity_kg_edge_type = nn.Parameter(entity_kg_edge_type, requires_grad=False)

        self.entity_proj1 = nn.Sequential(
            
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )

        self.entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.entity_kg_entity_proj1 = nn.Sequential(
            
            nn.Linear(entity_hidden_size, entity_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(entity_hidden_size // 2, entity_hidden_size),
        )

        self.entity_kg_entity_proj2 = nn.Linear(entity_hidden_size, hidden_size)

        self.token_proj1 = nn.Sequential(
            nn.Linear(token_hidden_size, token_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(token_hidden_size // 2, token_hidden_size),
        )
        self.token_proj2 = nn.Linear(token_hidden_size, hidden_size)

        self.user_proj1 = nn.Sequential(
            nn.Linear(user_hidden_size, user_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(user_hidden_size // 2, user_hidden_size),
        )
        self.user_proj2 = nn.Linear(user_hidden_size, hidden_size)
        

        self.self_attn_db = SelfAttentionLayer(hidden_size,hidden_size)
        self.self_attn_cf = SelfAttentionCF(hidden_size)
        # self.W_uv = nn.Parameter(torch.rand(1).cuda()) 
        # self.W_uv_conv = nn.Parameter(torch.rand(1).cuda())
        # self.W_uv_rec = nn.Parameter(torch.rand(1).cuda())

        self.cross_attn = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prompt_proj1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.prompt_proj2 = nn.Linear(hidden_size, n_layer * n_block * hidden_size)

        if self.n_prefix_rec is not None:
            self.rec_prefix_embeds = nn.Parameter(torch.empty(n_prefix_rec, hidden_size))
            nn.init.normal_(self.rec_prefix_embeds)
            self.rec_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        if self.n_prefix_conv is not None:
            self.conv_prefix_embeds = nn.Parameter(torch.empty(n_prefix_conv, hidden_size))
            nn.init.normal_(self.conv_prefix_embeds)
            self.conv_prefix_proj = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, hidden_size)
            )
        
        self.alpha = nn.Parameter(torch.rand(1).cuda())

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def set_and_fix_node_embed(self, node_embeds: torch.Tensor):
        self.node_embeds.data = node_embeds
        self.node_embeds.requires_grad_(False)

    def get_entity_embeds(self):
        #  R-GCN提取实体表示
        node_embeds = self.node_embeds
        entity_embeds = self.kg_encoder(node_embeds, self.edge_index, self.edge_type) + node_embeds
        # print("def 1 entity size:"+str(entity_embeds.size()))
        entity_embeds = self.entity_proj1(entity_embeds) + entity_embeds
        # print("def 2 entity size:"+str(entity_embeds.size()))
        entity_embeds = self.entity_proj2(entity_embeds)
        # print("def 3 entity size:"+str(entity_embeds.size()))
        # hidden_size
        return entity_embeds
    
    def get_entity_kg_entity_embeds(self):
        #  R-GCN提取实体表示
        entity_kg_node_embeds = self.entity_kg_node_embeds

        entity_kg_entity_embeds = self.kg_encoder_EntityKG(entity_kg_node_embeds, self.entity_kg_edge_index, self.entity_kg_edge_type) + entity_kg_node_embeds
        
        entity_kg_entity_embeds = self.entity_kg_entity_proj1(entity_kg_entity_embeds) + entity_kg_entity_embeds
        
        entity_kg_entity_embeds = self.entity_kg_entity_proj2(entity_kg_entity_embeds)
        
        return entity_kg_entity_embeds
     
    # 定义计算过程
    def forward(self, entity_ids=None, token_embeds=None,user_embeds=None, output_entity=False, use_rec_prefix=False,
                use_conv_prefix=False):
        batch_size, entity_embeds, entity_len, token_len = None, None, None, None

        if entity_ids is not None:
            batch_size, entity_len = entity_ids.shape[:2]
            entity_embeds = self.get_entity_embeds()
            entity_embeds = entity_embeds[entity_ids]  # (batch_size, entity_len, hidden_size)

            entity_kg_entity_embeds = self.get_entity_kg_entity_embeds() # 先计算entity_kg的所有项目表示，包含了前驱、后继、相关项目
            sub_entity_kg_entity_embeds = torch.zeros(batch_size, entity_len,self.hidden_size,device=self.device)

            for i in range(batch_size):
                for j in range(entity_len):
                    entity_id = entity_ids[i, j]
                    if entity_id < entity_kg_entity_embeds.size(0):
                        sub_entity_kg_entity_embeds[i, j, :] = entity_kg_entity_embeds[entity_id]
            
            # 然后融入
            entity_embeds = entity_embeds + self.alpha * sub_entity_kg_entity_embeds

        
        if token_embeds is not None:
            batch_size, token_len = token_embeds.shape[:2]
            token_embeds = self.token_proj1(token_embeds) + token_embeds  # (batch_size, token_len, hidden_size)
            token_embeds = self.token_proj2(token_embeds)

        if entity_embeds is not None and token_embeds is not None :
            attn_weights = self.cross_attn(token_embeds) @ entity_embeds.permute(0, 2, 1)  # (batch_size, token_len, entity_len)
            attn_weights /= self.hidden_size
            if output_entity:
                uv = torch.cat((entity_embeds,token_embeds),dim=1) # (bs,entity_len+token_len,hs)
                self.W_uv_rec = nn.Parameter(torch.randn(batch_size,entity_len,entity_len+token_len).cuda()) 
                stdv = math.sqrt(6.0 / (self.W_uv_rec.size(-2) + self.W_uv_rec.size(-1)))
                self.W_uv_rec.data.uniform_(-stdv, stdv)

                weight_mat = self.W_uv_rec @ uv

                token_weights = F.softmax(attn_weights, dim=1).permute(0, 2, 1)
                prompt_embeds = token_weights @ token_embeds  + entity_embeds + weight_mat
                prompt_len = entity_len

            else:
                uv = torch.cat((entity_embeds,token_embeds),dim=1) 
                self.W_uv_conv = nn.Parameter(torch.randn(batch_size,token_len,entity_len+token_len).cuda()) 
                stdv = math.sqrt(6.0 / (self.W_uv_conv.size(-2) + self.W_uv_conv.size(-1)))
                self.W_uv_conv.data.uniform_(-stdv, stdv)

                weight_mat = self.W_uv_conv @ uv
                
                entity_weights = F.softmax(attn_weights, dim=2)
                prompt_embeds = entity_weights @ entity_embeds + token_embeds + weight_mat
                prompt_len = token_len


        elif entity_embeds is not None and token_embeds is None:
            prompt_embeds = entity_embeds
            prompt_len = entity_len
        else:
            prompt_embeds = token_embeds
            prompt_len = token_len

        # recommendation task-specifc prompts
        if self.n_prefix_rec is not None and use_rec_prefix:
            prefix_embeds = self.rec_prefix_proj(self.rec_prefix_embeds) + self.rec_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)

            if user_embeds is not None:
                prompt_embeds = torch.cat([user_embeds,prompt_embeds],dim=1)
                prompt_len += self.n_prefix_rec+user_embeds.shape[1] 
            else:
                prompt_len += self.n_prefix_rec

        # conversation task-specifc prompts
        if self.n_prefix_conv is not None and use_conv_prefix:
            prefix_embeds = self.conv_prefix_proj(self.conv_prefix_embeds) + self.conv_prefix_embeds
            prefix_embeds = prefix_embeds.expand(prompt_embeds.shape[0], -1, -1)
            prompt_embeds = torch.cat([prefix_embeds, prompt_embeds], dim=1)
            prompt_len += self.n_prefix_conv

        prompt_embeds = self.prompt_proj1(prompt_embeds) + prompt_embeds
        prompt_embeds = self.prompt_proj2(prompt_embeds)
        prompt_embeds = prompt_embeds.reshape(
            batch_size, prompt_len, self.n_layer, self.n_block, self.n_head, self.head_dim
        ).permute(2, 3, 0, 4, 1, 5)  # (n_layer, n_block, batch_size, n_head, prompt_len, head_dim)
    
        return prompt_embeds

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, 'model.pt')
        torch.save(state_dict, save_path)

    def save_epoch(self, save_dir,bestepoch):
        os.makedirs(save_dir, exist_ok=True)
        state_dict = {k: v for k, v in self.state_dict().items() if 'edge' not in k}
        save_path = os.path.join(save_dir, f'model-{bestepoch}.pt')
        torch.save(state_dict, save_path)

    def load(self, load_dir):
        load_path = os.path.join(load_dir, 'model.pt')
        missing_keys, unexpected_keys = self.load_state_dict(
            torch.load(load_path, map_location=torch.device('cpu')), strict=False
        )
        print(missing_keys, unexpected_keys)