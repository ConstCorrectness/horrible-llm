
import torch
import torch.nn as nn



HORRIBLE_CONFIG_06_B = {
    'vocab_size': 151_936,
    'context_length': 40_960,
    'emb_dim': 1_024,
    'n_heads': 16,
    'n_layers': 28,
    'hidden_dim': 3_072,
    'head_dim': 1_024 // 16,            # d_model / num_head = d_head
    'qk_norm': True,                    # For Group-Query Attention, we normalize Q and K
    'n_kv_groups': 8,                   # We break up the 16 heads into 8 groups, so there's 16 Q, and each group shares share a (K, V)
    'rope_base': 1_000_000.0,           # RoPE's base 'theta'
    'dtype': torch.bfloat16,            # default is torch.float32.
}



class HorribleModel(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)

        # Token Embedding layer
        self.tok_emb = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'], dtype=cfg['dtype'])

        # Transformer Blocks: (Each has the self-attention --> FFN pair)
        self.trf_blocks =  nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])


        # Final Normalization
        self.final_norm = RMSNorm(cfg['emb_dim'])

        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])

        self.cfg = cfg
        self.current_pos = 0

    # TODO:
    def forward(self, in_idx, cache=None):
        pass



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)
        self.att = GroupedQueryAttention(
            d_in = cfg['emb_dim'],
            num_heads=cfg['n_heads'],
            head_dim=cfg['head_dim'],
            num_kv_groups=cfg['n_kv_groups'],
            qk_norm=cfg['qk_norm'],
            dtype=cfg['dtype']
        )


        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg['emb_dim'], eps=1e-6)
        self.norm2 = RMSNorm(cfg['emb_dim'], eps=1e-6)

    # TODO:
    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        pass



class FeedForward(nn.Module):
    def __init__(self, cfg):
        nn.Module.__init__(self)

        self.fc1 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.fc2 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.fc3 = nn.Linear(cfg['hidden_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)

    # TODO:
    def forward(self, x):
        pass



class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        nn.Module.__init__(self)

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = self.num_heads / self.num_kv_groups

        if head_dim is None:
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    # TODO:
    def forward(self, x, mask, cos, sin, start_pos=0, cache=None):
        pass



class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        nn.Module.__init__(self)

        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None

    # TODO: 
    def forward(self, x):
        pass




