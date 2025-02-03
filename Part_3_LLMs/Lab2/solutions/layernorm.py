class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # small value to avoid division by zero(
        self.scale = nn.Parameter(torch.ones(emb_dim)) # scale parameter (learnable)
        self.shift = nn.Parameter(torch.zeros(emb_dim)) # shift parameter (learnable)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True) 
        var = x.var(dim=-1, keepdim=True, unbiased=False) 
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        y = self.scale * norm_x + self.shift
        return y