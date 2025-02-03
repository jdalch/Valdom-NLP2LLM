class FeedForward(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, x):
        return self.layers(x)