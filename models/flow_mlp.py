import torch
import torch.nn as nn

class MLPFlowExtractor(nn.Module):
    """
    MLP to extract per-student flow embeddings.
    """
    def __init__(self, in_channels=None, hidden_dim=None, out_dim=None,temporal_pool='mean',dropout=0.1):
        super().__init__()
        assert temporal_pool in ("mean", "max")
        self.in_channels   = in_channels
        self.temporal_pool = temporal_pool
        self.norm = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        if flows.ndim < 5:
            raise ValueError(f"Expected >=5D (N,T,C,H,W); got shape {tuple(flows.shape)}")
        # flows: [N, T, C, H, W]
        x = flows.mean(dim=[3,4])   # spatial mean -> [N, T, C]
        x = x.mean(dim=1)           # temporal mean -> [N, C]
        return self.mlp(x)          # [N, out_dim]
    


# OPTION 1: Simple Temporal MLP (Flatten Time)
class TemporalMLPFlowExtractor(nn.Module):
    """
    MLP that flattens temporal dimension to learn from full sequence.
    """
    def __init__(self, in_channels=2, hidden_dim=512, out_dim=256, seq_len=150):
        super().__init__()
        # Input size is now seq_len * in_channels
        flattened_input_size = seq_len * in_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(flattened_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Added dropout for large input
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.ReLU()
        )
    
    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        # flows: [N, T, C, H, W]
        x = flows.mean(dim=[3,4])   # spatial mean -> [N, T, C]
        x = x.flatten(1)            # flatten time and channels -> [N, T*C]
        return self.mlp(x)          # [N, out_dim]


# OPTION 2: RNN/LSTM Temporal Modeling
class LSTMFlowExtractor(nn.Module):
    """
    LSTM to process temporal sequence of flow features.
    """
    def __init__(self, in_channels=2, hidden_dim=512, out_dim=256, num_layers=2):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 4),
            nn.ReLU()
        )
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 4,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, out_dim),
            nn.ReLU()
        )
    
    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        # flows: [N, T, C, H, W]
        x = flows.mean(dim=[3,4])       # spatial mean -> [N, T, C]
        
        # Process each timestep through spatial MLP
        batch_size, seq_len, channels = x.shape
        x = x.reshape(-1, channels)     # [N*T, C]
        x = self.spatial_mlp(x)         # [N*T, hidden_dim//4]
        x = x.reshape(batch_size, seq_len, -1)  # [N, T, hidden_dim//4]
        
        # Process temporal sequence
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: [N, T, hidden_dim//2]
        
        # Use final hidden state
        final_hidden = lstm_out[:, -1, :]  # [N, hidden_dim//2]
        
        return self.output_mlp(final_hidden)  # [N, out_dim]


# OPTION 3: 1D Convolution Over Time
class Conv1DFlowExtractor(nn.Module):
    """
    1D convolution over temporal dimension to capture temporal patterns.
    """
    def __init__(self, in_channels=2, hidden_dim=512, out_dim=256):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim // 4),
            nn.ReLU()
        )
        
        # 1D convolutions over time
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(hidden_dim // 4, hidden_dim // 2, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling over time
        )
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, out_dim),
            nn.ReLU()
        )
    
    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        # flows: [N, T, C, H, W]
        x = flows.mean(dim=[3,4])       # spatial mean -> [N, T, C]
        
        # Process each timestep through spatial MLP
        batch_size, seq_len, channels = x.shape
        x = x.reshape(-1, channels)     # [N*T, C]
        x = self.spatial_mlp(x)         # [N*T, hidden_dim//4]
        x = x.reshape(batch_size, seq_len, -1)  # [N, T, hidden_dim//4]
        
        # Conv1D expects [N, C, T]
        x = x.transpose(1, 2)           # [N, hidden_dim//4, T]
        x = self.temporal_conv(x)       # [N, hidden_dim//2, 1]
        x = x.squeeze(-1)               # [N, hidden_dim//2]
        
        return self.output_mlp(x)       # [N, out_dim]


# OPTION 4: Transformer-based (Bonus - most sophisticated)
class TransformerFlowExtractor(nn.Module):
    """
    Transformer to capture long-range temporal dependencies in flow.
    """
    def __init__(self, in_channels=2, hidden_dim=512, out_dim=256, num_heads=8, num_layers=3):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU()
        )
        
        # Positional encoding for temporal sequences
        self.pos_encoding = nn.Parameter(torch.randn(150, hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.ReLU()
        )
    
    def forward(self, flows: torch.Tensor) -> torch.Tensor:
        # flows: [N, T, C, H, W]
        x = flows.mean(dim=[3,4])       # spatial mean -> [N, T, C]
        
        # Process spatial features
        batch_size, seq_len, channels = x.shape
        x = x.reshape(-1, channels)     # [N*T, C]
        x = self.spatial_mlp(x)         # [N*T, hidden_dim]
        x = x.reshape(batch_size, seq_len, -1)  # [N, T, hidden_dim]
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Transformer processing
        x = self.transformer(x)         # [N, T, hidden_dim]
        
        # Global average pooling over time
        x = x.mean(dim=1)               # [N, hidden_dim]
        
        return self.output_mlp(x)       # [N, out_dim]


# Usage examples:
# Option 1 - Simple temporal
# flow_ex = TemporalMLPFlowExtractor(in_channels=2, hidden_dim=512, out_dim=256, seq_len=150)

# Option 2 - LSTM
# flow_ex = LSTMFlowExtractor(in_channels=2, hidden_dim=512, out_dim=256, num_layers=2)

# Option 3 - Conv1D
# flow_ex = Conv1DFlowExtractor(in_channels=2, hidden_dim=512, out_dim=256)

# Option 4 - Transformer
# flow_ex = TransformerFlowExtractor(in_channels=2, hidden_dim=512, out_dim=256, num_heads=8, num_layers=3)
