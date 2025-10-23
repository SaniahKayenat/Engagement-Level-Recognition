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
        x = flows.mean(dim=[3,4])   
        x = x.mean(dim=1)         
        return self.mlp(x)     
    


# 1: Simple Temporal MLP (Flatten Time)
class TemporalMLPFlowExtractor(nn.Module):

    def __init__(self, in_channels=2, hidden_dim=512, out_dim=256, seq_len=150):
        super().__init__()

        flattened_input_size = seq_len * in_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(flattened_input_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
            nn.ReLU()
        )
    
    def forward(self, flows: torch.Tensor) -> torch.Tensor:

        x = flows.mean(dim=[3,4]) 
        x = x.flatten(1)           
        return self.mlp(x)        


# 2: RNN/LSTM Temporal Modeling
class LSTMFlowExtractor(nn.Module):

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
        x = flows.mean(dim=[3,4])     
        
        batch_size, seq_len, channels = x.shape
        x = x.reshape(-1, channels)  
        x = self.spatial_mlp(x)        
        x = x.reshape(batch_size, seq_len, -1)  
        
        lstm_out, (hidden, cell) = self.lstm(x) 

        final_hidden = lstm_out[:, -1, :] 
        
        return self.output_mlp(final_hidden)  


#3: 1D Convolution Over Time
class Conv1DFlowExtractor(nn.Module):

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

        x = flows.mean(dim=[3,4])       

        batch_size, seq_len, channels = x.shape
        x = x.reshape(-1, channels)     
        x = self.spatial_mlp(x)       
        x = x.reshape(batch_size, seq_len, -1)  
        
        # Conv1D expects [N, C, T]
        x = x.transpose(1, 2)          
        x = self.temporal_conv(x)       
        x = x.squeeze(-1)             
        
        return self.output_mlp(x)      


#4: Transformer-based 
class TransformerFlowExtractor(nn.Module):

    def __init__(self, in_channels=2, hidden_dim=512, out_dim=256, num_heads=8, num_layers=3):
        super().__init__()
        self.spatial_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU()
        )
    
        # Positional encoding 
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

        x = flows.mean(dim=[3,4])    
        
        batch_size, seq_len, channels = x.shape
        x = x.reshape(-1, channels)  
        x = self.spatial_mlp(x)       
        x = x.reshape(batch_size, seq_len, -1)  
        
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        
        x = self.transformer(x)        
        
        # Global average pooling over time
        x = x.mean(dim=1)              
        
        return self.output_mlp(x)    


