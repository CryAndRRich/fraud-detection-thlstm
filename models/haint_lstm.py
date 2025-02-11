import torch
import torch.nn as nn
from .sta_nn import SpatialAttention, TemporalAttention

class HAINT_LSTM(nn.Module):
    """
    HAINT_LSTM integrates an LSTM with spatial and temporal attention mechanisms 
    to process sequential data and generate predictions
    """
    name: str = "haint_lstm"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 1, 
                 dropout_rate: float = 0.8):
        """
        Initialize the HAINT_LSTM model

        Parameters:
            input_dim: The number of input features
            hidden_dim: The number of features in the hidden state of the LSTM
            output_dim: The number of output classes or dimensions
            num_layers: The number of LSTM layers. Default is 1
            dropout_rate: Dropout probability (default: 0.8)
        """
        super(HAINT_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, 
                            dropout=dropout_rate if num_layers > 1 else 0)
        
        self.spatial_attn = SpatialAttention(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.temporal_attn = TemporalAttention(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the HAINT_LSTM model

        Parameters:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor containing the model prediction
        """
        # Reshape input to add a temporal dimension:
        # from (batch_size, input_dim) to (batch_size, 1, input_dim)
        x = x.reshape((x.shape[0], 1, x.shape[1]))
        
        lstm_out, _ = self.lstm(x)
        x = self.spatial_attn(lstm_out)
        x = self.dropout1(x)

        x = x.unsqueeze(1)
        x = self.temporal_attn(x)
        x = self.dropout2(x)

        out = self.fc(x)
        return out
