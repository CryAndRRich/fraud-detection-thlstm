import torch
import torch.nn as nn
import torch.nn.functional as func

class SpatialAttention(nn.Module):
    """
    SpatialAttention applies an attention mechanism along the spatial dimension of the input
    """
    def __init__(self, input_dim: int):
        """
        Initialize the SpatialAttention module

        Parameters:
            input_dim: The number of input features
        """
        super(SpatialAttention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SpatialAttention module

        Parameters:
            x: Input tensor

        Returns:
            torch.Tensor: The weighted sum of input features along the spatial dimension
        """
        attn_scores = self.attn(x).squeeze(-1)
        attn_weights = func.softmax(attn_scores, dim=1)
        x_weighted = x * attn_weights.unsqueeze(-1)
        return x_weighted.sum(dim=1)


class TemporalAttention(nn.Module):
    """
    TemporalAttention applies an attention mechanism over the temporal outputs of an LSTM
    """
    def __init__(self, input_dim: int):
        """
        Initialize the TemporalAttention module

        Parameters:
            input_dim: The number of features in the LSTM output.
        """
        super(TemporalAttention, self).__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TemporalAttention module

        Parameters:
            x: Input tensor from LSTM

        Returns:
            torch.Tensor: The weighted sum over time steps of the LSTM output
        """
        attn_scores = self.attn(x).squeeze(-1)
        attn_weights = func.softmax(attn_scores, dim=1)
        x_weighted = x * attn_weights.unsqueeze(-1)
        return x_weighted.sum(dim=1)


class STA_NN(nn.Module):
    """
    STA_NN (Spatial-Temporal Attention Neural Network) integrates spatial and temporal attention 
    mechanisms with an LSTM to capture both spatial and temporal dynamics from the input features
    """
    name: str = "str_nn"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 1, 
                 dropout_rate: float = 0.8):
        """
        Initialize the STA_NN model

        Parameters:
            input_dim: The number of input features
            hidden_dim: The number of features in the hidden state of the LSTM
            output_dim: The number of output classes or dimensions
            num_layers: The number of LSTM layers
            dropout_rate: Dropout probability 
        """
        super(STA_NN, self).__init__()
        self.spatial_attn = SpatialAttention(input_dim) 
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, 
                            dropout=dropout_rate if num_layers > 1 else 0)
        
        self.temporal_attn = TemporalAttention(hidden_dim) 
        self.dropout2 = nn.Dropout(dropout_rate) 
        
        self.fc = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the STA_NN model

        Parameters:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output tensor containing the model predictions
        """
        # Reshape input to add a spatial dimension:
        # from (batch_size, input_dim) to (batch_size, 1, input_dim)
        x = x.reshape((x.shape[0], 1, x.shape[1]))

        x = self.spatial_attn(x) 
        x = self.dropout1(x)
        
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        
        x = self.temporal_attn(lstm_out)
        x = self.dropout2(x)
        
        out = self.fc(x)
        return out
