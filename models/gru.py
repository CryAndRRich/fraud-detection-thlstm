import torch
import torch.nn as nn

class GRU(nn.Module):
    """
    GRU model that processes sequential data using a GRU layer 
    followed by a fully connected layer for classification, with dropout
    """
    name: str = "gru"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 1, 
                 dropout_rate: float = 0.8):
        """
        Initialize the GRU model
        
        Parameters:
            input_dim: The number of input features per time step
            hidden_dim: The number of features in the hidden state of the GRU
            output_dim: The number of output classes
            num_layers: The number of GRU layers
            dropout_rate: Dropout probability
        """
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                          batch_first=True, 
                          dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GRU model
        
        Parameters:
            x: Input tensor
        
        Returns:
            torch.Tensor: Output tensor containing the model predictions
        """
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
