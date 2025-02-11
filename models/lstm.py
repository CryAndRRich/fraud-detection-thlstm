import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    LSTM model that processes sequential data using an LSTM layer 
    followed by a fully connected layer for classification
    """
    name: str = "lstm"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 1, 
                 dropout_rate: float = 0.8):
        """
        Initialize the LSTM model
        
        Parameters:
            input_dim: The number of input features per time step
            hidden_dim: The number of features in the hidden state of the LSTM
            output_dim: The number of output classes
            num_layers: The number of LSTM layers
            dropout_rate: Dropout probability
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, 
                            dropout=dropout_rate if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LSTM model
        
        Parameters:
            x: Input tensor
        
        Returns:
            torch.Tensor: Output tensor containing the model predictions
        """
        x, _ = self.lstm(x)
        x = self.dropout(x) 
        x = self.fc(x)
        return x
