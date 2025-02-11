import torch
import torch.nn as nn

class Bi_RNN(nn.Module):
    """
    Bi-RNN model that processes sequential data using a bidirectional LSTM layer 
    followed by a fully connected layer for classification, with dropout
    """
    name: str = "bi_rnn"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 1, 
                 dropout_rate: float = 0.8):
        """
        Initialize the Bi_RNN model
        
        Parameters:
            input_dim: The number of input features per time step
            hidden_dim: The number of features in the hidden state of the LSTM (for each direction)
            output_dim: The number of output classes
            num_layers: The number of LSTM layers
            dropout_rate: Dropout probability
        """
        super(Bi_RNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True, 
                            dropout=dropout_rate if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Bi_RNN model
        
        Parameters:
            x: Input tensor
        
        Returns:
            torch.Tensor: Output tensor containing the model predictions
        """
        x, _ = self.lstm(x)
        x = self.dropout(x) 
        x = self.fc(x)
        return x
