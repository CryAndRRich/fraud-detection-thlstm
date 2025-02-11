import torch
import torch.nn as nn
import torch.nn.functional as func

class TemporalAttention(nn.Module):
    """
    TemporalAttention applies an attention mechanism over the temporal outputs of an LSTM
    """
    def __init__(self, hidden_dim: int):
        """
        Initialize the TemporalAttention module

        Parameters:
            hidden_dim: The dimension of the LSTM hidden state
        """
        super(TemporalAttention, self).__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)  
        self.V = nn.Linear(hidden_dim, 1)          

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TemporalAttention module

        Parameters:
            lstm_output: Output from an LSTM 

        Returns:
            context: The context vector computed as the weighted sum of LSTM outputs
        """
        intermediate = torch.tanh(self.W(lstm_output))
        scores = self.V(intermediate).squeeze(-1)
        attn_weights = func.softmax(scores, dim=1)
        context = torch.sum(attn_weights.unsqueeze(-1) * lstm_output, dim=1)
        return context

class TAH_LSTM(nn.Module):
    """
    TAH_LSTM integrates an LSTM with a temporal attention mechanism to process sequential data 
    and generate predictions
    """
    name: str = "tah_lstm"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_layers: int = 1, 
                 dropout_rate: float = 0.8):
        """
        Initialize the TAH_LSTM model

        Parameters:
            input_dim: The number of features in the input
            hidden_dim: The number of hidden units in the LSTM
            output_dim: The number of output classes or dimensions
            num_layers: The number of LSTM layers
            dropout_rate: Dropout probability
        """
        super(TAH_LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.temporal_attn = TemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TAH_LSTM model

        Parameters:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor containing the predictions
        """
        # Add a time dimension to the input tensor:
        # reshape from (batch_size, input_dim) to (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        context = self.temporal_attn(lstm_out)
        context = self.dropout(context) 
        output = self.fc(context)
        return output
