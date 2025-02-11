import torch
import torch.nn as nn

class T_LSTM(nn.Module):
    """
    T_LSTM is a time-aware LSTM model that incorporates a learnable time decay factor 
    into the hidden state update. This decay is computed from a provided delta time vector,
    enabling the model to modulate past information based on the time gap between events
    """
    name: str = "t_lstm"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 batch_size: int = 1000, 
                 dropout_rate: float = 0.8):
        """
        Initialize the T_LSTM model

        Parameters:
            input_dim: The number of input features.
            hidden_dim: The number of hidden units in the LSTM cell.
            output_dim: The number of output classes or dimensions.
            batch_size: The batch size used to initialize hidden and cell states
            dropout_rate: Dropout probability 
        """
        super(T_LSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # Linear layer to compute the time decay factor gamma from delta_t
        self.W_gamma = nn.Linear(1, hidden_dim)
        # Learnable bias for gamma computation
        self.b_gamma = nn.Parameter(torch.zeros(hidden_dim))

        # LSTMCell to update the hidden state with input features
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer to transform the hidden state into the final output
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Initialize hidden state and cell state as zero tensors
        self.h_t = torch.zeros(batch_size, hidden_dim)
        self.c_t = torch.zeros(batch_size, hidden_dim)
        
        # Create a delta_t vector representing time intervals for each batch element
        # Shape: (batch_size, 1) with values from 1 to batch_size (as a simple increasing sequence)
        self.delta_t = torch.arange(1, batch_size + 1).float().unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the T_LSTM model

        Parameters:
            x: Input tensor

        Returns:
            torch.Tensor: Output tensor containing the predictions
        """
        gamma_t = torch.exp(-torch.relu(self.W_gamma(self.delta_t) + self.b_gamma))
        h_t_candidate = gamma_t * self.h_t
        self.h_t, self.c_t = self.lstm(x, (h_t_candidate, self.c_t))
        self.h_t = self.dropout(self.h_t) 
        output = self.fc(self.h_t)
        return output

    def _detach(self):
        """
        Detach the hidden and cell states from the current computation graph, 
        ensuring that gradients are not propagated indefinitely across sequences
        """
        self.h_t, self.c_t = self.h_t.detach(), self.c_t.detach()
