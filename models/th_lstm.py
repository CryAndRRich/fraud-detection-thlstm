import torch
import torch.nn as nn
import torch.nn.functional as func

class TimeAwareGate(nn.Module):
    """
    Time-aware Gate to adjust hidden state contribution based on time interval.
    This module helps the model learn the impact of time gaps between transactions
    """
    def __init__(self, hidden_dim: int):
        """
        Initialize the TimeAwareGate
        
        Parameters:
            hidden_dim: The number of hidden units in the LSTM
        """
        super(TimeAwareGate, self).__init__()
        self.W_gamma = nn.Linear(1, hidden_dim)  # Linear layer for time-based transformation
        self.b_gamma = nn.Parameter(torch.zeros(hidden_dim))  # Bias term for flexibility
    
    def forward(self, 
                delta_t: torch.Tensor, 
                h_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TimeAwareGate
        
        Parameters:
            delta_t: Time intervals between transactions
            h_t: Previous hidden state
        
        Returns:
            torch.Tensor: Adjusted hidden state
        """
        gamma_t = torch.exp(-func.relu(self.W_gamma(delta_t) + self.b_gamma))  # Compute time decay factor
        return gamma_t * h_t  # Apply time decay to hidden state

class HistoricalAttention(nn.Module):
    """
    Current-Historical Attention Module to extract dependencies between current and past transactions.
    This module helps identify influential past transactions
    """
    def __init__(self, hidden_dim: int):
        """
        Initialize the HistoricalAttention module
        
        Parameters:
            hidden_dim: The number of hidden units in the LSTM
        """
        super(HistoricalAttention, self).__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim)  # Linear transformation for current hidden state
        self.W_c = nn.Linear(hidden_dim, hidden_dim)  # Linear transformation for past hidden state
        self.v = nn.Linear(hidden_dim, 1)  # Scoring function for attention weights
    
    def forward(self, h_t: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the HistoricalAttention module
        
        Parameters:
            h_t: Current hidden state
            c_t: Past hidden states
        
        Returns:
            torch.Tensor: Context vector computed via attention
        """
        h_t = h_t.unsqueeze(1)
        c_t = c_t.unsqueeze(1)
        score = self.v(torch.tanh(self.W_h(h_t) + self.W_c(c_t)))  # Compute attention scores
        attn_weights = func.softmax(score, dim=1)  # Normalize scores across time steps
        context = torch.sum(attn_weights * c_t, dim=1)  # Compute weighted sum of historical states
        return context

class TH_LSTM(nn.Module):
    """
    Time-aware Historical-attention-based LSTM (TH-LSTM).
    This model integrates time decay and historical attention mechanisms for improved sequence modeling
    """
    name: str = "th_lstm"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 batch_size: int = 1000,
                 dropout_rate: float = 0.8):
        """
        Initialize the TH_LSTM model
        
        Parameters:
            input_dim: The number of input features per time step
            hidden_dim: The number of hidden units in the LSTM
            output_dim: The number of output classes or dimensions
            batch_size: The batch size used to initialize hidden and cell states
            dropout_rate: Dropout probability
        """
        super(TH_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)  # LSTM cell for sequential modeling
        self.time_gate = TimeAwareGate(hidden_dim)  # Time decay mechanism
        self.dropout = nn.Dropout(dropout_rate)  # Dropout
        self.attention = HistoricalAttention(hidden_dim)  # Attention mechanism for historical transactions
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Fully connected layer combining h_t and context
        
        self.h_t = torch.zeros(batch_size, hidden_dim)
        self.c_t = torch.zeros(batch_size, hidden_dim)
        self.delta_t = torch.arange(1, batch_size + 1).float().unsqueeze(1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the TH_LSTM model
        
        Parameters:
            x: Input tensor
        
        Returns:
            torch.Tensor: Output tensor containing the predictions
        """
        h_t_candidate = self.time_gate(self.delta_t, self.h_t)  # Apply time-aware gate to previous hidden state
        self.h_t, self.c_t = self.lstm(x, (h_t_candidate, self.c_t))  # Update hidden and cell states via LSTM cell
        self.h_t = self.dropout(self.h_t)  # Apply dropout to hidden state
        context = self.attention(self.h_t, self.c_t)  # Compute attention-weighted historical context
        output = self.fc(torch.cat((self.h_t, context), dim=1))  # Merge hidden state and context for final output
        return output
    
    def _detach(self):
        """
        Detach the hidden and cell states from the current computation graph, 
        ensuring that gradients are not propagated indefinitely across sequences
        """
        self.h_t, self.c_t, self.delta_t = self.h_t.detach(), self.c_t.detach(), self.delta_t.detach()
