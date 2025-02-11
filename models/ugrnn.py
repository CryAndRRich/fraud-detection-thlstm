import torch
import torch.nn as nn

class UGRNN(nn.Module):
    """
    UGRNN (Update Gate Recurrent Neural Network) is a simple gated recurrent unit that uses an 
    update gate to control the flow of information and update the hidden state
    """
    name: str = "ugrnn"
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 dropout_rate: float = 0.8):
        """
        Initialize the UGRNN model

        Parameters:
            input_dim: The number of input features
            hidden_dim: The dimension of the hidden state
            output_dim: The number of output classes or dimensions
            dropout_rate: Dropout probability
        """
        super(UGRNN, self).__init__()
        self.hidden_dim = hidden_dim

        # Layers for computing the update gate u_t
        self.W_u = nn.Linear(input_dim, hidden_dim)
        self.U_u = nn.Linear(hidden_dim, hidden_dim)

        # Layers for computing the candidate hidden state
        self.W_h = nn.Linear(input_dim, hidden_dim)
        self.U_h = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layer to produce the final output from the hidden state
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Initialize the hidden state as a zero tensor
        self.h_t = torch.zeros(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the UGRNN model

        Parameters:
            x: Input tensorFor batch processing

        Returns:
            torch.Tensor: Output tensor containing the model predictions
        """
        u_t = torch.sigmoid(self.W_u(x) + self.U_u(self.h_t))
        h_t_candidate = torch.tanh(self.W_h(x) + self.U_h(self.h_t))
        self.h_t = u_t * self.h_t + (1 - u_t) * h_t_candidate
        self.h_t = self.dropout(self.h_t)
        output = self.fc(self.h_t)
        return output

    def _detach(self):
        """
        Detach the hidden state from the current computation graph, 
        ensuring that gradients are not propagated indefinitely through time
        """
        self.h_t = self.h_t.detach()
