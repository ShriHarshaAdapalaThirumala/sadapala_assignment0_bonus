import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self: 'NeuralNetwork') -> None:
        """
        Initializes the neural network by creating the layers of the network.
        """
        super(NeuralNetwork,self).__init__()
        
        # Create the layers of the neural network
        self.linear1 = torch.nn.Linear(in_features=152, out_features=200)
        self.linear2 = torch.nn.Linear(in_features=200, out_features=100)
        self.linear3 = torch.nn.Linear(in_features=100, out_features=64)
        self.linear4 = torch.nn.Linear(in_features=64, out_features=64)
        self.output = torch.nn.Linear(in_features=64, out_features=1)
    
    def forward(self: 'NeuralNetwork',x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input Tensor with shape (#batches, #rows, #columns)

        Returns:
            torch.Tensor: output predicted value (#batches, #rows, 1)
        """
        # Propagate the input tensor through the layers of the neural network
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        x = torch.nn.functional.relu(self.linear4(x))
        x = self.output(x)
        return x