import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[32, 16]):
        super(QNetwork, self).__init__()
        self.layers = nn.ModuleList()
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            prev_size = size
        self.layers.append(nn.Linear(prev_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
