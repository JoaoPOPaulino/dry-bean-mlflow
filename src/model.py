import torch.nn as nn


class BeanMLP(nn.Module):
    def __init__(
        self,
        input_size=16,
        hidden_size1=64,
        hidden_size2=32,
        num_classes=7,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),

            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),

            nn.Linear(hidden_size2, num_classes),
        )

    def forward(self, x):
        return self.network(x)