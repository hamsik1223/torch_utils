from torch import nn

class NN_simple_classify(nn.Module):
    def __init__(self, num_class, dropout_p= 0, ):
        super(NN_simple_classify, self).__init__()
        self.dropout_p = dropout_p

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(37, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_class),
        )
    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y