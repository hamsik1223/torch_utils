import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch_utils.ml_models.modules.regularizer import GaussianNoise

class FFN_layer(nn.Module):
    def __init__(self,
                in_dim,
                out_dim,
                activation_fn = nn.ReLU(),
                dp_rate=None):
        super(FFN_layer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act_fn = activation_fn
        if dp_rate is not None:
            self.dp = nn.Dropout(dp_rate)
        else:
            self.dp = None
    def forward(self, x):
        x = self.act_fn(self.bn(self.fc(x)))
        if self.dp is not None:
            x = self.dp(x)
        return x

        
class MLP_model(nn.Module):
    def __init__(self,
                in_dim,
                hidden_units,
                dropout_rates,
                out_dim=1,
                activation_fn = nn.SiLU()):
        super(MLP_model, self).__init__()
        '''
        control the output by the out_dim, and thus it can do regression or classification
        '''
        
        self.bn = nn.BatchNorm1d(in_dim)
        self.ffn_layers = nn.ModuleList([])

        hidden_units = [in_dim] + hidden_units
        for i in range(len(hidden_units)-1):
            self.ffn_layers.append(FFN_layer(hidden_units[i], 
                                             hidden_units[i+1],
                                             activation_fn,
                                             dropout_rates[i]))
        self.output_dense = nn.Linear(hidden_units[-1], out_dim)

    def forward(self, x):
        x = self.bn(x)
        for layer in self.ffn_layers:
            x = layer(x)
        x = self.output_dense(x)    
        return x
        

class MLP_with_AE_model(nn.Module):
    def __init__(self,
                in_dim,
                hidden_units,
                dropout_rates,
                out_dim=1,
                activation_fn = nn.SiLU()):
        super(MLP_with_AE, self).__init__()
        
        self.bn = nn.BatchNorm1d(in_dim)
        self.encoder = nn.Sequential(GaussianNoise(sigma=dropout_rates[0]),
                                      FFN_layer(in_dim, 
                                             hidden_units[0],
                                             activation_fn,
                                             None))
        self.decoder = nn.Sequential(nn.Dropout(dropout_rates[1]),
                                      nn.Linear(hidden_units[0], in_dim))
        self.ae = nn.Sequential(FFN_layer(in_dim, 
                                          hidden_units[1],
                                          activation_fn,
                                          dropout_rates[2]),
                                nn.Linear(hidden_units[1], out_dim))


        self.ffn_layers = nn.ModuleList([])
        self.ffn_layers.append(nn.BatchNorm1d(in_dim + hidden_units[0]))
        self.ffn_layers.append(nn.Dropout(dropout_rates[3]))
        mlp_hidden_units = [in_dim + hidden_units[0]] + hidden_units[2:]
        for i in range(len(mlp_hidden_units)-1):
            self.ffn_layers.append(FFN_layer(mlp_hidden_units[i], 
                                             mlp_hidden_units[i+1],
                                             activation_fn,
                                             dropout_rates[i]))
            
        self.output_dense = nn.Linear(mlp_hidden_units[-1], out_dim)

    def forward(self, x):
        x0 = self.bn(x)
        x_encoded = self.encoder(x0)
        x_decoded = self.decoder(x_encoded)
        y_ae = self.ae(x_decoded)

        x = torch.cat([x_encoded, x0], dim=1)
        for layer in self.ffn_layers:
            x = layer(x)
        y_mlp = self.output_dense(x)    
        return x_decoded, y_ae, y_mlp
