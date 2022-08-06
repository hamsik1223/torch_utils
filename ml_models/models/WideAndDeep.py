from torch import nn 
import torch
from ml_models.MLP import MLP_model

class Embedding_Layer(nn.Module):
    def __init__(self,
                embed_vocab,  
                embed_dim):
        super(Embedding_Layer, self).__init__()
        
        self.embed_layer = nn.Embedding(embed_vocab,embed_dim)
        self.layernorm = nn.LayerNorm(embed_dim)
      
    def forward(self, x):
        x = self.embed_layer(x)
        x = self.layernorm(x)
        x = x.flatten(1)
        return x 


class WideandDeep_model(nn.Module):
    def __init__(self,
                in_dim,
                embed_vocab,  
                embed_dim ,
                embed_h_units,
                hidden_units,
                dropout_rates,
                out_dim=1,
                activation_fn = nn.SiLU()):
        super(WideandDeep_model, self).__init__()
        '''
        control the output by the out_dim, and thus it can do regression or classification
        '''
        
        self.bn = nn.BatchNorm1d(in_dim[0])

        self.embedding_layer = Embedding_Layer(embed_vocab, embed_dim)

        if embed_h_units:
            self.embedding_mlp = MLP_model(in_dim = embed_dim * in_dim[1],
                                        hidden_units = embed_h_units, 
                                        dropout_rates = dropout_rates,
                                        out_dim = embed_h_units[-1], 
                                        use_bn = False)
        else:
            self.embedding_mlp = None
        self.ffn_layers = nn.ModuleList([])

        if embed_h_units:
            concat_in_dim = in_dim[0] + embed_h_units[-1]
        else:
            concat_in_dim = in_dim[0] + embed_dim * in_dim[1]

        self.concat_mlp = MLP_model(in_dim = concat_in_dim,
                                     hidden_units = hidden_units, 
                                     dropout_rates = dropout_rates,
                                     out_dim = 1,
                                     use_bn = True)

    def forward(self, x, x_dis):
        x = self.bn(x)
        x_embed = self.embedding_layer(x_dis)
        if self.embedding_mlp:
            x_embed = self.embedding_mlp(x_embed)

        x = torch.cat([x, x_embed], 1)
        x = self.concat_mlp(x)

        return x
