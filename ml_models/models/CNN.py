import torch
from torch import nn

class Basic_Conv1d(nn.Module):
    def __init__(self,
                in_chnnels,
                 out_channels,
                 dropout_rates,
                activation_fn = nn.SiLU()):
        super(Basic_Conv1d, self).__init__()
        '''
        '''
        ##input size: 
        self.conv1 = nn.Conv1d(1, out_channels, kernel_size=2, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size = 2, stride=2)
        self.flt = nn.Flatten()
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rates)
      
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        #B * 1 * C
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        #B * C_out * C
        # x = self.bn1(x)
        x = self.maxpool(x)
        # print(x.shape)
        #B * C_out * 1
        x = self.flt(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        return x
     

class CNN_1d_Model(nn.Module):
    def __init__(self,
                in_dim,
                cnn_hidden_units,
                hidden_units,
                dropout_rates,
                out_dim=1,
                activation_fn = nn.SiLU()):
        super(CNN_1d_Model, self).__init__()
        '''
        control the output by the out_dim, and thus it can do regression or classification
        '''
        
        self.bn = nn.BatchNorm1d(in_dim)
        self.conv_layer = Basic_Conv1d(in_dim, cnn_hidden_units[0], dropout_rates[0])

        self.ffn_layers = nn.ModuleList([])
        hidden_units = [cnn_hidden_units[0] * int( (1+in_dim)/2) ] + hidden_units
        for i in range(len(hidden_units)-1):
            self.ffn_layers.append(FFN_layer(hidden_units[i], 
                                             hidden_units[i+1],
                                             activation_fn,
                                             dropout_rates[i]))
        self.output_dense = nn.Linear(hidden_units[-1], out_dim)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv_layer(x)
        for layer in self.ffn_layers:
            x = layer(x)
        x = self.output_dense(x)    
        return x


###

class CNN_Model_js_test(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size):
        super(Model, self).__init__()
        '''
        cnn model from https://www.kaggle.com/code/pyoungkangkim/1dcnn-pytorch-jstreet
        '''
        cha_1 = 64
        cha_2 = 128
        cha_3 = 128

        cha_1_reshape = int(hidden_size / cha_1)
        cha_po_1 = int(hidden_size / cha_1 / 2)
        cha_po_2 = int(hidden_size / cha_1 / 2 / 2) * cha_3

        self.cha_1_reshape = cha_1_reshape
        self.cha_1 = cha_1
        self.cha_2 = cha_2
        self.cha_3 = cha_3

        self.batch_norm1 = nn.BatchNorm1d(num_features)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm_c1 = nn.BatchNorm1d(cha_1)
        self.dropout_c1 = nn.Dropout(0.1)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(cha_1, cha_2, kernel_size=5, stride=1, padding=2, bias=False), dim=None)
        self.ave_po_c1 = nn.AdaptiveAvgPool1d(output_size=cha_po_1)

        self.batch_norm_c2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2 = nn.Dropout(0.1)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)

        self.batch_norm_c2_1 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_1 = nn.Dropout(0.3)
        self.conv2_1 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_2, kernel_size=3, stride=1, padding=1, bias=True), dim=None)

        self.batch_norm_c2_2 = nn.BatchNorm1d(cha_2)
        self.dropout_c2_2 = nn.Dropout(0.2)
        self.conv2_2 = nn.utils.weight_norm(nn.Conv1d(cha_2, cha_3, kernel_size=5, stride=1, padding=2, bias=True), dim=None)

        self.max_po_c2 = nn.MaxPool1d(kernel_size=4, stride=2, padding=1)

        self.flt = nn.Flatten()

        self.batch_norm3 = nn.BatchNorm1d(cha_po_2)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(cha_po_2, num_targets))


    def forward(self, x):
        # print(x.shape)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = F.celu(self.dense1(x), alpha=0.06)

        x = x.reshape(x.shape[0], self.cha_1, self.cha_1_reshape)
        # print(x.shape)
        x = self.batch_norm_c1(x)
        x = self.dropout_c1(x)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = self.ave_po_c1(x)
        # print(x.shape)
        x = self.batch_norm_c2(x)
        x = self.dropout_c2(x)
        x = F.relu(self.conv2(x))
        x_s = x

        x = self.batch_norm_c2_1(x)
        x = self.dropout_c2_1(x)
        x = F.relu(self.conv2_1(x))

        x = self.batch_norm_c2_2(x)
        x = self.dropout_c2_2(x)
        x = F.relu(self.conv2_2(x))
        x = x * x_s

        x = self.max_po_c2(x)

        x = self.flt(x)

        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.dense3(x)

        return x