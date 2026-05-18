import torch
import torch.nn as nn
import torch.nn.functional as F
from module_test.medgnn.layers.Embed import Multi_Resolution_Data, Frequency_Embedding
from module_test.medgnn.layers.Medformer_EncDec import Encoder, EncoderLayer
from module_test.medgnn.layers.SelfAttention_Family import FormerLayer, DifferenceFormerlayer
from module_test.medgnn.layers.Multi_Resolution_GNN import MRGNN
from module_test.medgnn.layers.Difference_Pre import DifferenceDataEmb, DataRestoration


class Model(nn.Module):
    def __init__(self, num_classes=2):
        super(Model, self).__init__()
        # self.enc_in = configs.enc_in   # 通道数，这里是16
        # self.seq_len = configs.seq_len # 256
        # self.d_model = configs.d_model # 256
        # self.d_ff = configs.d_ff # 512
        # self.n_heads = configs.n_heads # 8
        # self.e_layers = configs.e_layers # 4
        # self.dropout = configs.dropout # 0.1
        # self.output_attention = configs.output_attention #False
        # self.activation = configs.activation   # 'gelu'
        # self.resolution_list = list(map(int, configs.resolution_list.split(",")))   # '2,4,6,8'
        #
        #
        # self.res_num = len(self.resolution_list)   # 4
        # self.stride_list = self.resolution_list    # [2, 4, 6, 8]
        # self.res_len = [int(self.seq_len//res)+1 for res in self.resolution_list]  # [129, 65, 43, 33]
        # self.augmentations = configs.augmentations.split(",")   # 'none,drop0.35' -> ['none', 'drop0.35']

        self.enc_in = 64  # 通道数，这里是16
        self.seq_len = 256  # 256
        self.d_model = 256  # 256
        self.d_ff = 512  # 512
        self.n_heads = 8  # 8
        self.e_layers = 4 # 4
        self.dropout = 0.1 # 0.1
        self.output_attention = False  # False
        self.activation = 'gelu'  # 'gelu'
        self.resolution_list = [2, 4, 6, 8]  # '2,4,6,8'

        self.res_num = len(self.resolution_list)  # 4
        self.stride_list = self.resolution_list  # [2, 4, 6, 8]
        self.res_len = [int(self.seq_len // res) + 1 for res in self.resolution_list]  # [129, 65, 43, 33]
        self.augmentations = ['none', 'drop0.35']  # 'none,drop0.35' -> ['none', 'drop0.35']

        self.num_classes = num_classes
        self.nodedim = 10


        configs = self
        configs.resolution_list = '2,4,6,8'
        configs.augmentations = 'none,drop0.35'

        # step1: multi_resolution_data
        self.multi_res_data = Multi_Resolution_Data(self.enc_in, self.resolution_list, self.stride_list)

        # step2.1: frequency convolution network
        self.freq_embedding = Frequency_Embedding(self.d_model, self.res_len, self.augmentations)

        # step2.2: difference attention network
        self.diff_data_emb = DifferenceDataEmb(self.res_num, self.enc_in, self.d_model)
        self.difference_attention = Encoder(
            [
                EncoderLayer(
                    DifferenceFormerlayer(
                        self.enc_in,
                        self.res_num,
                        self.d_model,
                        self.n_heads,
                        self.dropout,
                        self.output_attention
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.data_restoration = DataRestoration(self.res_num, self.enc_in, self.d_model)
        self.embeddings = nn.ModuleList([nn.Linear(res_len, self.d_model) for res_len in self.res_len])

        # step 3: transformer
        self.encoder = Encoder(
            [
                EncoderLayer(
                    FormerLayer(
                        len(self.resolution_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # step 4: multi-resolution GNN
        self.mrgnn = MRGNN(configs, self.res_len)

        # step 5: projection
        self.projection = nn.Linear(self.d_model * self.enc_in, self.num_classes)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        B, T, C = x_enc.shape

        # step1: multi_resolution_data
        multi_res_data = self.multi_res_data(x_enc)

        # step2.1: frequency convolution network
        enc_out_1 = self.freq_embedding(multi_res_data)

        # step2.2: difference attention network
        x_diff_emb, x_padding = self.diff_data_emb(multi_res_data)
        x_diff_enc, attns = self.difference_attention(x_diff_emb, attn_mask=None)
        enc_out_2 = self.data_restoration(x_diff_enc, x_padding)
        enc_out_2 = [self.embeddings[l](enc_out_2[l]) for l in range(self.res_num)]

        # step 3: transformer
        data_enc = [enc_out_1[l] + enc_out_2[l] for l in range(self.res_num)]
        enc_out, attns = self.encoder(data_enc, attn_mask=None)

        # step 4: multi-resolution GNN
        output, adjacency_matrix_list = self.mrgnn(enc_out)

        # step 5: projection
        output = output.reshape(B, -1)
        output = self.projection(output)

        return output