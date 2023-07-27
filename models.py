from sqlite3 import connect
from torch import nn
import torch
import torch.nn.functional as F
torch.random.manual_seed(1234)

class Xvector(nn.Module):
    def __init__(self,args):
        super().__init__()
        in_channels = args.n_mels
        cnn_channels = [int(idx) for idx in args.cnn_channels.split(',')]
        cnn_kernel = [int(idx) for idx in args.cnn_kernel.split(',')]
        cnn_dilation = [int(idx) for idx in args.cnn_dilation.split(',')]
        n_embed = args.n_embed
        self.blocks = nn.ModuleList()
        self.activation = nn.LeakyReLU()

        for block_index in range(len(cnn_channels)):
            out_channels = cnn_channels[block_index]
            self.blocks.extend([nn.Conv1d(in_channels=in_channels,
                                            out_channels = out_channels,
                                            kernel_size = cnn_kernel[block_index],
                                            dilation = cnn_dilation[block_index]),
                                self.activation,
                                nn.BatchNorm1d(out_channels)])
            in_channels = cnn_channels[block_index]
        
        self.pooling = StatisticsPooling()
        self.linear = nn.Sequential(nn.Linear(out_channels*2,n_embed),
                                    self.activation,
                                    nn.BatchNorm1d(n_embed),
                                    nn.Linear(n_embed,n_embed),
                                    self.activation,
                                    nn.BatchNorm1d(n_embed)
                                                )
    def forward(self,x):
        #x : N X T X F
        x = x.transpose(1,2)
        #x : N X F X T
        for layer in self.blocks:
            x = layer(x)
            #x : N X F X T
        x = x.transpose(1,2)
        # x : N X T X F
        x = self.pooling(x)
        # x : N x 1 x 2F
        x = x.squeeze(1)
        # x : N X 2F
        x = self.linear(x)
        # x : N X E
        return x 

class Xattention(nn.Module):
    def __init__(self,args):
        super().__init__()
        in_channels = args.n_mels
        cnn_channels = [int(idx) for idx in args.cnn_channels.split(',')]
        cnn_kernel = [int(idx) for idx in args.cnn_kernel.split(',')]
        cnn_dilation = [int(idx) for idx in args.cnn_dilation.split(',')]
        n_heads = args.n_heads
        n_embed = args.n_embed
        self.blocks = nn.ModuleList()
        self.activation = nn.LeakyReLU()

        for block_index in range(len(cnn_channels)):
            out_channels = cnn_channels[block_index]
            self.blocks.extend([nn.Conv1d(in_channels=in_channels,
                                            out_channels = out_channels,
                                            kernel_size = cnn_kernel[block_index],
                                            dilation = cnn_dilation[block_index]),
                                self.activation,
                                nn.BatchNorm1d(out_channels)])
            in_channels = cnn_channels[block_index]
        self.attention = nn.MultiheadAttention(embed_dim = out_channels,num_heads = n_heads)
        self.bn = nn.BatchNorm1d(out_channels)
        self.poolimg = StatisticsPooling()
        self.fc1 = nn.Sequential(nn.Linear(2*out_channels,n_embed),
                                self.activation,
                                nn.BatchNorm1d(n_embed))
        self.fc2 = nn.Sequential(nn.Linear(n_embed,n_embed),
                                self.activation,
                                nn.BatchNorm1d(n_embed))
    def forward(self,x):
        # x : N x T x F
        x = x.transpose(1,2)
        # x : N x F x T
        for layer in self.blocks:
            x = layer(x)
            # x: N x F x T
        x_p = x.permute(2,0,1)
        # x : T x N X F
        x_attention = self.attention(x_p,x_p,x_p)
        # x : T X N X F
        x_attention = x_attention[0].permute(1,2,0)
        # x : N X F X T
        x_attention = self.activation(x_attention)
        # x : N X F X T

        x = x + x_attention
        # x : N X F X T
        x = self.bn(x)
        # x : N X F X T
        x = x.transpose(1,2)
        # x : N X T X F
        pooling = StatisticsPooling()
        x = pooling(x)
        # x : N x 1 x 2F
        x = x.squeeze(1)
        # x : N x 2F
        x = self.fc1(x)
        # x : N X E

        x = self.fc2(x)
        # x : N X E

        return x
class BCResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv_5x5 = nn.Sequential(nn.Conv2d(1, 128, 5, stride=(2,1), padding=2),
                                     nn.BatchNorm2d(128),
                                     nn.LeakyReLU())

        self.bcres_1 = BCResBlock(128, 64, 1, 1)
        self.bcres_2 = BCResBlock(64, 64, 1, 1)

        self.bcres_3 = BCResBlock(64, 64, (2,1), (1,2))
        self.bcres_4 = BCResBlock(64, 96, 1, (1,2))

        self.bcres_5 = BCResBlock(96, 96, (2,1), (1,4))
        self.bcres_6 = BCResBlock(96, 96, 1, (1,4))
        self.bcres_7 = BCResBlock(96, 96, 1, (1,4))
        self.bcres_8 = BCResBlock(96, 128, 1, (1,4))

        self.bcres_9 = BCResBlock(128, 128, 1, (1,8))
        self.bcres_10 = BCResBlock(128, 128, 1, (1,8))
        self.bcres_11 = BCResBlock(128, 128, 1, (1,8))
        self.bcres_12 = BCResBlock(128, 160, 1, (1,8))

        self.dwconv_5x5 = nn.Conv2d(160, 160, 5, groups=20, padding=(0,2))
        self.conv_1x1 = nn.Conv2d(160, 256, 1)

        self.conv_out = nn.Conv2d(256, args.n_embed, 1)

    def forward(self, x):
        # input: N x T x F
        x = x.transpose(1,2).unsqueeze(1)
        # N x 1 x F x T

        x = self.conv_5x5(x)

        x = self.bcres_1(x)
        x = self.bcres_2(x)

        x = self.bcres_3(x)
        x = self.bcres_4(x)

        x = self.bcres_5(x)
        x = self.bcres_6(x)
        x = self.bcres_7(x)
        x = self.bcres_8(x)

        x = self.bcres_9(x)
        x = self.bcres_10(x)
        x = self.bcres_11(x)
        x = self.bcres_12(x)

        x = self.dwconv_5x5(x)
        x = self.conv_1x1(x)
        x = x.mean(-1).unsqueeze(-1)
        x = self.conv_out(x)
        x = torch.squeeze(x)

        return x


class BCResBlock(nn.Module):
    def __init__(self, in_size, out_size, stride=1, dilation=1, S=5):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.F2 = nn.Sequential(nn.Conv2d(in_size, out_size, 1),
                                nn.BatchNorm2d(out_size),
                                nn.LeakyReLU(),
                                DepthWiseConv(out_size, out_size, stride, dilation),
                                SubSpectralNorm(out_size, S)
                               )
        self.F1 = nn.Sequential(DepthSeparableConv(out_size, out_size),
                                nn.BatchNorm2d(out_size),
                                nn.SiLU(),
                                nn.Conv2d(out_size, out_size, 1)      
                                )
    def forward(self, x):
        x2 = self.F2(x)
        xp = x2.mean(2).unsqueeze(2)
        x1 = self.F1(xp)
        return F.leaky_relu(x2 + x1)


class DepthWiseConv(nn.Module):
    def __init__(self, in_size, out_size, stride=1, dilation=1):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_size,
                                    out_channels=out_size,
                                    kernel_size=(3,1),
                                    stride=stride,
                                    padding=(1,0),
                                    dilation=dilation,
                                    groups=in_size)
    def forward(self, x):
        x = self.depth_conv(x)
        return x
class DepthSeparableConv(nn.Module):
    def __init__(self, in_size, out_size, stride=1, dilation=1):
        super().__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_size,
                                    out_channels=in_size,
                                    kernel_size=(1,3),
                                    stride=stride,
                                    padding=(0,1),
                                    dilation=dilation,
                                    groups=in_size)
        self.point_conv = nn.Conv2d(in_channels=in_size,
                                    out_channels=out_size,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class SubSpectralNorm(nn.Module):
    def __init__(self, C, S=5, eps=1e-5, affine=True):
        super().__init__()
        self.C = C
        self.S = S
        self.eps = eps
        self.gamma = 1.
        self.beta = 0.
        if affine:
            self.gamma = nn.Parameter(torch.FloatTensor(1, C*S, 1, 1))
            self.beta = nn.Parameter(torch.FloatTensor(1, C*S, 1, 1))
            nn.init.xavier_uniform_(self.gamma)
            nn.init.xavier_uniform_(self.beta)

    def forward(self, x):
        S, eps, gamma, beta = self.S, self.eps, self.gamma, self.beta
        N, C, F, T = x.size()
        x = x.view(N, C*S, F//S, T)
        mean = x.mean([0,2,3]).view([1, C*S, 1, 1])
        var = x.var([0,2,3]).view([1, C*S, 1, 1])
        x = gamma * (x - mean) / (var + eps).sqrt() + beta
        return x.view(N, C, F, T)

class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.
    It returns the concatenated mean and std of input tensor.
    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor = sp_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    """
    def __init__(self):
        super().__init__()
        # Small value for GaussNoise
        self.eps = 1e-5

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if lengths is None:
            mean = x.mean(dim=1)
            std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                mean.append(
                    torch.mean(x[snt_id, 1 : actual_size - 1, ...], dim=0)
                )
                std.append(
                    torch.std(x[snt_id, 1 : actual_size - 1, ...], dim=0)
                )
            mean = torch.stack(mean)
            std = torch.stack(std)
        gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
        gnoise = gnoise
        mean += gnoise
        std = std + self.eps
        # Append mean and std of the batch
        pooled_stats = torch.cat((mean, std), dim=1)
        pooled_stats = pooled_stats.unsqueeze(1)
        return pooled_stats

    def _get_gauss_noise(self, shape_of_tensor, device="cpu"):
        """Returns a tensor of epsilon Gaussian noise.
        Arguments
        ---------
        shape_of_tensor : tensor
            It represents the size of tensor for generating Gaussian noise.
        """
        gnoise = torch.randn(shape_of_tensor, device=device)
        gnoise -= torch.min(gnoise)
        gnoise /= torch.max(gnoise)
        gnoise = self.eps * ((1 - 9) * gnoise + 9)
        return gnoise