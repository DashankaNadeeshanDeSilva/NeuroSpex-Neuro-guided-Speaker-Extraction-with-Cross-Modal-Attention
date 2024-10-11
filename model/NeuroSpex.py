import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#from apex import amp
import copy
import numpy as np

from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention

EPS = 1e-8

def _clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Neural Auditory Attention based Target Speaker Extraction
class ConvTasNet_dConv_MHA_Net(nn.Module):
    def __init__(self, args, N=256, L=20, B=256, H=512, P=3, X=4, R=4, causal=False):
        super(ConvTasNet_dConv_MHA_Net, self).__init__()
        self.N, self.L, self.B, self.H, self.P, self.X, self.R = N, L, B, H, P, X, R
        self.args = args
        
        self.speech_encoder = Speech_Encoder(L, N)
        self.eeg_encoder = EEG_Encoder()
        self.separator = Separator(args, N, B, H, P, X, R, causal)
        self.speech_decoder = Speech_Decoder(N, L)

        '''
        Below code initializes the weight matrices of the neural network using Xavier initialization. 
        It is a common practice to apply good weight initialization strategies
        to improve the convergence and performance of neural networks during training. 
        The purpose of using Xavier initialization is to set the initial values of the weights 
        in a way that helps in avoiding issues like vanishing or exploding gradients during training.
        '''

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture, eeg):

        # get mix speech processed
        mixture_w = self.speech_encoder(mixture)

        # get reference signal processed from EEG
        eeg_ref = self.eeg_encoder(eeg)
        
        # seperate target speaker voice given eeg and mix speech
        est_mask = self.separator(mixture_w, eeg_ref)
        
        # masking step to get target speaker speech        
        est_source = self.speech_decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source


class Speech_Encoder(nn.Module):
    def __init__(self, L, N):
        super(Speech_Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2)

    def forward(self, mixture):
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Speech_Decoder(nn.Module):
    def __init__(self, N, L):
        super(Speech_Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L)

    def forward(self, mixture_w, est_mask):
        est_source = mixture_w * est_mask  # [M,  N, K]
        est_source = torch.transpose(est_source, 2, 1) # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source


class Separator(nn.Module):
    def __init__(self, args, N, B, H, P, X, R, causal):
        super(Separator, self).__init__()
        self.args = args

        self.layer_norm = ChannelWiseLayerNorm(N)

        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1)

        self.tcn = _clones(TCN_block(X,P,B,H,causal), R)

        self.mask_conv1x1 = nn.Conv1d(B, N, 1)

        self.cross_attn = _clones(Cross_Attention_Encoder(), R)
        
        self.fusion = _clones(nn.Conv1d(B+128, B, 1, bias=False), R)


    def forward(self, x, eeg):
        D = x.size()[-1]

        mixture_w = x
        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        eeg = F.interpolate(eeg, (D), mode='linear')

        # cross-att and tcn blocks
        for i in range(len(self.tcn)):

            y = self.cross_attn[i](x, eeg) # ok
            
            x = torch.cat((x, y),1)

            x = self.fusion[i](x) # match the dimentions 128(cross-att)+256(x) -> 256(x)

            x = self.tcn[i](x)
            
        x = self.mask_conv1x1(x)
        x = F.relu(x)

        return x


class Cross_Attention_Encoder(nn.Module):

    """
    Cross attention encoder allows the targe or query (eeg signal "eeg") to capture relationship from source or key-value (speech signal "x") 
    Cross attention allows the model to attend to different parts of the source sequence based on the specific needs of the target sequence. 
    """

    def __init__(self):
        super(Cross_Attention_Encoder, self).__init__()
        self.print = True
        self.nheads = 2
        self.eeg_dim = 64
        self.kv_dim = 256

        #self.po_encoding_q = PositionalEncoding(d_model=self.eeg_dim)
        #self.po_encoding_kv = PositionalEncoding(d_model=self.kv_dim)

        self.cross_attention = MultiheadAttention(embed_dim=self.eeg_dim, num_heads=self.nheads, kdim=self.kv_dim, vdim=self.kv_dim)

        self.layernorm = torch.nn.LayerNorm(normalized_shape=self.eeg_dim ,eps=1e-6)
        #self.dropout = torch.nn.Dropout(0.75)


    def forward(self, x, eeg):

        x = torch.permute(x, (2,0,1))
        #x = self.po_encoding_kv(x)
        eeg = torch.permute(eeg, (2,0,1))
        #eeg = self.po_encoding_q(eeg)

        corss_att_proj, _ = self.cross_attention(query=eeg, key=x, value=x)

        corss_att_proj = self.layernorm(eeg + corss_att_proj)

        # step 2
        corss_att_proj = torch.cat((corss_att_proj, eeg),2)

        corss_att_proj = torch.permute(corss_att_proj, (1,2,0))

        return corss_att_proj


class EEG_Encoder_old(nn.Module):  
    """
    EEG encoder block to generate EEG-speaker embeddings
    Uses Transfomer encoders (mha) blocks and Dilated convolutions
    """ 

    def __init__(self):
        super(EEG_Encoder, self).__init__()

        # Dpth-wise Conv 1x1
        self.in_channels = 64
        self.kernal_size = 8
        self.depthwise_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=self.kernal_size, groups=self.in_channels)
        self.prelu1 = nn.PReLU()
        
        # Dilated Conv 1x1
        #self.dilation = 2
        #self.dconv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=self.kernal_size, dilation=self.dilation)
        #self.prelu2 = nn.PReLU()

        # Transformer encoder for Slef-Attention
        self.attention_encoder = TransformerEncoder(TransformerEncoderLayer(d_model=64, nhead=2, dim_feedforward=64*4), num_layers=4)

        
    def forward(self, eeg):

        eeg = eeg.transpose(1,2)

        eeg = self.depthwise_conv(eeg)
        eeg = self.prelu1(eeg)

        #eeg = self.dconv(eeg)
        #eeg = self.prelu2(eeg)

        eeg = torch.permute(eeg, (2,0,1))
        eeg = self.attention_encoder(eeg, mask=None)
        eeg = torch.permute(eeg, (1,2,0))

        return eeg

class EEG_Encoder(nn.Module):  
    """
    EEG encoder block to generate EEG-speaker embeddings
    Uses Transfomer encoders (mha) blocks and Depthwise convolutions
    """ 
    def __init__(self, R=4):
        super(EEG_Encoder, self).__init__()

        # Pre-Conv
        self.in_channels = 64
        self.pre_conv_kernel_size = 3
        self.padding = self.pre_conv_kernel_size // 2  # Calculate padding to maintain same input size
        self.pre_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=self.pre_conv_kernel_size, padding=self.padding)

        # Depthwise Conv - Attention block
        self.R = R
        self.d_Conv_Att_block = nn.ModuleList([])
        for i in range(self.R):
            self.d_Conv_Att_block.append(Depthwise_Conv_Attention_Block())

        
    def forward(self, eeg):

        eeg = eeg.transpose(1,2)
        eeg = self.pre_conv(eeg)

        for i in range(self.R):
            eeg = self.d_Conv_Att_block[i](eeg)

        return eeg


class Depthwise_Conv_Attention_Block(nn.Module):
    def __init__(self):
        super(Depthwise_Conv_Attention_Block, self).__init__()

        # Dpth-wise Conv 1x1
        self.in_channels = 64
        self.kernal_size = 9
        self.padding = (self.kernal_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=self.kernal_size, padding=self.padding, groups=self.in_channels)
        self.prelu = nn.PReLU()
        self.layernorm_depth_conv = nn.LayerNorm(normalized_shape=512 ,eps=1e-6)

        self.nheads = 1
        self.eeg_dim = 64
        self.attention = nn.MultiheadAttention(embed_dim=self.eeg_dim, num_heads=self.nheads)
        self.layernorm_att = nn.LayerNorm(normalized_shape=self.eeg_dim ,eps=1e-6)


    def forward(self, eeg):

        eeg = torch.permute(eeg, (2,0,1))
        eeg_att_proj, _ = self.attention(query=eeg, key=eeg, value=eeg) # query, key, value
        eeg = self.layernorm_att(eeg + eeg_att_proj)
        eeg = torch.permute(eeg, (1,2,0))

        eeg_depth_conv = self.depthwise_conv(eeg)
        eeg_depth_conv = self.prelu(eeg_depth_conv)
        eeg = self.layernorm_depth_conv(eeg + eeg_depth_conv)

        # OLD
        #eeg = torch.permute(eeg, (2,0,1))
        #eeg_att_proj, _ = self.attention(query=eeg, key=eeg, value=eeg) # query, key, value
        #eeg = self.layernorm_att(eeg + eeg_att_proj)
        #eeg = torch.permute(eeg, (1,2,0))

        return eeg


class TCN_block(nn.Module):
    def __init__(self, X, P, B, H, causal):
        super(TCN_block, self).__init__()
        tcn_blocks = []
        for x in range(X):
            tcn_blocks += [Conv1DBlock(B, H, P, dilation = 2**x, causal=causal)]
        self.tcn = nn.Sequential(*tcn_blocks)

    def forward(self, x):
        x = self.tcn(x)
        return x


class Dilated_Conv_Block(nn.Module):
    """
    1-D Dilated Convolution block
        DConv1x1 - ReLu
    """
    def __init__(self):
        super(Dilated_Conv_Block, self).__init__()
        self.eeg_input_dimension=64
        self.kernel_size=4
        self.dilation_filters=64

        dilation = self.kernel_size**1
        padding = (self.kernel_size - 1) * dilation // 2
        self.d_conv1 = torch.nn.Conv1d(
            in_channels=self.eeg_input_dimension,
            out_channels=self.dilation_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=True,
            padding_mode='zeros')

        dilation = self.kernel_size**2
        padding = (self.kernel_size - 1) * dilation // 2
        self.d_conv2 = torch.nn.Conv1d(
            in_channels=self.eeg_input_dimension,
            out_channels=self.dilation_filters,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            groups=1,
            bias=True,
            padding_mode='zeros')

        self.dc_relu = torch.nn.ReLU() 

    def forward(self, eeg):

        eeg_proj = self.d_conv1(eeg)
        eeg_proj = self.dc_relu(eeg_proj)

        eeg_proj = self.d_conv2(eeg_proj)
        eeg_proj = self.dc_relu(eeg_proj)

        return eeg_proj


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = nn.Conv1d(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()

        self.causal = causal
        if self.causal:
            self.lnorm1 = cumulative_ChannelWiseLayerNorm(conv_channels)
            self.lnorm2 = cumulative_ChannelWiseLayerNorm(conv_channels)
        else:
            self.lnorm1 = ChannelWiseLayerNorm(conv_channels)
            self.lnorm2 = ChannelWiseLayerNorm(conv_channels)

        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation)
        self.prelu2 = nn.PReLU()
        
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1)
        # different padding way
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000*4):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class cumulative_ChannelWiseLayerNorm(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cumulative_ChannelWiseLayerNorm, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())


class ChannelWiseLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = torch.transpose(x, 1, 2)
        return x


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result