import torch

import torch.nn as nn
import torch.nn.functional as F

from lcasr.models.cca_layer import CCALayer
from lcasr.models.custom_modules import LogSpectrogramModule, TemporalBatchNorm


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='linear')


def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
                nn.init.constant_(param[m.hidden_size:m.hidden_size*2], 1)  # fg bias
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    if isinstance(m, nn.ELU):
        m.inplace = True


class NormalizeModule(nn.Module):
    def __init__(self):
        super(NormalizeModule, self).__init__()

    @staticmethod
    def forward(x):
        return F.normalize(x, p=2, dim=1)


class AttentionBlock(nn.Module):
    """ Attention convolutional block used before the audio input """

    def __init__(self, args, num_filters=24, padding_mode='zeros'):
        super(AttentionBlock, self).__init__()

        # y filter map dimensions pre-cca embedding layer
        spec_context = int(args.snippet_len * args.fps)
        y_fm = (args.spec_bins // 16, spec_context // 16)

        # four main convolutional blocks...
        self.block1 = CNNBlock(input_channels=1,
                               output_channels=num_filters,
                               padding_mode=padding_mode)

        self.block2 = CNNBlock(input_channels=num_filters,
                               output_channels=num_filters * 2,
                               padding_mode=padding_mode)

        self.block3 = CNNBlock(input_channels=num_filters * 2,
                               output_channels=num_filters * 4,
                               padding_mode=padding_mode)

        self.block4 = CNNBlock(input_channels=num_filters * 4,
                               output_channels=num_filters * 4,
                               padding_mode=padding_mode)

        # ...followed by a convolutional layer with (spec_context) filter maps...
        self.conv = nn.Conv2d(in_channels=num_filters * 4,
                              out_channels=spec_context,
                              kernel_size=(1, 1),
                              padding=(0, 0))

        # ...then the average of each filter map is computed to generate a (snippet_len, 1, 1) tensor
        self.gap = nn.AvgPool2d(kernel_size=y_fm)

    def forward(self, y):
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)
        y = self.block4(y)

        y = self.conv(y)
        y = self.gap(y)

        y = F.softmax(y, dim=1)

        return y


class CNNEncoder(nn.Module):
    """ encodes snippet into a 'args.snippet_emb_dim' dimensional embedding """
    def __init__(self, args, is_audio=False, num_filters=24, groupnorm=True, normalize_input=False):
        super(CNNEncoder, self).__init__()

        self.linear_num_features = args.snippet_emb_dim
        if is_audio:
            spec_context = int(args.snippet_len * args.fps)
            linear_input_size = self.linear_num_features * (args.spec_bins // 16) * (spec_context // 16)
            freq_bins = args.spec_bins
        else:
            linear_input_size = self.linear_num_features * (args.staff_height // 32) * (args.sheet_context // 32)
            freq_bins = args.staff_height // 2

        modules = []
        if normalize_input:
            modules.append(TemporalBatchNorm(freq_bins, affine=False))

        modules.extend([
            nn.Conv2d(1, num_filters, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, num_filters) if groupnorm else nn.BatchNorm2d(num_filters),
            nn.ELU(inplace=False),
            nn.Conv2d(num_filters, num_filters, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, num_filters) if groupnorm else nn.BatchNorm2d(num_filters),
            nn.ELU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(num_filters, num_filters * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, num_filters * 2) if groupnorm else nn.BatchNorm2d(num_filters * 2),
            nn.ELU(inplace=False),
            nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, num_filters * 2) if groupnorm else nn.BatchNorm2d(num_filters * 2),
            nn.ELU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, num_filters * 4) if groupnorm else nn.BatchNorm2d(num_filters * 4),
            nn.ELU(inplace=False),
            nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, num_filters * 4) if groupnorm else nn.BatchNorm2d(num_filters * 4),
            nn.ELU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, num_filters * 4) if groupnorm else nn.BatchNorm2d(num_filters * 4),
            nn.ELU(inplace=False),
            nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=(3, 3), padding=(1, 1)),
            nn.GroupNorm(1, num_filters * 4) if groupnorm else nn.BatchNorm2d(num_filters * 4),
            nn.ELU(inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(num_filters * 4, self.linear_num_features, kernel_size=(1, 1), padding=(0, 0)),
            nn.GroupNorm(1, self.linear_num_features) if groupnorm else nn.BatchNorm2d(self.linear_num_features),
            # nn.ELU(inplace=False)
        ])

        self.cnn_blocks = nn.Sequential(*modules)

        self.dense = nn.Sequential(nn.Linear(in_features=linear_input_size, out_features=args.snippet_emb_dim),
                                   # nn.LayerNorm(args.snippet_emb_dim) if groupnorm else nn.BatchNorm1d(args.snippet_emb_dim),
                                   # nn.ELU(inplace=False)
                                   )

    def forward(self, x):
        x = self.cnn_blocks(x)

        x = x.view(x.shape[0], -1)
        x = self.dense(x)

        return x


class SequenceEncoder(nn.Module):

    def __init__(self, args, is_audio=False, groupnorm=True, use_cca=False, pre_rnn_norm=False, post_rnn_norm=False,
                 normalize_input=False):
        super(SequenceEncoder, self).__init__()

        # use this only if the snippet sequence preparation is being done on the gpu
        # if is_audio:
        #     spec_context = int(args.snippet_len * args.fps)
        #     self.hop = int(spec_context * (1 - args.spec_overlap))
        # else:
        #     self.hop = int(args.sheet_context // 2 * (1 - args.sheet_overlap))

        self.is_audio = is_audio
        self.cnn_encoder = CNNEncoder(args, is_audio, groupnorm=groupnorm, normalize_input=normalize_input)

        self.use_cca = use_cca
        self.cca_layer = CCALayer(in_dim=args.snippet_emb_dim) if use_cca else None
        self.do_pre_rnn_norm = pre_rnn_norm
        self.pre_rnn_norm = NormalizeModule() if pre_rnn_norm else None

        self.use_lstm = args.use_lstm
        if args.use_lstm:
            self.seq_model = nn.LSTM(input_size=args.snippet_emb_dim, hidden_size=args.rnn_hidden_size, num_layers=1,
                                     batch_first=True, bidirectional=False)
        else:
            self.seq_model = nn.GRU(input_size=args.snippet_emb_dim, hidden_size=args.rnn_hidden_size, num_layers=1,
                                    batch_first=True, bidirectional=False)
        self.emb_encoder = nn.Sequential(
            nn.Linear(args.rnn_hidden_size, args.emb_dim),
            # nn.LayerNorm(args.emb_dim) if groupnorm else nn.BatchNorm1d(args.emb_dim),
            # nn.ELU(inplace=False)
            )

        self.emb_encoder.add_module('post_rnn_norm', NormalizeModule()) if post_rnn_norm else None

    def forward(self, x, hidden=None):

        lengths = [snippet.shape[0] for snippet in x]

        # cnn encoder
        x = self.cnn_encoder(torch.cat(x).permute(0, 1, 3, 2))

        # cca layer
        if self.use_cca:
            x = self.cca_layer(None, x)[1] if self.is_audio else self.cca_layer(x, None)[0]

        # post cca L2 norm
        if self.do_pre_rnn_norm:
            x = self.pre_rnn_norm(x)

        # rnn model
        x = list(torch.split(x, lengths))
        x = torch.nn.utils.rnn.pack_sequence(x, enforce_sorted=False)
        output, hidden = self.seq_model(x, hidden)

        # embedding encoder with post rnn L2 norm
        seq_embedding = self.emb_encoder(hidden[0][-1]) if self.use_lstm else self.emb_encoder(hidden[-1])

        return seq_embedding


class CrossModalEncoder(nn.Module):
    """ two-pathway convolution model for cross-modal embedding learning """

    def __init__(self, args, groupnorm=True, use_cca=False, pre_rnn_norm=True, post_rnn_norm=True):
        """ initialize model  """
        super(CrossModalEncoder, self).__init__()

        num_filters = 24

        # sequential encoder net for sheet path (x)
        self.x_net = SequenceEncoder(args, is_audio=False, groupnorm=False, pre_rnn_norm=pre_rnn_norm,
                                     post_rnn_norm=post_rnn_norm, use_cca=use_cca)

        # sequential encoder net for audio path (y)
        self.y_net = SequenceEncoder(args, is_audio=True, groupnorm=groupnorm, pre_rnn_norm=pre_rnn_norm,
                                     post_rnn_norm=post_rnn_norm, normalize_input=True, use_cca=use_cca)

        # initializing the attention branch in case it is present in the model
        self.use_att = args.use_att

        if self.use_att:
            self.att_layer = AttentionBlock(args, num_filters=num_filters, padding_mode='zeros')

        # uses He uniform initialization for the convolutional and linear layers
        self.apply(initialize_weights)

    def forward(self, x, y, return_att=False):
        """
        Forward pass
        x -> sheet music snippet
        y -> spectrogram excerpt
        """

        # -- view 1 - sheet
        x = self.x_net(x)

        # -- view 2 - spectrogram
        # getting the attention mask if present in the model and applying it to the audio input
        if self.use_att:
            att = self.att_layer(y)
            att = att.permute(0, 3, 2, 1)
            y = torch.mul(y, att) * y.shape[-1]

        y = self.y_net(y)

        if return_att and self.use_att:
            return x, y, att.squeeze()
        return x, y


class CMSnippetEncoder(nn.Module):
    """ two-pathway convolution model for cross-modal embedding learning """

    def __init__(self, args, groupnorm=True, use_cca=True):
        """ initialize model  """
        super(CMSnippetEncoder, self).__init__()

        # sequential encoder net for sheet path (x)
        self.x_net = CNNEncoder(args, is_audio=False, groupnorm=False, normalize_input=False)

        # sequential encoder net for audio path (y)
        self.y_net = CNNEncoder(args, is_audio=True, groupnorm=True, normalize_input=True)

        self.use_cca = use_cca
        if self.use_cca:
            self.cca_layer = CCALayer(in_dim=args.emb_dim)

        # uses He uniform initialization for the convolutional and linear layers
        self.apply(init_weights)

    def forward(self, x, y, return_pre_cca=False):
        """
        Forward pass
        x -> sheet music snippet
        y -> spectrogram excerpt
        """

        # -- view 1 - sheet
        x = self.x_net(x.permute(0, 1, 3, 2))

        # -- view 2 - spectrogram
        y = self.y_net(y.permute(0, 1, 3, 2))

        # returns pre-cca latent representations for refining the model after training
        if return_pre_cca:
            return x, y

        # merge modalities by cca projection
        if self.use_cca:
            x, y = self.cca_layer(x, y)

        # normalizing the output final embeddings to length 1.0
        x = F.normalize(x, p=2, dim=1)
        y = F.normalize(y, p=2, dim=1)

        return x, y
