from torch import nn
import torch


class GCN(nn.Module):
    def __init__(self, dim_1_channels, dim_2_channels):
        super().__init__()

        self.conv1d_1 = nn.Conv1d(dim_1_channels, dim_1_channels, 1)
        self.conv1d_2 = nn.Conv1d(dim_2_channels, dim_2_channels, 1)

    def forward(self, x):
        h = self.conv1d_1(x).permute(0, 2, 1)
        return self.conv1d_2(h).permute(0, 2, 1)


class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """

    def __init__(self, num_in, num_mid,
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()

        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(self.num_s, self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04)  # should be zero initialized

    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))

        return out


# class GloRe_Unit(nn.Module):
#     def __init__(self, in_channels, mid_channels, N):
#         super().__init__()
#         self.in_channels = in_channels
#         self.mid_channels = mid_channels
#         self.N = N
#
#         self.phi = nn.Conv2d(in_channels, mid_channels, 1)
#         self.theta = nn.Conv2d(in_channels, N, 1)
#         self.gcn = GCN(N, mid_channels)
#         self.phi_inv = nn.Conv2d(mid_channels, in_channels, 1)
#
#     def forward(self, x):
#         batch_size, in_channels, h, w = x.shape
#         mid_channels = self.mid_channels
#         N = self.N
#
#         B = self.theta(x).view(batch_size, N, -1)
#         x_reduced = self.phi(x).view(batch_size, mid_channels, h * w)
#         x_reduced = x_reduced.permute(0, 2, 1)
#         v = B.bmm(x_reduced)
#
#         z = self.gcn(v)
#         y = B.permute(0, 2, 1).bmm(z).permute(0, 2, 1)
#         y = y.view(batch_size, mid_channels, h, w)
#         x_res = self.phi_inv(y)
#
#         return x + x_res