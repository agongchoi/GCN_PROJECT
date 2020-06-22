import torch.nn as nn
import torch.nn.functional as F
import torch

from nets.mca import MSA_ST
from nets.combclassifier import CombinatorialClassifier, CombinatorialClassifierWithSA
from nets.net_utils import FC, MLP, LayerNorm

# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


# --------------------------------------------------------
# ---- Main MCAN Model With Combinatorial Classifiers ----
# --------------------------------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted


class CombiNet(nn.Module):
    def __init__(self, args, feat_dim, answer_size):
        super(CombiNet, self).__init__()

        self.feat_dim = feat_dim
        self.num_metaclassifiers = args.num_partitionings
        self.meta_class = args.num_partitions
        self.hidden_dim = args.hidden_dim
        self.attention = args.attention
        self.mode = args.mode
        self.combination = args.combination

        self.img_feat_linear = nn.Linear(
            self.feat_dim,
            args.hidden_dim * args.num_partitionings
        )

        self.backbone = MSA_ST(args)
        if self.attention:

            self.AtModule = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.hidden_dim, 1),
                nn.Softmax(dim=1)
            )

        #self.attflat_img = AttFlat(args)

        #self.proj_norm = LayerNorm(args.FLAT_OUT_SIZE)
        #self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)
        self.partitionings = torch.load(args.partitionings_path)[0 * args.num_partitionings
                                                                 :(0 + 1) * args.num_partitionings].t()

        self.proj = CombinatorialClassifierWithSA(answer_size, args.num_partitionings, args.num_partitions,
                                            self.hidden_dim, attention=args.attention, mode=self.mode, combination=self.combination)
        self.proj.set_partitionings(self.partitionings)


    def forward(self, img_feat, output_sum=True, return_meta_dist=False, with_feat=False):
        assert self.partitionings.sum() > 0, 'Partitionings is never given to the module.'



        # Pre-process Image Feature
        img_feat = self.img_feat_linear(img_feat).view(-1, self.num_metaclassifiers, self.hidden_dim)


        # Make mask
        #img_feat_mask = self.make_mask(img_feat)
        img_feat_mask = None

        # Backbone Framework
        img_feat = self.backbone(
            img_feat,
            img_feat_mask
        )

        if self.attention:
            attention = self.AtModule(img_feat)
        #proj_feat = lang_feat + img_feat
        #proj_feat = self.proj_norm(proj_feat)
        #proj_feat = torch.sigmoid(self.proj(proj_feat))
        if self.attention:
            logit = self.proj(img_feat, attention_weight=attention)
        else:
            logit = self.proj(img_feat)

        return logit

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
