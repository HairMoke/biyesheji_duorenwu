# 来源
# https://github.com/fchest/ListenNet/blob/main/dep/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNA(nn.Module):
    def __init__(self, channels, factor=8):
        super(CNA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=17)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)

    def forward(self, x_1, x_2):
        b, c, h_1, w = x_1.size()
        _, _, h_2, _ = x_2.size()

        group_x1 = x_1.reshape(b * self.groups, -1, h_1, w)  # b*g,c//g,h,w
        group_x2 = x_2.reshape(b * self.groups, -1, h_2, w)  # b*g,c//g,h,w

        x_h1 = self.pool_h(group_x1) # 1024, 2, 64, 1
        x_w1 = self.pool_w(group_x1).permute(0, 1, 3, 2) # 1024,2,241,1

        x_h2 = self.pool_h(group_x2)
        x_w2 = self.pool_w(group_x2).permute(0, 1, 3, 2)

        hw1 = self.conv1x1(torch.cat([x_h1, x_w1], dim=2))
        x_h1, x_w1 = torch.split(hw1, [h_1, w], dim=2)

        hw2 = self.conv1x1(torch.cat([x_h2, x_w2], dim=2))
        x_h2, x_w2 = torch.split(hw2, [h_2, w], dim=2)

        x1 = self.gn(group_x1 * x_h1.sigmoid() * x_w1.permute(0, 1, 3, 2).sigmoid())
        x2 = self.gn(group_x2 * x_h2.sigmoid() * x_w2.permute(0, 1, 3, 2).sigmoid())

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw

        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw

        tmp = torch.cat([torch.matmul(x11, x12), torch.matmul(x21, x22)], dim=2)
        weights = self.conv1d(tmp)
        weights = weights.reshape(b * self.groups, 1, h_2, w)
        return (group_x2 * weights.sigmoid()).reshape(b, c, h_2, w)


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0)


class MaxNormDefaultConstraint(object):
    """
    Applies max L2 norm 2 to the weights until the final layer and L2 norm 0.5
    to the weights of the final layer as done in [1]_.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """

    def apply(self, model):
        last_weight = None
        for name, module in list(model.named_children()):
            if hasattr(module, "weight") and (
                    not module.__class__.__name__.startswith("BatchNorm")
            ):
                module.weight.data = torch.renorm(
                    module.weight.data, 2, 0, maxnorm=2
                )
                last_weight = module.weight
        if last_weight is not None:
            last_weight.data = torch.renorm(last_weight.data, 2, 0, maxnorm=0.5)


class DilatedInception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(DilatedInception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [1, 2, 3, 5]
        cout1 = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout1, (1, kern), dilation=(1, dilation_factor)))
        self.norm = nn.BatchNorm2d(cout)

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class ListenNet(nn.Module):
    """
    ListenNet for the paper
    """

    def __init__(self, chans=64, samples=1000, num_classes=2, kernel=8, depth=16, avepool=10):
        super(ListenNet, self).__init__()
        self.channel_weight2 = nn.Parameter(torch.randn(depth, depth).cuda().float(), requires_grad=True)
        self.align = Align(chans, depth)
        nn.init.xavier_uniform_(self.channel_weight2.data)

        self.temporal = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=depth, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=(1, kernel), groups=depth, bias=False),
            nn.BatchNorm2d(depth),
            nn.GELU(),
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(depth, depth, kernel_size=(1, 1), groups=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.Conv2d(depth, depth, kernel_size=(chans, 1), groups=depth, bias=False),
            nn.BatchNorm2d(depth),
            nn.GELU(),
        )

        self.MSTE = DilatedInception(depth, depth, dilation_factor=1)
        self.BN = nn.BatchNorm2d(depth)
        self.skip_convs0 = nn.Conv2d(in_channels=chans, out_channels=depth, kernel_size=(1, 1))
        self.skip_convs1 = nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=(chans, 1), groups=depth,
                                     bias=False)
        self.dropout = 0.65

        self.skip0 = nn.Conv2d(in_channels=depth,
                               out_channels=depth,
                               kernel_size=(chans, 1),
                               groups=depth,
                               bias=True)

        # example
        out = torch.ones((1, 1, chans, samples))  # 1 1 64 128
        out = self.temporal(out)  # 1 64 64 128
        out = self.spatial(out)
        N, C, H, W = out.size()  # N=1 C=64 H=64 W=121
        self.CNA = CNA(C, C // 2)
        if W < avepool:
            avepool = W

        self.GAP = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 1, avepool)),
            nn.Dropout(p=0.65),
        )
        out = self.GAP(out)

        n_out_time = out.cpu().data.numpy().shape
        print('In ListenNet, n_out_time shape: ', n_out_time)
        feat_dim = n_out_time[-1] * n_out_time[-2] * n_out_time[-3]
        self.classifier = nn.Linear(feat_dim, num_classes)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):  # 64,1,64,128
        # STDE
        Et = self.temporal(x)
        Es = self.spatial(Et)

        # MSTE
        InU = Et
        OutU = self.MSTE(InU)
        skip = self.skip0(F.dropout(OutU, self.dropout, training=self.training))
        skip_resized = F.interpolate(skip, size=(1, Es.size(-1)), mode='bilinear', align_corners=False)
        InEs = skip_resized + Es
        InEs = self.BN(InEs)

        # Align Reshape
        Et = Et.permute(0, 2, 1, 3)
        Et = self.align(Et)
        InEt = torch.einsum('bdcw,sc->bdsw', Et, self.channel_weight2)

        # CNA
        Est = self.CNA(InEt, InEs)

        # Classifer
        Est = self.GAP(Est)
        feat = torch.flatten(Est, 1)
        logits = self.classifier(feat)

        return feat, logits



