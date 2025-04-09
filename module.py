import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.nn import init
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# saq
class block1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(block1, self).__init__()
        # (1x5) ---- (5x1)
        self.conv1_1 = nn.Conv2d(in_channel, out_channel, 1,0)
        self.conv3_1 = nn.Conv2d(in_channel, out_channel, (3,1), padding=1)
        self.conv1_3 = nn.Conv2d(in_channel, out_channel, (1,3), padding=0)

        self.conv5_1 = nn.Conv2d(in_channel, out_channel, (5,1), padding=2)
        self.conv1_5 = nn.Conv2d(in_channel, out_channel, (1,5), padding=0)


        self.gn = nn.GroupNorm(32, None, eps=1e-6)
        self.gelu = nn.GELU(inplace=True)

    def inforward(self, x):
        y1_1 = self.gn(self.conv1_3(self.conv3_1(x)))
        y2_1 = self.gn(self.conv5_1(self.conv1_5(x)))
        y3 = self.gelu(self.gn(self.conv1_1(x)))

        y1_2 = torch.cat(y1_1, y2_1)
        y2_2 = torch.cat(y2_1, y1_1)

        y1_3 = self.gelu(self.gn(self.conv3_3(y1_2)))
        y2_3 = self.gelu(self.gn(self.conv3_3(y2_2)))
        #y1_3 = self.gelu(self.gn(self.DWconv3_3(y1_2)))
        #y2_3 = self.gelu(self.gn(self.DWconv5_5(y2_2)))

        y2 = self.conv1_1(torch.cat(y1_3, y2_3))

        y = y2 + y3

        return y

class block2(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(block2, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channel, out_channel, 1,0)
        self.conv3_3 = nn.Conv2d(in_channel, out_channel,(3,3), padding=1)

        self.gn = nn.GroupNorm(32, 16, eps=1e-6)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(out_channel)

    def inforward(self, x):
        y1_1 = self.conv3_3(x)
        y1_2 = torch.cat(y1_1, y1_1)
        y1_2 = self.conv3_3(y1_2)
        y1_3 = torch.cat(y1_1, y1_2)
        y1_4 = self.gelu(self.gn(self.conv1_1(self.gap(y1_3))))
        y1_4 = y1_2 * y1_4
        y1_4 = x + y1_4
        y1_4 = self.conv3_3(y1_4)
        y1_5 = y1_4 + y1_4
        y1_5 = self.conv3_3(y1_5)
        y1_5 = y1_4 + y1_5
        y = self.sigmoid(self.relu(self.bn(self.conv1_1(self.gap(y1_5)))))


class block3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(block3, self).__init__()

        self.conv1_1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, padding=0)
        self.conv3_3 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=3)

        self.gn = nn.GroupNorm(32, None, eps=1e-6)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm2d(True)

    def inforward(self, x):
        y1_1 = self.conv3_3(x)
        y2_1 = self.conv3_3(x)

        y1_1 = torch.cat(y1_1, y2_1)
        y1_1 = self.conv3_3(y1_1)

        y3_1 = torch.cat(y1_1, y2_1)
        y3_1 = self.sigmoid(self.gelu(self.gn(self.conv1_1(self.gap(y3_1)))))

        y1_2 = y3_1 * y1_1
        y2_2 = y3_1 * y2_1
        y = y1_2 + y2_2
        return y


class AGCA(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(AGCA, self).__init__()
        hide_channel = in_channel // ratio#隐藏层通道数目为输入通道数除以比例
        self.avg_pool = nn.AdaptiveAvgPool2d(1)#自适应平均池化 特征图缩小到1×1
        self.conv1 = nn.Conv2d(in_channel, hide_channel, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(2)#在第三个维度应用
        # Choose to deploy A0 on GPU or CPU according to your needs
        self.A0 = torch.eye(hide_channel).to(device)#在gpu上初始化矩阵
        # self.A0 = torch.eye(hide_channel)
        # self.A0 = torch.eye(hide_channel)
        # self.A0 = torch.eye(hide_channel)
        # A2 is initialized to 1e-6
        self.A2 = nn.Parameter(torch.FloatTensor(torch.zeros((hide_channel, hide_channel))), requires_grad=True)

        init.constant_(self.A2, 1e-6)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=1, bias=False).to(device)
        self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(hide_channel, in_channel, kernel_size=1, bias=False).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y)
        B, C, _, _ = y.size()#bastchsize channel h*w
        y = y.flatten(2).transpose(1, 2)#将输入张量的第二维和第三维展平，然后交换第一维和第二维的位置
        A1 = self.softmax(self.conv2(y))
        A1 = A1.expand(B, C, C).to(device)#size expand

        A0 = self.A0.to(device)
        A2 = self.A2.to(device)
        # A = (self.A0 * A1) + self.A2
        A = (A0 * A1) + A2
        y = y.to(device)
        y = torch.matmul(y, A).to(device)
        y = self.relu(self.conv3(y))
        y = y.transpose(1, 2).view(-1, C, 1, 1).to(device)#调整y的维度
        y = self.sigmoid(self.conv4(y)).to(device)

        out = x.to(device) * y
        return out.to(device)



class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=2,
                                    dilation=2,
                                    groups=in_channel)
        self.depth_conv2 = nn.Conv2d(in_channels=in_channel,
                                     out_channels=in_channel,
                                     kernel_size=3,
                                     stride=1,
                                     padding=4,
                                     dilation=4,
                                     groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, (3, 1), padding=1),
            nn.Conv2d(in_channel, in_channel, (1, 3)),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )

        self.att = AGCA(out_channel)#加了这个
        self.cout = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        # self.cac = Cac(out_channel)

    def forward(self, input):
        h, w = input.size(2), input.size(3)
        # o = self.pool(input)
        o = self.conv1(input)
        # o = F.interpolate(o, size=(h, w), mode="bilinear", align_corners=True)

        out1 = self.depth_conv(input)
        out = self.depth_conv2(out1) + out1 + o

        y = self.point_conv(out)
        # y = self.cout(torch.cat([o,out],dim=1))
        y = self.att(y)
        # return self.cac(y)
        return y

class absy(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(absy, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        # print("sim......")
        b, c, h, w = x.size()

        n = w * h - 1
        # x: 1, 2, 3, 3
        xmean = x.mean(dim=[2, 3], keepdim=True)
        # xmean: 1, 2, 1, 1
        x_minus_mu_square = self.activaton((abs(x - xmean)))
        y = x_minus_mu_square

        return x * self.activaton(y)


# Channel adaptive calibration
class Cac(nn.Module):
    def __init__(self, channel):
        super(Cac, self).__init__()
        # path a
        self.pa_con = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

        # self.pa_pool = nn.AdaptiveAvgPool2d(1)

        # path b
        self.pb_max = DownSample(channel)
        self.pb_con = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.pb_up = nn.ConvTranspose2d(channel, channel, 2, stride=2)
        self.pb_pool = nn.AdaptiveAvgPool2d(1)

        # path out
        self.out_con = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, kernel_size=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )
        self.convo = nn.Sequential(
            nn.Conv2d(channel, channel, 1),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        # self.absy = absy()

    def forward(self, x):
        # pa_pool = self.pa_pool(self.pa_con(x))
        # 1
        pa_pool = self.pa_con(x)
        # 2


        pb_pool = self.pb_pool(self.pb_up(self.pb_con(self.pb_max(x))))

        w = pa_pool + pb_pool
        out = self.out_con(x) * w
        return out


class Attention(nn.Module):
    def __init__(self, in_channel, ratio=2):
        super(Attention, self).__init__()
        self.W_c1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel // ratio, kernel_size=1),
            nn.BatchNorm2d(in_channel // ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel // ratio, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.W_c2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv3x3 = nn.Conv2d(in_channel, 1, kernel_size=3, padding=1)
        self.out = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        W_c_1 = self.W_c1(x)
        W_c_2 = self.W_c2(W_c_1)
        W_c = self.pool(W_c_2)
        x1 = self.relu1(self.bn1(self.conv1x1(x)))
        W_s = self.conv3x3(x1)
        M_c = W_c * W_c_1
        M = W_s * M_c
        # y = torch.cat([x,M],dim=1)
        y = x + M
        y = self.out(y)
        return y


class DownSample(nn.Module):
    def __init__(self, channel):
        super(DownSample, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()

        )  ###下采样 通道不变，图像大小减半

    def forward(self, x):
        return self.layer(x)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups)
        self.bn = nn.BatchNorm2d(out_planes, momentum=0.01) \
            if bn else None
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, kernel_size=3, padding=1, groups=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, groups=1)
        )
        self.relu = nn.ReLU() if relu else None


# Domain adaptive adjustment (Daa)
class Daa(nn.Module):
    def __init__(self, inchannel_e, inchannel_d):
        super(Daa, self).__init__()
        # e
        self.cone1 = nn.Sequential(
            nn.Conv2d(inchannel_e, inchannel_e, 1),
            nn.BatchNorm2d(inchannel_e),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel_e, 1, 1)
        )
        # d
        self.cond1 = nn.Sequential(
            nn.Conv2d(inchannel_d, inchannel_d, 1),
            nn.BatchNorm2d(inchannel_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel_d, 1, 1)
        )
        self.con7 = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=3),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )

    def forward(self, e, d):
        es = self.cone1(e)
        ds = self.cond1(d)
        s = torch.cat([es, ds], dim=1)
        s = self.con7(s)
        fe = e * s
        fd = d * s
        out = torch.cat([fe, fd], dim=1)
        return out


# Multi-scale adaptive calibration (Mac)
class Mac(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(Mac, self).__init__()

        self.inicon = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )

        self.con1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.con3 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, (3, 1), padding=1),
            nn.Conv2d(inchannel, outchannel, (1, 3)),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.con5 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, (3, 1), padding=1),
            nn.Conv2d(inchannel, inchannel, (1, 3)),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, inchannel, (3, 1), padding=1),
            nn.Conv2d(inchannel, outchannel, (1, 3)),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.depth_conv_b1 = nn.Conv2d(in_channels=inchannel,
                                       out_channels=inchannel,
                                       kernel_size=3,
                                       stride=1,
                                       padding=4,
                                       dilation=4,
                                       groups=inchannel)
        self.point_conv_b2 = nn.Sequential(
            nn.Conv2d(in_channels=inchannel,
                      out_channels=outchannel,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      groups=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        self.pool = DownSample(inchannel)
        self.conm = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, 1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True)
        )
        self.up = nn.ConvTranspose2d(inchannel, outchannel, 2, stride=2)
        self.c1 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
        )
        self.c2 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
        )
        self.c3 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
        )

        # self.ta = TripletAttention()

    def forward(self, x):
        # init
        # x = self.inicon(x)
        # f4
        fm = self.conm(self.pool(x))
        f4 = self.up(fm)

        # f3
        # f3 = self.c1(self.con5(x) + f4)
        # f3 = self.con5(x)
        f3 = self.point_conv_b2(self.depth_conv_b1(x)) + f4

        # f2
        f2 = self.c2(self.con3(x) + f3)
        # f2 = self.con3(x) + f3
        # f1
        f1 = self.c3(self.con1(x) + f2)
        # f1 = self.con1(x) + f2

        return f1


class DoubleConv(nn.Module):
    def __init__(self, inchannel, outchannel):
        super(DoubleConv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(inchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )

        # self.ta = TripletAttention(outchannel)

    def forward(self, x):
        return self.ta(self.layer(x))


class channel_bind(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(channel_bind, self).__init__()

        self.r1 = ResidualBlock(in_ch + out_ch, in_ch//16)
        self.r2 = ResidualBlock(in_ch//16, in_ch)

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        x = self.r1(x)
        x = self.r2(x)
        #    x = self.p1(x, masks)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        # self.se = SELayer(out_c, out_c)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        # x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    # torch.manual_seed(seed=20200910)
    model = block2(64,128)
    input = torch.rand(1, 64, 224, 224)
    out1 = model(input).to(device)
    #
    print(out1.shape)
    #
    # summary(model, (64, 224, 224), device='cpu')
