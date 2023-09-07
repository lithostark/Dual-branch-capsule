# Noahs ark back up
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3'

basic = 1
seed=1
dropRate = 0.3
cuda = torch.cuda.is_available()
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['PYTHONHASHSEED'] = str(seed)
if cuda:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out
    
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, presence):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
            
        # attn -= (1-presence.unsqueeze(1)) * 1e0
        attn = self.dropout( F.softmax(attn,dim = -1) )
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v


        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, presence):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        # residual = k

        q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q, attn = self.attention(q, k, v, presence)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        return q, attn
# ---------------------------------------- capsule --------------------------------------- #
if basic==1:
    class BasicBlock(nn.Module):
        def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
            super(BasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.relu2 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
            self.equalInOut = (in_planes == out_planes)
            self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                    padding=0, bias=False) or None
        def forward(self, x):
            if not self.equalInOut:
                x = self.relu1(self.bn1(x))
            else:
                out = self.relu1(self.bn1(x))
            out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(out)
            return torch.add(x if self.equalInOut else self.convShortcut(x), out)
if basic == 2:
    class BasicBlock(nn.Module):
        def __init__(self, in_planes, out_planes, stride, dropRate=0.0, leakyRate=0.01, actBeforeRes=True):
            super(BasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.relu1 = nn.LeakyReLU(leakyRate, inplace=True)
            # self.relu_1 = nn.ReLU(inplace=False) # change bn to ReLu
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_planes)
            self.relu2 = nn.LeakyReLU(leakyRate, inplace=True)
            # self.relu_2 = nn.ReLU(inplace=False) # change bn to ReLu
            self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                                   padding=1, bias=False)
            self.droprate = dropRate
            self.equalInOut = False # (in_planes == out_planes)
            self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                    padding=0, bias=False) or None
            self.activateBeforeResidual= actBeforeRes
        def forward(self, x):
            if not self.equalInOut and self.activateBeforeResidual:
                x = self.relu1(self.bn1(x))
                # x = self.relu_1(self.bn1(x)) # change bn to ReLu
                out = self.conv1(x)
            else:
                out = self.conv1(self.relu1(self.bn1(x)))
                # out = self.conv1(self.relu_1(self.bn1(x))) # change bn to ReLu
            #out = self.conv1(out if self.equalInOut else x)
            #out = self.conv1(self.equalInOut and out or x)
            if self.droprate > 0:
                out = F.dropout(out, p=self.droprate, training=self.training)
            out = self.conv2(self.relu2(self.bn2(out)))
            # out = self.conv2(self.relu_2(self.bn2(out))) # change bn to ReLu
            res = self.convShortcut(x) if not self.equalInOut else x
            return torch.add(res, out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
    

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, K=3, P=4, iters=3):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        self.primary_caps = PrimaryCaps(nChannels[1], int(nChannels[1]/P/P), 1, P, stride=1)
        self.conv_caps1 = ConvCaps(int(nChannels[1]/P/P),  int(nChannels[2]/P/P), K, P, stride=2, iters=iters, attn_dim=16)
        self.conv_caps2 = ConvCaps(int(nChannels[2]/P/P),  int(nChannels[3]/P/P), K, P, stride=2, iters=iters, attn_dim=16)
        self.class_caps = ConvCaps(int(nChannels[3]/P/P), num_classes, 1, P, stride=1, iters=iters,
                                        coor_add=True, w_shared=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self, x, netTYPE='RESsolo'):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, out.shape[2])
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
    
# --------------------------------- widersenet ----------------------------------- #
class PrimaryCaps(nn.Module):
    r"""Creates a primary convolutional capsule layer
    that outputs a pose matrix and an activation.

    Note that for computation convenience, pose matrix
    are stored in first part while the activations are
    stored in the second part.

    Args:
        A: output of the normal conv layer
        B: number of types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution

    Shape:
        input:  (*, A, h, w)
        output: (*, h', w', B*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*A*B*P*P + B*P*P
    """
    def __init__(self, A=32, B=32, K=1, P=4, stride=1, padding=0):
        super(PrimaryCaps, self).__init__()
        self.pose = nn.Conv2d(in_channels=A, out_channels=B*P*P,
                            kernel_size=K, stride=stride, padding=padding, bias=True)
        self.a = nn.Conv2d(in_channels=A, out_channels=B,
                            kernel_size=K, stride=stride, padding=padding, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.MHAttn = MultiHeadAttention(1,16,16,16)

    def forward(self, x):
        self.pose_out = self.pose(x) # [bs,B*16,8,8]
        a = self.a(x)
        self.a_out = self.sigmoid(a) # [bs,B,8,8]
        out = torch.cat([self.pose_out, self.a_out], dim=1)
        out = out.permute(0, 2, 3, 1)
        return out, self.pose_out.permute(0, 2, 3, 1), self.a_out.permute(0, 2, 3, 1)

        # bs = x.shape[0]
        # p = self.pose(x) # [bs,B*16,16,16]
        # w = p.shape[2]
        # a = self.a(x)
        # a = self.sigmoid(a) # [bs,B,8,8]
        # p = p.permute(0, 2, 3, 1).contiguous().view(bs,-1,16)
        # aw = a.permute(0, 2, 3, 1).contiguous().view(bs,-1,1)
        # pwa,_ = self.MHAttn(p,p,p,aw)
        # pwa = pwa.view(bs,-1,w,w,16).permute(0,2,3,1,4).contiguous().view(bs,w,w,-1)
        # a = a.permute(0, 2, 3, 1)
        # return torch.cat([pwa, a], dim=3)

class ConvCaps(nn.Module):
    r"""Create a convolutional capsule layer
    that transfer capsule layer L to capsule layer L+1
    by EM routing.

    Args:
        B: input number of types of capsules
        C: output number on types of capsules
        K: kernel size of convolution
        P: size of pose matrix is P*P
        stride: stride of convolution
        iters: number of EM iterations
        coor_add: use scaled coordinate addition or not
        w_shared: share transformation matrix across w*h.

    Shape:
        input:  (*, h,  w, B*(P*P+1))
        output: (*, h', w', C*(P*P+1))
        h', w' is computed the same way as convolution layer
        parameter size is: K*K*B*C*P*P + B*P*P
    """
    def __init__(self, B=8, C=16, K=3, P=4, stride=2, iters=3,
                 coor_add=False, w_shared=False, dropout=0.3, attn_dim=4*4):
        super(ConvCaps, self).__init__()
        # TODO: lambda scheduler
        # Note that .contiguous() for 3+ dimensional tensors is very slow
        self.dropout = dropout
        self.B = B
        self.C = C
        self.K = K
        self.P = P
        self.psize = P*P
        self.stride = stride
        self.iters = iters
        self.coor_add = coor_add
        self.w_shared = w_shared
        # constant
        self.eps = 1e-8
        self._lambda = 1e-03
        if cuda:
            self.ln_2pi = nn.Parameter(torch.FloatTensor(1).fill_(math.log(2*math.pi)), requires_grad=False)
        else:
            self.ln_2pi = torch.FloatTensor(1).fill_(math.log(2*math.pi))
#         self.ln_2pi = torch.FloatTensor(1).fill_(math.log(2*math.pi))
        # params
        # Note that \beta_u and \beta_a are per capsule type,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=rJUY2VdbM
        self.beta_u = nn.Parameter(torch.zeros(C))
        self.beta_a = nn.Parameter(torch.zeros(C))
        # Note that the total number of trainable parameters between
        # two convolutional capsule layer types is 4*4*k*k
        # and for the whole layer is 4*4*k*k*B*C,
        # which are stated at https://openreview.net/forum?id=HJWLfGWRb&noteId=r17t2UIgf
        self.weights = nn.Parameter(torch.randn(1, K*K*B, C, P, P))
        # op
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=2)
        
        self.MHAttn = MultiHeadAttention(1,attn_dim,4,16)
        self.att_linear = nn.Linear(in_features=16,out_features=16)
        self.att_linear1 = nn.Linear(in_features=16,out_features=16)

    def m_step(self, a_in, r, v, eps, b, B, C, psize):
        """
            \mu^h_j = \dfrac{\sum_i r_{ij} V^h_{ij}}{\sum_i r_{ij}}
            (\sigma^h_j)^2 = \dfrac{\sum_i r_{ij} (V^h_{ij} - mu^h_j)^2}{\sum_i r_{ij}}
            cost_h = (\beta_u + log \sigma^h_j) * \sum_i r_{ij}
            a_j = logistic(\lambda * (\beta_a - \sum_h cost_h))

            Input:
                a_in:      (b, C, 1)
                r:         (b, B, C, 1)
                v:         (b, B, C, P*P)
            Local:
                cost_h:    (b, C, P*P)
                r_sum:     (b, C, 1)
            Output:
                a_out:     (b, C, 1)
                mu:        (b, 1, C, P*P)
                sigma_sq:  (b, 1, C, P*P)
        """
        r = r * a_in
        r = r / (r.sum(dim=2, keepdim=True) + eps)
        r_sum = r.sum(dim=1, keepdim=True)
        coeff = r / (r_sum + eps)
        coeff = coeff.view(b, B, C, 1)

        mu = torch.sum(coeff * v, dim=1, keepdim=True)
        sigma_sq = torch.sum(coeff * (v - mu)**2, dim=1, keepdim=True) + eps

        r_sum = r_sum.view(b, C, 1)
        sigma_sq = sigma_sq.view(b, C, psize)
        cost_h = (self.beta_u.view(C, 1) + torch.log(sigma_sq.sqrt())) * r_sum

        a_out = self.sigmoid(self._lambda*(self.beta_a - cost_h.sum(dim=2)))
        sigma_sq = sigma_sq.view(b, 1, C, psize)

        return a_out, mu, sigma_sq

    def e_step(self, mu, sigma_sq, a_out, v, eps, b, C):
        """
            ln_p_j = sum_h \dfrac{(\V^h_{ij} - \mu^h_j)^2}{2 \sigma^h_j}
                    - sum_h ln(\sigma^h_j) - 0.5*\sum_h ln(2*\pi)
            r = softmax(ln(a_j*p_j))
              = softmax(ln(a_j) + ln(p_j))

            Input:
                mu:        (b, 1, C, P*P)
                sigma:     (b, 1, C, P*P)
                a_out:     (b, C, 1)
                v:         (b, B, C, P*P)
            Local:
                ln_p_j_h:  (b, B, C, P*P)
                ln_ap:     (b, B, C, 1)
            Output:
                r:         (b, B, C, 1)
        """
        ln_p_j_h = -1. * (v - mu)**2 / (2 * sigma_sq) - torch.log(sigma_sq.sqrt())  - 0.5*self.ln_2pi

        ln_ap = ln_p_j_h.sum(dim=3) + torch.log(a_out.view(b, 1, C))
        r = self.softmax(ln_ap)
#         return r
        return r, ln_p_j_h.sum(dim=3)

    def caps_em_routing(self, v, a_in, C, eps):
        """
            Input:
                v:         (b, B, C, P*P)
                a_in:      (b, C, 1)
            Output:
                mu:        (b, 1, C, P*P)
                a_out:     (b, C, 1)

            Note that some dimensions are merged
            for computation convenient, that is
            `b == batch_size*oh*ow`,
            `B == self.K*self.K*self.B`,
            `psize == self.P*self.P`
        """
        b, B, c, psize = v.shape
        assert c == C
        assert (b, B, 1) == a_in.shape

        if cuda:
            r = Variable(torch.FloatTensor(b, B, C).fill_(1./C), requires_grad=False).to(device)
        else:
            r = torch.FloatTensor(b, B, C).fill_(1./C)
            
        for iter_ in range(self.iters):
            a_out, mu, sigma_sq = self.m_step(a_in, r, v, eps, b, B, C, psize)
            if iter_ < self.iters - 1:
#                 r = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)
                r, p_zx = self.e_step(mu, sigma_sq, a_out, v, eps, b, C)

#         return mu, a_out
        return mu, a_out, r, p_zx, mu, sigma_sq

    def add_pathes(self, x, B, K, psize, stride):
    # yunqi --- this function extracts patches using conv2d kernels and creates 2 new dimensions to store them
    # yunqi --- the extracted patches have shape (K, K, B*(P*P+1))
    # yunqi --- there are totally (b, H', W') of patches
        """
            Shape:
                Input:     (b, H, W, B*(P*P+1))
                Output:    (b, H', W', K, K, B*(P*P+1))
                yunqi ----  H' and W'      are number of patches in these 2 dims (height and width), 
                            K, K           is the size of patches extracted from the input
        """
        b, h, w, c = x.shape
        assert h == w
        # assert c == B*(psize+1)
        oh = ow = int((h - K) / stride + 1)
        idxs = [[(h_idx + k_idx)                 for k_idx in range(0, K)]                 for h_idx in range(0, h - K + 1, stride)]
        x = x[:, idxs, :, :]
        x = x[:, :, :, idxs, :]
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x, oh, ow

    def transform_view(self, x, w, C, P, w_shared=False):
    # yunqi --- this function transforms pose of low-caps to votes
    # yunqi --- no matter shared weights or not, the number of transformation matrix is always K*K*B 
    # yunqi --- the difference is that for class_caps, the number of low-caps is H*W*B
    # yunqi ---                    and for conv_caps,  the number of low-caps is K*K*B
        """
                w:         (1, K*K*B, C, P, P)
            For conv_caps:
                Input:     (b*H*W, K*K*B, P*P)
                Output:    (b*H*W, K*K*B, C, P*P)
            For class_caps:
                Input:     (b, H*W*B, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        b, B, psize = x.shape
        assert psize == P*P

        x = x.view(b, B, 1, P, P)
        
        # yunqi --- for conv_caps, "B"=K*K*B, w.size(1)=K*K*B, so hw=1
        # yunqi --- for class_caps, "B"=H*W*B, w.size(1)=1*1*B (K=1), so hw=H*W, for every 
        if w_shared:
            hw = int(B / w.size(1))
            w = w.repeat(1, hw, 1, 1, 1)

        w = w.repeat(b, 1, 1, 1, 1)
        x = x.repeat(1, 1, C, 1, 1)
        v = torch.matmul(x, w)
        v = v.view(b, B, C, P*P)
        return v

    def add_coord(self, v, b, h, w, B, C, psize):
        """
            Shape:
                Input:     (b, H*W*B, C, P*P)
                Output:    (b, H*W*B, C, P*P)
        """
        assert h == w
        v = v.view(b, h, w, B, C, psize)
        coor = torch.arange(h, dtype=torch.float32) / h
        if cuda:
            coor_h = Variable(torch.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.), requires_grad=False).to(device)
            coor_w = Variable(torch.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.), requires_grad=False).to(device)
        else:
            coor_h = torch.FloatTensor(1, h, 1, 1, 1, self.psize).fill_(0.)
            coor_w = torch.FloatTensor(1, 1, w, 1, 1, self.psize).fill_(0.)
        coor_h[0, :, 0, 0, 0, 0] = coor
        coor_w[0, 0, :, 0, 0, 1] = coor
        v = v + coor_h + coor_w
        v = v.view(b, h*w*B, C, psize)
        return v

    def forward(self, x, wide=torch.zeros(1).to(device)):
        self.wide = wide
        b, h, w, c = x.shape
        
        # yunqi --- for weights shared, K has to be 1
        # yunqi --- what every w_shared is True or False, weights are shared over batches and number of patches
        # yunqi --- thus the transformation matrix is always between K*K*B low-caps and C high-caps
        
        if not self.w_shared:
            # add patches
            x, oh, ow = self.add_pathes(x, self.B, self.K, self.psize, self.stride)
            if self.wide.numel()!=1: # (wide!= None):
                wide, oh, ow = self.add_pathes(wide, self.B, self.K, self.psize, self.stride)
            # yunqi --- this function extracts patches using conv2d kernels and creates 2 new dimensions to store them

            # transform view
            p_in = x[:, :, :, :, :, :self.B*self.psize].contiguous()
            a_in = x[:, :, :, :, :, self.B*self.psize:].contiguous()
            
            self.pose_in = p_in.view(b*oh*ow, self.K*self.K*self.B, self.psize)
            if self.wide.numel()!=1: # (wide!= None):
                self.wide_in = wide.view(b*oh*ow, self.K*self.K*self.B, self.psize)

            # yunqi --- oh and ow are number of patches extracted from input, 
            # self.K*self.K is the size of the patch
            
            self.a_in = a_in.view(b*oh*ow, self.K*self.K*self.B, 1)
            
            # change of information between capsule and wideresnet
            if self.wide.numel()!=1: # (wide!= None):
                pose_in_attn,_ = self.MHAttn(self.wide_in, self.pose_in, self.pose_in, self.a_in)
                # pose_in_attn = self.att_linear(pose_in_attn)
                # pose_in_attn,_ = self.MHAttn(self.wide_in, pose_in_attn, pose_in_attn, self.a_in)
                pose_in_attn = self.att_linear1(pose_in_attn)
                # self.pose_in_attn,_ = self.MHAttn(pose_in_attn, pose_in_attn, pose_in_attn, self.a_in)
                self.pose_in_attn = pose_in_attn # + self.pose_in
            else:
                pose_in_attn,_ = self.MHAttn(self.pose_in, self.pose_in, self.pose_in, self.a_in)
                self.pose_in_attn,_ = self.MHAttn(pose_in_attn, pose_in_attn, pose_in_attn, self.a_in)

            v = self.transform_view(self.pose_in_attn, self.weights, self.C, self.P)
#             v = self.transform_view(self.pose_in, self.weights, self.C, self.P)
            # yunqi --- this function transforms pose of low-caps to votes
            self.vote = v

            # em_routing
#             p_out, a_out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out, a_out, r, p_zx, self.mu, self.sigma_sq = self.caps_em_routing(v, self.a_in, self.C, self.eps)
            
            self.pose_out = p_out
            self.a_out = a_out
            self.p_xz = r
            self.p_zx = p_zx
            
            p_out = p_out.view(b, oh, ow, self.C*self.psize)
            a_out = a_out.view(b, oh, ow, self.C)
            self.pose_out = p_out
            self.a_out = a_out
            
            out = torch.cat([p_out, a_out], dim=3)
        else:
            # assert c == self.B*(self.psize+1)
            assert 1 == self.K
            assert 1 == self.stride
            p_in = x[:, :, :, :self.B*self.psize].contiguous()
            self.pose_in = p_in.view(b, h*w*self.B, self.psize)
            
            a_in = x[:, :, :, self.B*self.psize:].contiguous()
            self.a_in = a_in.view(b, h*w*self.B, 1)
            if self.wide.numel()!=1: # (wide!= None):
                self.wide_in = wide.view(b, h*w*self.B, self.psize)

            if self.wide.numel()!=1: # (wide!= None):
                pose_in_attn,_ = self.MHAttn(self.wide_in, self.pose_in, self.pose_in, self.a_in)
                # pose_in_attn,_ = self.MHAttn(self.wide_in, pose_in_attn, pose_in_attn, self.a_in)
                self.pose_in_attn,_ = self.MHAttn(pose_in_attn, pose_in_attn, pose_in_attn, self.a_in)
            else:
                pose_in_attn,_ = self.MHAttn(self.pose_in, self.pose_in, self.pose_in, self.a_in)
                self.pose_in_attn,_ = self.MHAttn(pose_in_attn, pose_in_attn, pose_in_attn, self.a_in)
                
            # transform view
            v = self.transform_view(self.pose_in_attn, self.weights, self.C, self.P, self.w_shared)
#             v = self.transform_view(self.pose_in, self.weights, self.C, self.P, self.w_shared)
            
            self.vote = v

            # coor_add
            if self.coor_add:
                v = self.add_coord(v, b, h, w, self.B, self.C, self.psize)
                
            # em_routing
#             p_out, out = self.caps_em_routing(v, a_in, self.C, self.eps)
            p_out, out, r, p_zx, self.mu, self.sigma_sq = self.caps_em_routing(v, self.a_in, self.C, self.eps)
            
            self.pose_out = p_out
            self.a_out = out
            self.p_xz = r
            self.p_zx = p_zx

        return out #, self.pose_out, self.a_out



    

class CapsNet(nn.Module):
    def __init__(self, depth, widen_factor=1, num_class=10, K=3, P=4, iters=3, datachan=1, dropRate=0.4):
        super(CapsNet, self).__init__()
        self.dim = (1,1,1,1)
        block = BasicBlock
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        #nChannels = [16, 32*widen_factor, 16*widen_factor, 16*widen_factor]
        #nChannels = [64, 512, 256, 256]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        self.conv1 = nn.Conv2d(datachan, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.in_planes = 64
        print('| Wide-Resnet %dx%d' %(depth, widen_factor))
        
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        self.primary_caps = PrimaryCaps(nChannels[1], int(nChannels[1]/P/P), 1, P, stride=1)
        
        self.conv_caps1 = ConvCaps(int(nChannels[1]/P/P),  int(nChannels[2]/P/P), K, P, stride=2, iters=iters, attn_dim=16)
        self.conv_caps2 = ConvCaps(int(nChannels[2]/P/P),  int(nChannels[3]/P/P), K, P, stride=2, iters=iters, attn_dim=16)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3],num_class)
        self.class_caps = ConvCaps(int(nChannels[3]/P/P), num_class, 1, P, stride=1, iters=iters,
                                        coor_add=True, w_shared=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, netTYPE='RESsolo'):
        if netTYPE=='incepCAP':
            self.preout = self.conv1(x)
            self.b1out = self.block1(self.preout)
            
            self.pcout, self.pcpout, self.pcaout = self.primary_caps(self.b1out)
            
            self.b2out = self.block2(self.b1out)
            self.b3out = self.block3(self.b2out)
            
            self.cc1out = self.conv_caps1(self.pcout, self.b1out.permute(0,2,3,1).contiguous())
            self.cc1per = F.pad(self.cc1out.permute(0,3,1,2).contiguous(),[1,0,1,0]\
                              ,"constant",value=0).permute(0,2,3,1).contiguous()
            
            self.cc2out = self.conv_caps2(self.cc1per, self.b2out.permute(0,2,3,1).contiguous())
            self.cc2per = F.pad(self.cc2out.permute(0,3,1,2).contiguous(),[1,0,1,0]\
                              ,"constant",value=0).permute(0,2,3,1).contiguous()
            
            self.classout = self.class_caps(self.cc2per, self.b3out.permute(0,2,3,1).contiguous())
            return self.classout, self.class_caps.pose_out, self.class_caps.mu, self.class_caps.sigma_sq, self.class_caps.vote

        # ------------------------------------------------------------------------------------------------------
        elif netTYPE=='CAPsolo':
            self.preout = self.conv1(x)
            self.b1out = self.block1(self.preout)
            
            self.pcout, self.pcpout, self.pcaout = self.primary_caps(self.b1out)
            
            self.cc1out = self.conv_caps1(self.pcout, self.b1out.permute(0,2,3,1).contiguous())
            self.cc1per = F.pad(self.cc1out.permute(0,3,1,2).contiguous(),[1,0,1,0]\
                              ,"constant",value=0).permute(0,2,3,1).contiguous()
            
            self.cc2out = self.conv_caps2(self.cc1per)
            self.cc2per = F.pad(self.cc2out.permute(0,3,1,2).contiguous(),[1,0,1,0]\
                              ,"constant",value=0).permute(0,2,3,1).contiguous()
            
            self.classout = self.class_caps(self.cc2per)
            return self.classout, self.class_caps.pose_out, self.class_caps.mu, self.class_caps.sigma_sq, self.class_caps.vote
        # -----------------------------------------------------------------------------------
        elif netTYPE=='RESsolo':
            self.preout = self.conv1(x)
            
            out = self.block1(self.preout)
            out = self.block2(out)
            out = self.block3(out)
            out = self.relu(self.bn1(out))
            out = F.avg_pool2d(out, out.shape[2])
            out = out.flatten(1)
            self.classout = self.fc(out)
        
        return self.classout
    
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)
    
    
# ################################################################################################ #
# ------------------------------------------------------------------------------------------------ #
# ################################################################################################ #


class CapsNetv2(nn.Module):
    def __init__(self, depth, widen_factor=1, num_class=10, K=3, P=4, iters=3, datachan=1, dropRate=0.3, capslayer=2):
        super(CapsNetv2, self).__init__()
        self.capslayer = capslayer
        self.dim = (1,1,1,1)
        block = BasicBlock
        nChannels = [64, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        #nChannels = [16, 64*widen_factor, 32*widen_factor, 16*widen_factor]

        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        self.conv1 = nn.Conv2d(datachan, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.in_planes = 16
        print('| Wide-Resnet %dx%d' %(depth, widen_factor))
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        
        self.primary_caps = PrimaryCaps(nChannels[1], int(nChannels[1]/P/P), 1, P, stride=1)
        self.conv_caps1 = ConvCaps(int(nChannels[1]/P/P),  int(nChannels[2]/P/P), K, P, stride=2, iters=iters, attn_dim=16)
        self.conv_caps2 = ConvCaps(int(nChannels[2]/P/P),  int(nChannels[3]/P/P), K, P, stride=2, iters=iters, attn_dim=16)
        
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        if capslayer >= 4:
            self.conv_caps3 = ConvCaps(int(nChannels[3]/P/P),  int(nChannels[3]/P/P), \
                                        K, P, stride=1, iters=iters, attn_dim=16)
            self.conv_caps4 = ConvCaps(int(nChannels[3]/P/P),  int(nChannels[3]/P/P), \
                                        K, P, stride=1, iters=iters, attn_dim=16)
            # 2-1 block
            self.block4 = NetworkBlock(n, nChannels[3], nChannels[3], block, 1, dropRate)
            # 2-2 block
            self.block5 = NetworkBlock(n, nChannels[3], nChannels[3], block, 1, dropRate)
        if capslayer == 6:
            self.conv_caps5 = ConvCaps(int(nChannels[3]/P/P),  int(nChannels[3]/P/P), \
                                        K, P, stride=1, iters=iters, attn_dim=16)
            self.conv_caps6 = ConvCaps(int(nChannels[3]/P/P),  int(nChannels[3]/P/P), \
                                        K, P, stride=1, iters=iters, attn_dim=16)
            # 2-3 block
            self.block6 = NetworkBlock(n, nChannels[3], nChannels[3], block, 1, dropRate)
            # 2-4 block
            self.block7 = NetworkBlock(n, nChannels[3], nChannels[3], block, 1, dropRate)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3],num_class)
        self.class_caps = ConvCaps(int(nChannels[3]/P/P), num_class, 1, P, stride=1, iters=iters,
                                        coor_add=True, w_shared=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, netTYPE='incepCAP'):
        if netTYPE=='incepCAP':
            self.preout = self.conv1(x)
            self.b1out = self.block1(self.preout)
#             print(self.b1out.shape)
            
            self.pcout, self.pcpout, self.pcaout = self.primary_caps(self.b1out)
#             print('pcout')
#             print(self.pcout.shape)
            
            self.cc1out = self.conv_caps1(self.pcout, self.b1out.permute(0,2,3,1).contiguous())
            self.cc1per = F.pad(self.cc1out.permute(0,3,1,2).contiguous(),[1,0,1,0]\
                              ,"constant",value=0).permute(0,2,3,1).contiguous()
#            print('cc1out')
#            print(self.cc1out.shape)
#            print('cc1per, vote')
#            print(self.cc1per.shape)
#            print(self.conv_caps1.vote.shape)
            
            self.b2out = self.block2(self.b1out)
#             print(self.b2out.shape)
            self.cc2out = self.conv_caps2(self.cc1per, self.b2out.permute(0,2,3,1).contiguous())
            self.cc2per = F.pad(self.cc2out.permute(0,3,1,2).contiguous(),[1,0,1,0]\
                              ,"constant",value=0).permute(0,2,3,1).contiguous()
#             print('cc2out')
#             print(self.cc2out.shape)
#            print('cc2per')
#            print(self.cc2per.shape)
#            print(self.conv_caps2.vote.shape)
            
            self.b3out = self.block3(self.b2out)
#             print('b3out')
#             print(self.b3out.shape)
            
            if self.capslayer >= 4:
                self.cc3out = self.conv_caps3(self.cc2per, self.b3out.permute(0,2,3,1).contiguous())
                self.cc3per = F.pad(self.cc3out.permute(0,3,1,2).contiguous(),[1,1,1,1]\
                                    ,"constant",value=0).permute(0,2,3,1).contiguous()
#                 print('cc3out')
#                 print(self.cc3out.shape)
#                 print('cc3per')
#                 print(self.cc3per.shape)
                
                self.b4out = self.block4(self.b3out)
                self.cc4out = self.conv_caps4(self.cc3per, self.b4out.permute(0,2,3,1).contiguous())
                self.cc4per = F.pad(self.cc3out.permute(0,3,1,2).contiguous(),[1,1,1,1]\
                                    ,"constant",value=0).permute(0,2,3,1).contiguous()
#                 print('cc4out')
#                 print(self.cc4out.shape)
#                 print('cc4per')
#                 print(self.cc4per.shape)
                
                self.b5out = self.block5(self.b4out)
                
            if self.capslayer == 6:
                self.cc5out = self.conv_caps5(self.cc4per, self.b5out.permute(0,2,3,1).contiguous())
                self.cc5per = F.pad(self.cc5out.permute(0,3,1,2).contiguous(),[1,1,1,1]\
                                    ,"constant",value=0).permute(0,2,3,1).contiguous()
#                 print('cc5out')
#                 print(self.cc5out.shape)
#                 print('cc5per')
#                 print(self.cc5per.shape)
                
                self.b6out = self.block6(self.b5out)
                self.cc6out = self.conv_caps6(self.cc5per, self.b6out.permute(0,2,3,1).contiguous())
                self.cc6per = F.pad(self.cc5out.permute(0,3,1,2).contiguous(),[1,1,1,1]\
                                    ,"constant",value=0).permute(0,2,3,1).contiguous()
#                 print('cc6out')
#                 print(self.cc6out.shape)
#                 print('cc6per')
#                 print(self.cc6per.shape)
                
                self.b7out = self.block7(self.b6out)
            
            if self.capslayer == 2:
                self.classout = self.class_caps(self.cc2per, self.b3out.permute(0,2,3,1).contiguous())
#                print('=========== classcaps, activation, pose, vote ==============')
#                print(self.classout.shape)
#                print(self.class_caps.pose_out.shape)
#                print(self.class_caps.vote.shape)
            elif self.capslayer == 4:
                self.classout = self.class_caps(self.cc4per, self.b5out.permute(0,2,3,1).contiguous())
            elif self.capslayer == 6:
                self.classout = self.class_caps(self.cc6per, self.b7out.permute(0,2,3,1).contiguous())
            return self.classout, self.class_caps.mu, self.class_caps.sigma_sq, self.class_caps.vote

        # ------------------------------------------------------------------------------------------------------
        elif netTYPE=='CAPsolo':
            self.preout = self.conv1(x)
            self.b1out = self.block1(self.preout)
            
            self.pcout, self.pcpout, self.pcaout = self.primary_caps(self.b1out)
            
            self.cc1out = self.conv_caps1(self.pcout)
            self.cc1per = F.pad(self.cc1out.permute(0,3,1,2).contiguous(),[1,0,1,0]\
                              ,"constant",value=0).permute(0,2,3,1).contiguous()
            
            self.cc2out = self.conv_caps2(self.cc1per)
            self.cc2per = F.pad(self.cc2out.permute(0,3,1,2).contiguous(),[1,0,1,0]\
                              ,"constant",value=0).permute(0,2,3,1).contiguous()
            
            if self.capslayer >= 4:
                self.cc3out = self.conv_caps3(self.cc2per)
                self.cc3per = F.pad(self.cc3out.permute(0,3,1,2).contiguous(),[1,1,1,1]\
                                    ,"constant",value=0).permute(0,2,3,1).contiguous()
                
                #self.b4out = self.block4(self.b3out)
                self.cc4out = self.conv_caps4(self.cc3per)
                self.cc4per = F.pad(self.cc3out.permute(0,3,1,2).contiguous(),[1,1,1,1]\
                                    ,"constant",value=0).permute(0,2,3,1).contiguous()
            
            if self.capslayer == 6:
                self.cc5out = self.conv_caps5(self.cc4per)
                self.cc5per = F.pad(self.cc5out.permute(0,3,1,2).contiguous(),[1,1,1,1]\
                                    ,"constant",value=0).permute(0,2,3,1).contiguous()
                
                #self.b6out = self.block6(self.b5out)
                self.cc6out = self.conv_caps6(self.cc5per)
                self.cc6per = F.pad(self.cc5out.permute(0,3,1,2).contiguous(),[1,1,1,1]\
                                    ,"constant",value=0).permute(0,2,3,1).contiguous()
                
            if self.capslayer == 2:
                self.classout = self.class_caps(self.cc2per)
            elif self.capslayer == 4:
                self.classout = self.class_caps(self.cc4per)
            elif self.capslayer == 6:
                self.classout = self.class_caps(self.cc6per)
            return self.classout, self.class_caps.mu, self.class_caps.sigma_sq, self.class_caps.vote
            
        # -----------------------------------------------------------------------------------
        elif netTYPE=='RESsolo':
            self.preout = self.conv1(x)
            
#             self.b1out = self.layer1(self.preout)
#             self.b2out = self.layer2(self.b1out)
#             self.b3out = self.layer3(self.b2out)
            self.b1out = self.block1(self.preout)
            self.b2out = self.block2(self.b1out)
            self.b3out = self.block3(self.b2out)
            if self.capslayer == 2:
                self.bfinalout = self.b3out
            elif self.capslayer == 4:
                self.b4out = self.block4(self.b3out)
                self.bfinalout = self.block5(self.b4out)
            elif self.capslayer == 6:
                self.b4out = self.block4(self.b3out)
                self.b5out = self.block5(self.b4out)
                self.b6out = self.block6(self.b5out)
                self.bfinalout = self.block7(self.b6out)
                
            out = self.relu(self.bn1(self.bfinalout))
            out = F.avg_pool2d(out, 8)
            self.feature = out.flatten(1)
            self.classout = self.fc(self.feature)
        
        return self.classout, self.classout, self.classout, self.classout
    
    
    
    
class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=10):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target, r):
        b, E = x.shape
        assert E == self.num_class
        margin = self.m_min + (self.m_max - self.m_min)*r

        if cuda:
            at = Variable(torch.FloatTensor(b).fill_(0), requires_grad=False).to(device)
        else:
            at = torch.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
        at = at.view(b, 1).repeat(1, E)

        zeros = x.new_zeros(x.shape)
        loss = torch.max(margin - (at - x), zeros)
        loss = loss**2
        loss = loss.sum() / b - margin**2

        return loss
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
