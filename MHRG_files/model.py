import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
import torchio as tio
import albumentations as A
from torch.nn import init
import lightning as L

def load_model(model, model_path):
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                # Added for maintaining same initialization for weights
                # seed = 3
                # np.random.seed(seed)
                # torch.manual_seed(seed)
                # torch.cuda.manual_seed(seed)
                # torch.backends.cudnn.deterministic = True
                # torch.backends.cudnn.benchmark = False

                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        """
        elif classname.find('BatchNorm3d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        """
    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def configure_optimizers(parameters, weight_decay, learning_rate):
    # start with all of the candidate parameters (that require grad)
    all_param = [p for p in parameters if p.requires_grad]

    decay_params = [p for p in all_param if p.dim() >= 2]
    nodecay_params = [p for p in all_param if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
    return optimizer

#-------------- UnetGenerator3d --------------#
class UnetGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a 3D Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator3d, self).__init__()
        # construct unet structure
        # print('at generating model')
        # GPUtil.showUtilization()
        unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer

        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock3d(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock3d(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # print(unet_block)
        unet_block = UnetSkipConnectionBlock3d(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # print(unet_block)
        unet_block = UnetSkipConnectionBlock3d(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        # print(unet_block)
        self.model = UnetSkipConnectionBlock3d(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        # print(self.model)
        return self.model(input)

class UnetSkipConnectionBlock3d(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock3d, self).__init__()

        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        
        # Down sampling part    
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        # Down sampling part
        
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        interp = 'trilinear'  #'nearest'
        transp_conv = False # Use transposed convolution or resize convolution?
        #transp_conv = True

        if outermost:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1)
                upconv = [upsamp, conv]

            down = [downconv]
            up = [uprelu, *upconv, nn.ReLU()]
            
            model = down + [submodule] + up
        elif innermost:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
                upconv = [upsamp, conv]

            down = [downrelu, downconv]
            up = [uprelu, *upconv, upnorm]
            model = down + up
        else:
            if transp_conv is True:
                upconv = [nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)]
            else:
                upsamp = nn.Upsample(scale_factor=2, mode=interp)
                conv = nn.Conv3d(inner_nc * 2, outer_nc, kernel_size=3, stride=1, padding=1, bias=use_bias)
                upconv = [upsamp, conv]
            down = [downrelu, downconv, downnorm]
            up = [uprelu, *upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        if self.outermost:
            print(x.size())
            return self.model(x)
        else:   # add skip connections
            print(x.size())
            # a = self.model(x)
            # print(a.size())
            return torch.cat([x, self.model(x)], 1)

#-------------- Standard 3D Unet --------------#

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.IN = nn.InstanceNorm3d(out_ch)
        #self.BN = nn.BatchNorm3d(out_ch)
        #block = [conv1, relu, BN, conv2, relu, BN]
        #self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.IN(self.relu(self.conv2(self.IN(self.relu(self.conv1(x))))))
        #return self.BN(self.relu(self.conv2(self.BN(self.relu(self.conv1(x))))))
        #return self.model(x)

class Block_dropout(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.IN = nn.InstanceNorm3d(out_ch)
        # Dropout3d laye
        dropout_prob = 0.4
        self.dropout = nn.Dropout3d(p=dropout_prob)
        #self.BN = nn.BatchNorm3d(out_ch)
        #block = [conv1, relu, BN, conv2, relu, BN]
        #self.model = nn.Sequential(*block)

    def forward(self, x):
        return self.IN(self.relu(self.conv2(self.IN(self.dropout(self.relu(self.conv1(x)))))))
        #return self.BN(self.relu(self.conv2(self.BN(self.relu(self.conv1(x))))))
        #return self.model(x)
        
class Encoder(nn.Module):
    def __init__(self, in_ch, chs=(32, 64, 128, 256, 512)):
        super().__init__()
        chs = (in_ch,) + chs
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 2)]+
                                         [Block_dropout(chs[len(chs) - 2], chs[len(chs) - 1])])
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            # print('Downconv without maxpool:{}'.format(x.size()))
            x = self.pool(x)
            # print('Downconv with maxpool:{}'.format(x.size()))
        return ftrs

class ResizeConv(nn.Module):
    # Resize (upsample) with nearest neighbor or trilinear interpolation and then do convolution
    def __init__(self, in_ch, out_ch, interp='nearest'):
        super().__init__()
        self.upsamp = nn.Upsample(scale_factor=2, mode=interp)
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.upsamp(x)
        x = self.conv(x)
        # print('Upconv after upsampling and convolution{}:'.format(x.size()))
        return x

class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.chs = chs
        #self.upconvs = nn.ModuleList([nn.ConvTranspose3d(chs[i], chs[i + 1], 2, 2) for i in range(len(chs) - 1)])
        ## customize for latent
        chs_upconvs = list(chs) 
        #chs_upconvs[0] = 259
        chs_upconvs[0] = 515
        #chs_upconvs[0] = 512
        chs_upconvs = tuple(chs_upconvs)
        self.upconvs = nn.ModuleList([ResizeConv(chs_upconvs[i], chs_upconvs[i + 1], interp='trilinear') for i in range(len(chs) - 1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.chs) - 1):
            x = self.upconvs[i](x)
            # print(x.size())
            # print(encoder_features[i].size())
            #enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, encoder_features[i]], dim=1)
            x = self.dec_blocks[i](x)
        return x

    # def crop(self, enc_ftrs, x):
    #     _, _, H, W = x.shape
    #     enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
    #     return enc_ftrs

class UNet3D(nn.Module):
    def __init__(self, in_ch=7, enc_chs=(32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32), out_ch=1):
        super().__init__()
        self.encoder = Encoder(in_ch, enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv3d(dec_chs[-1], out_ch, 1)

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        # import time
        #
        # print('length of encoding ftrs {}'.format(len(enc_ftrs)))
        # a = enc_ftrs[::-1][0]
        # print(a.size())
        # b = enc_ftrs[::-1][1]
        # print(b.size())
        # time.sleep(10)
        out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        #activation = nn.Sigmoid()
        activation = nn.ReLU()
        out = activation(out)
        return out

class UNet3D2(nn.Module):
    def __init__(self, in_ch=7, enc_chs=(32, 64, 128, 256, 512), dec_chs=(512, 256, 128, 64, 32), out_ch=1):
    #def __init__(self, in_ch=7, enc_chs=(16, 32, 64, 128, 256), dec_chs=(256, 128, 64, 32, 16), out_ch=1):        
        super().__init__()
        self.in_ch = in_ch
        self.encoder = Encoder(in_ch, enc_chs)
        self.decoder = Decoder(dec_chs)
        self.head = nn.Conv3d(dec_chs[-1], out_ch, 1)
        bottleneck_dim = 2
        input_dim = 3
        self.latent_decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, input_dim))
            
    def forward(self, x, latent):
        enc_ftrs = self.encoder(x[:,:self.in_ch])
        # import time
        #
        # print('length of encoding ftrs {}'.format(len(enc_ftrs)))
        # a = enc_ftrs[::-1][0]
        # print(a.size())
        # b = enc_ftrs[::-1][1]
        # print(b.size())
        # time.sleep(10)
        out1 = self.latent_decoder(latent)
        out1_expanded_dims = out1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) # Shape becomes (1, 5, 1, 1)
        out1_broadcasted = out1_expanded_dims.expand(-1, -1, 8, 8, 8)
        out2 = torch.cat((enc_ftrs[::-1][0], out1_broadcasted), dim=1)
        out = self.decoder(out2, enc_ftrs[::-1][1:])
        #out = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out = self.head(out)
        #activation = nn.Sigmoid()
        activation = nn.ReLU()
        out = activation(out)
        return out
        
#-------------- Attn 3D Unet --------------#
#https://github.com/ozan-oktay/Attention-Gated-Networks/tree/master

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)

class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple): self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list): self.sub_sample_factor = tuple(sub_sample_factor)
        else: self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(3, in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(3,in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0, bias=False)
        self.phi = conv_nd(3, in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(3, in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        # Initialise weights
        for m in self.children():
            init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')


    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = F.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2,2,2)):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   )

class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1

class Attn_UNet3D(nn.Module):

    def __init__(self, feature_scale=4, out_ch=21, is_deconv=True, in_ch=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2,2), is_batchnorm=True):
        super(Attn_UNet3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_ch
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.out_ch = out_ch

        #filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        #filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.gating = UnetGridGatingSignal3(filters[4], filters[4], kernel_size=(1, 1, 1), is_batchnorm=self.is_batchnorm)

        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=self.out_ch, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=self.out_ch, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=self.out_ch, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=self.out_ch, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv3d(self.out_ch*4, self.out_ch, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        return final

class LightWrapper(L.LightningModule):
    def __init__(self, module, loss_function, weight_decay,lr_rate):
        super().__init__()
        self.module = module(in_ch=7, out_ch=1)
        init_weights(self.module, init_type='xavier', init_gain=1) 
        self.save_hyperparameters(loss_function.selected_losses) 
        self.train_losses = []
        self.loss_function = loss_function
        self.weight_decay = weight_decay
        self.lr_rate = lr_rate


    def training_step(self, batch, batch_idx):
        #self.lr_schedulers().step()
        x, y, hist, bins = batch
        predicted_dose = self.module(x)
        loss = self.loss_function(x, predicted_dose, y.unsqueeze(1), hist, bins)
        self.log("train_loss", loss, prog_bar=False, on_step=True, on_epoch=True, logger=True)
        #self.log("lr_rate", self.lr_schedulers().get_last_lr()[0], prog_bar=False, on_step=True, on_epoch=False, logger=True)
        self.train_losses.append(loss.detach().cpu())
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y, hist, bins = batch
        predicted_dose = self.module(x)
        loss = self.loss_function(x, predicted_dose, y.unsqueeze(1), hist, bins)
        self.log("val_loss", loss, prog_bar=False, on_step=True, on_epoch=True, logger=True)
        return loss
    

    def configure_optimizers(self):
        optimizer = configure_optimizers(self.module.parameters(), self.weight_decay, self.lr_rate)
        #scheduler = LinearWarmupCosineAnnealingLR(optimizer, self.config.step_per_epoch, self.config.epochs, self.config.lr_rate, self.config.warmup_steps)
        return [optimizer]


#------------- Eval -------------------------#
def compute_eval_metrics(x, y, prediction, reduce = True):
    # Return value for each sample in batch  
    # Dose score:
    if reduce:
        dose_score = torch.nn.functional.l1_loss(prediction, y.unsqueeze(1),reduction="mean")
    else:
        dose_score = torch.nn.functional.l1_loss(prediction, y.unsqueeze(1),reduction="none").view(prediction.size(0), -1).mean(1)
    
    # DVH score starting with OAR 
    oar_pred_dose = x[:,1:7] * prediction
    oar_real_dose = x[:,1:7] * y.unsqueeze(1)
    pread_oar_mean = oar_pred_dose.mean((2,3,4)) 
    real_oar_mean = oar_real_dose.mean((2,3,4))
    pred_oar_max = torch.amax(oar_pred_dose, dim=(2,3,4))
    real_oar_max = torch.amax(oar_real_dose, dim=(2,3,4))
    
    # DVH score for PTV
    ptv_pred = (x[:,6].unsqueeze(1) * prediction).view(x.shape[0],-1)
    ptv_real = (x[:,6].unsqueeze(1) * y.unsqueeze(1)).view(x.shape[0],-1)
    
    real_D1 = torch.quantile(ptv_pred, 0.99, dim=1, keepdim=True)
    pred_D1 = torch.quantile(ptv_real, 0.99, dim=1, keepdim=True)

    real_D95 = torch.quantile(ptv_pred, 0.05, dim=1, keepdim=True)
    pred_D95 = torch.quantile(ptv_real, 0.05, dim=1, keepdim=True)

    real_D99 = torch.quantile(ptv_pred, 0.01, dim=1, keepdim=True)
    pred_D99 = torch.quantile(ptv_real, 0.01, dim=1, keepdim=True)


    final_DVH_scores_pred = torch.cat((pread_oar_mean, pred_oar_max, pred_D1, pred_D95, pred_D99), dim=1)
    final_DVH_scores_real = torch.cat((real_oar_mean, real_oar_max, real_D1, real_D95, real_D99), dim=1)

    if reduce:
        DVH_score = torch.nn.functional.l1_loss(final_DVH_scores_pred, final_DVH_scores_real,reduction="mean") 
    else:
        DVH_score = torch.nn.functional.l1_loss(final_DVH_scores_pred, final_DVH_scores_real,reduction="none").view(final_DVH_scores_pred.size(0), -1).mean(1)

    return dose_score, DVH_score

#-------------- Loss functions --------------#

class FinalLoss(torch.nn.Module):
    def __init__(self, selected_losses):
        super().__init__()
        MAE_loss = torch.nn.L1Loss()  # MAE_loss(predicted_dose, y.unsqueeze(1))
        MSE_loss = torch.nn.MSELoss()  # MSE_loss(predicted_dose, y.unsqueeze(1))
        DVH_loss = DVHLoss()  # DVH_loss(x, predicted_dose, y.unsqueeze(1), target_hist, target_bins)
        Moment_loss = MomentLoss()  # Moment_loss(x, predicted_dose, y.unsqueeze(1), target_hist, target_bins)

        self.available_losses = {
            "MAE": MAE_loss,
            "MSE": MSE_loss,
            "DVH": DVH_loss,
            "Moment": Moment_loss
        }

        self.selected_losses = selected_losses

    def forward(self, x, predicted_dose, dose, target_hist=None, target_bins=None):
        total_loss = 0
        # Loop through each loss type and calculate weighted loss
        for loss_name, weight in self.selected_losses.items():
            loss_func = self.available_losses[loss_name]
            
            if loss_name == "MAE" or loss_name == "MSE":
                loss_value = loss_func(predicted_dose, dose)
            elif loss_name == "DVH" or loss_name == "Moment":
                loss_value = loss_func(x, predicted_dose, dose, target_hist, target_bins)
                
            total_loss += weight * loss_value
        return total_loss
    
class DVHLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, x, predicted_dose, dose, target_hist, target_bins):
        oar = x[:,1:7]
        vols = torch.sum(oar, axis=(2, 3, 4))
        n_bins = target_bins.shape[1]
        hist = torch.zeros_like(target_hist)
        bin_w = target_bins[0,1] - target_bins[0,0]

        for i in range(n_bins):
            diff = torch.sigmoid((predicted_dose - target_bins[:,i].view(target_bins.shape[0], 1, 1, 1, 1)) / bin_w)
            diff = diff.repeat(1, oar.shape[1], 1, 1, 1) * oar
            num = torch.sum(diff, axis=(2, 3, 4))
            hist[:,i] = (num / vols)

        return self.loss(hist, target_hist)

# Defines Bhattacharya loss class
class BhattLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, x, predicted_dose, dose, target_hist, target_bins):
        oar = x[:,1:7]
        vols = torch.sum(oar, axis=(2, 3, 4))
        n_bins = target_bins.shape[1]
        hist = torch.zeros_like(target_hist)
        bin_w = target_bins[0,1] - target_bins[0,0]

        #print(vols.shape, hist.shape)

        for i in range(n_bins):
            diff = torch.sigmoid((predicted_dose - target_bins[:,i].view(target_bins.shape[0], 1, 1, 1, 1)) / bin_w)
            diff = diff.repeat(1, oar.shape[1], 1, 1, 1) * oar
            num = torch.sum(diff, axis=(2, 3, 4))
            hist[:,i] = (num / vols)

        #print(hist.detach().cpu().numpy())
        #print(vols.detach().cpu().numpy())
        histprod = torch.sqrt(hist * target_hist)
        #histprod = torch.clamp(histprod, 1e-8)
        #print(histprod.detach().cpu().numpy())
        bhattdist = torch.sum(histprod, axis=(1,2))  # Sum of bhattacharya distances for each OAR
        bhattdist = torch.clamp(bhattdist, 1e-3)
        #print(bhattdist.detach().cpu().numpy())
        bhattloss = torch.mean(-torch.log(bhattdist))
        return bhattloss

##Moment loss
class MomentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def __call__(self, x, predicted_dose, dose, target_hist, target_bins):
        # Calculate predicted hist
        oar = x[:,1:7]
        vols = torch.sum(oar, axis=(2, 3, 4))
        # n_bins = target_bins.shape[1]
        # hist = torch.zeros_like(target_hist)
        # bin_w = target_bins[0,1] - target_bins[0,0]
        #keys = ['Eso', 'Cord', 'Heart', 'Lung_L', 'Lung_R', 'PTV']
        #moment = [[1, 2, 10], [1, 2, 10], [1, 2, 10], [1, 2, 10], [1, 2, 10], [2, 4, 6]]
        #momentOfStructure = dict(zip(keys, moment))
        momentOfStructure = {'Cord': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                'Eso': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                'Heart': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                'Lung_L': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                'Lung_R': {'moments': [1, 2, 10], 'weights': [1, 1, 1]},
                'PTV': {'moments': [2, 4, 6], 'weights': [1, 1, 1]}}
        keys = list(momentOfStructure.keys())
        #momentOfStructure = dict([(k, v) for k in keys for v in values])
        oarPredMoment = []
        oarRealMoment = []
        pres = 60
        epsilon = 0.00001#Added epsilon as the loss function can become sqrt(0)
        for i in range(oar.shape[1]):
            moments = momentOfStructure[keys[i]]['moments']
            weights = momentOfStructure[keys[i]]['weights']
            for j in range(len(moments)):
                gEUDa = moments[j]
                weight = weights[j]
                oarpreddose = predicted_dose*oar[:, i, :, :, :]
                oarRealDose = dose * oar[:, i, :, :, :]
                if i < (oar.shape[1]-1):
                    #oarPredMomenta = torch.pow((1 / vols[0, i]) * (torch.sum(torch.pow(oarpreddose-pres, gEUDa), axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                    #oarRealMomenta = torch.pow((1 / vols[0, i]) * (torch.sum(torch.pow(oarRealDose - pres, gEUDa), axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                    #print(oarPredMomenta)
                    #Use (1 / (oar[:, i, ...] == 1).sum()) to count the number of voxels
                #else:
                    oarPredMomenta = weight*torch.pow((1 / vols[0, i]) * (torch.sum(torch.pow(oarpreddose, gEUDa), axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                    oarRealMomenta = weight*torch.pow((1 / vols[0, i]) * (torch.sum(torch.pow(oarRealDose, gEUDa), axis=(2, 3, 4))) + epsilon, 1 / gEUDa)
                    # if i == oar.shape[1] - 1:
                    #     print(oar[:, i, :, :, :].max())
                    #     print(oarPredMomenta)
                    #     print(oarRealMomenta)
                    #     print(torch.max(oarRealDose))
                    # print(oarPredMomenta, oarRealMomenta)
                    oarPredMoment.append(oarPredMomenta)
                    oarRealMoment.append(oarRealMomenta)
        oarPredMoment = torch.stack(oarPredMoment)
        oarRealMoment = torch.stack(oarRealMoment)

        return self.loss(oarPredMoment, oarRealMoment)