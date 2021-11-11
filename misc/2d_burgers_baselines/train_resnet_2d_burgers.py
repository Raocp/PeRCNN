import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import scipy.io as scio
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.set_default_dtype(torch.float32)

torch.manual_seed(66)
np.random.seed(66)

# model.train() and model.eval()

dx_2d_op  = [[[[    0,   0,  1/12,   0,    0],
               [    0,   0, -8/12,   0,    0],
               [    0,   0,     0,   0,    0],
               [    0,   0,  8/12,   0,    0],
               [    0,   0, -1/12,   0,    0]]]]

dy_2d_op  = [[[[    0,     0,    0,     0,      0],
               [    0,     0,    0,     0,      0],
               [ 1/12, -8/12,    0,  8/12,  -1/12],
               [    0,     0,    0,     0,      0],
               [    0,     0,    0,     0,      0]]]]

lap_2d_op = [[[[    0,   0, -1/12,   0,     0],
               [    0,   0,   4/3,   0,     0],
               [-1/12, 4/3,   - 5, 4/3, -1/12],
               [    0,   0,   4/3,   0,     0],
               [    0,   0, -1/12,   0,     0]]]]

def conv3x3(in_planes: int, out_planes: int, stride = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True, padding_mode='circular')

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

class ResidualBlock(nn.Module):
    def __init__( self, inplanes: int, planes: int, stride: int = 1,  norm_layer = None) -> None:
        super(ResidualBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, 64, stride)
        self.tanh = nn.Tanh()
        self.conv2 = conv3x3(64, 64)
        # self.conv3 = conv3x3(64, 32)

        self.conv3 = conv1x1(64, 2, stride = 1)  # as output layer

        self.conv1.bias.data.fill_(0.0)
        self.conv2.bias.data.fill_(0.0)
        self.conv3.bias.data.fill_(0.0)
        c = 0.5
        self.conv1.weight.data.uniform_(-c * np.sqrt(1 / (2 * 64 * 3 * 3)), c * np.sqrt(1 / (2 * 64 * 3 * 3)))
        self.conv2.weight.data.uniform_(-c * np.sqrt(1 / (64 * 64 * 3 * 3)), c * np.sqrt(1 / (64 * 64 * 3 * 3)))
        self.conv3.weight.data.uniform_(-c * np.sqrt(1 / (64 * 2 * 1 * 1)), c * np.sqrt(1 / (64 * 2 * 1 * 1)))

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.tanh(out)

        out = self.conv2(out)
        out = self.tanh(out)
        out = self.conv3(out)        # 1 by 1 conv to produce 2-channel output

        out = self.tanh(out)

        out = out + identity

        return out


class RecurrentResNet(nn.Module):

    ''' Recurrent ResNet '''

    def __init__(self, input_channels, hidden_channels, output_channels, init_state_low, input_kernel_size,
                       input_stride, input_padding, step=1, effective_step=None):

        super(RecurrentResNet, self).__init__()
        
        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.init_state_low = init_state_low
        self.init_state = []

        name = 'ResBlock'
        cell = ResidualBlock(inplanes = 2, planes = 64)

        setattr(self, name, cell)
        self._all_layers.append(cell)


    def forward(self):
        self.init_state = F.interpolate(self.init_state_low, (100, 100), mode='bicubic')
        internal_state = []
        outputs = [self.init_state]
        second_last_state = []

        for step in range(self.step):
            name = 'ResBlock'
            # all cells are initialized in the first step
            if step == 0:
                h = self.init_state
                internal_state = h

            # forward
            h = internal_state
            # hidden state + output
            h = getattr(self, name)(h)
            o = h
            internal_state = h

            if step == (self.step - 2):
                #  last output is a dummy for central FD
                second_last_state = internal_state.clone()   # save state to enable batch training

            # after many layers output the result save at time step t
            if step in self.effective_step:
                outputs.append(o)

        return outputs, second_last_state


class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=5, name=''):
        '''
        :param DerFilter: constructed derivative filter, e.g. Laplace filter
        :param resol: resolution of the filter, used to divide the output, e.g. c*dt, c*dx or c*dx^2
        :param kernel_size:
        :param name: optional name for the operator
        '''
        super(Conv2dDerivative, self).__init__()
        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
                                1, padding=0, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.tensor(DerFilter, dtype=torch.float32), requires_grad=False)

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class loss_generator(nn.Module):
    ''' Loss generator for physics loss '''

    def __init__(self, dt = (0.00025), dx = (1.0/100)):

        '''
        Construct the derivatives, X = Width, Y = Height      

        '''

        self.dt = dt
        self.dx = dx
       
        super(loss_generator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lap_2d_op,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        # temporal derivative operator
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1, 1, 0]]],
            resol = (dt*1),
            kernel_size = 3,
            name = 'partial_t').cuda()

        # Spatial derivative operator
        self.dx = Conv2dDerivative(
            DerFilter = dx_2d_op,
            resol = (dx),
            kernel_size = 5,
            name = 'dx_operator').cuda()

        # Spatial derivative operator
        self.dy = Conv2dDerivative(
            DerFilter = dy_2d_op,
            resol = (dy),
            kernel_size = 5,
            name = 'dy_operator').cuda()


    def get_phy_Loss(self, output):  # Staggered version
        '''
        Calculate the physics loss

        Args:
        -----
        output: tensor, dim:
            shape: [time, channel, height, width]

        Returns:
        --------
        f_u: float
            physics loss of u

        f_v: float
            physics loss of v
        '''

        # spatial derivatives
        laplace_u = self.laplace(output[0:-2, 0:1, :, :])  # 201x1x128x128
        laplace_v = self.laplace(output[0:-2, 1:2, :, :])  # 201x1x128x128

        u_x = self.dx(output[0:-2, 0:1, :, :])  # 201x1x128x128
        u_y = self.dy(output[0:-2, 0:1, :, :])  # 201x1x128x128

        v_x = self.dx(output[0:-2, 1:2, :, :])  # 201x1x128x128
        v_y = self.dy(output[0:-2, 1:2, :, :])  # 201x1x128x128

        # temporal derivatives - u
        u = output[:, 0:1, 2:-2, 2:-2]
        lent = u.shape[0]
        lenx = u.shape[3]
        leny = u.shape[2]
        u_conv1d = u.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        u_conv1d = u_conv1d.reshape(lenx*leny, 1, lent)
        u_t = self.dt(u_conv1d)  # lent-2 due to no-padding
        u_t = u_t.reshape(leny, lenx, 1, lent-2)
        u_t = u_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        # temporal derivatives - v
        v = output[:, 1:2, 2:-2, 2:-2]
        v_conv1d = v.permute(2, 3, 1, 0)  # [height(Y), width(X), c, step]
        v_conv1d = v_conv1d.reshape(lenx*leny, 1, lent)
        v_t = self.dt(v_conv1d)  # lent-2 due to no-padding
        v_t = v_t.reshape(leny, lenx, 1, lent-2)
        v_t = v_t.permute(3, 2, 0, 1)  # [step-2, c, height(Y), width(X)]

        u = output[0:-2, 0:1, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]
        v = output[0:-2, 1:2, 2:-2, 2:-2]  # [step, c, height(Y), width(X)]

        # make sure the dimensions consistent
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # Burger's eqn
        nu = 1/200

        f_u = (u_t - nu*laplace_u + u*u_x + v*u_y)/1
        f_v = (v_t - nu*laplace_v + u*v_x + v*v_y)/1

        return f_u, f_v

def get_ic_loss(model):
    Upconv = model.UpconvBlock
    init_state_low = model.init_state_low
    init_state_bicubic = F.interpolate(init_state_low, (100, 100), mode='bicubic')
    mse_loss = nn.MSELoss()
    init_state_pred = Upconv(init_state_low)
    loss_ic = mse_loss(init_state_pred, init_state_bicubic)
    return loss_ic

def loss_gen(output, loss_func):
    '''calculate the phycis loss'''
    
    # Padding x axis due to periodic boundary condition
    # shape: [27, 2, 131, 131]
    output = torch.cat((output[:, :, :, -2:], output, output[:, :, :, 0:3]), dim=3)
    output = torch.cat((output[:, :, -2:, :], output, output[:, :, 0:3, :]), dim=2)

    # get physics loss
    mse_loss = nn.MSELoss()
    f_u, f_v = loss_func.get_phy_Loss(output)
    loss = mse_loss(f_u, torch.zeros_like(f_u).cuda()) + \
           mse_loss(f_v, torch.zeros_like(f_v).cuda())
    return loss


def train(model, truth, n_iters, time_batch_size, learning_rate, dt, dx, cont=True): # cont=False
    # define some parameters
    model.train()
    train_loss_list = []
    best_loss = 10000
    # model
    if cont:
        model, optimizer, scheduler = load_model(model)
    else:
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=0.001)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.97)
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = param_group['lr']*0.7
    loss_func = loss_generator(dt, dx)
    for epoch in range(n_iters):
        # input: [time,. batch, channel, height, width]
        optimizer.zero_grad()
        num_time_batch = 1
        batch_loss, phy_loss, ic_loss, data_loss, val_loss = [0]*5
        for time_batch_id in range(num_time_batch):
            # update the first input for each time batch
            if time_batch_id == 0:
                pass
            else:
                pass
            # output is a list
            output, second_last_state = model()
            output = torch.cat(tuple(output), dim=0)
            mse_loss = nn.MSELoss()
            # 21x2x50x50
            pred, gt = output[0:-1:40, :, ::2, ::2], truth[::40, :, ::2, ::2].cuda()
            idx = int(pred.shape[0] * 0.9)
            pred_tra, pred_val = pred[:idx], pred[idx:]  # prediction
            gt_tra, gt_val = gt[:idx], gt[idx:]          # ground truth
            loss_data = mse_loss(pred_tra, gt_tra)       # data loss
            loss_valid = mse_loss(pred_val, gt_val)
            # get physics loss (for validation, not used for training)
            loss_phy = loss_gen(output, loss_func)
            loss = loss_data    # + loss_phy
            loss.backward(retain_graph=True)
            batch_loss += loss.item()
            phy_loss, data_loss, val_loss = loss_phy.item(), loss_data.item(), loss_valid.item()
        optimizer.step()
        scheduler.step()
        # print loss in each epoch
        print('[%d/%d %d%%] loss: %.11f, data_loss: %.11f, val_loss: %.7f, phy_loss: %.11f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0),
              batch_loss, data_loss, val_loss, phy_loss))
        train_loss_list.append(batch_loss)
        # Save checkpoint if hit best_loss (check every 10 epoch)
        if val_loss < best_loss and epoch%10==0:
            best_loss = val_loss
            for param_group in optimizer.param_groups:
                print(param_group['lr'])
            print('save model!!!')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, './model/checkpoint.pt')
    return train_loss_list

def save_model(model, model_name, save_path):
    ''' save the model '''
    torch.save(model.state_dict(), save_path + model_name + '.pt')


def load_model(model):
    # Load model and optimizer state
    checkpoint = torch.load('./model/checkpoint.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=0.0, weight_decay=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.97)
    return model, optimizer, scheduler


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def postProcess(output, truth, low_res, xmin, xmax, ymin, ymax, num, fig_save_path):
    ''' num: Number of time step
      '''
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_low_res, v_low_res = low_res[num, 0, ...], low_res[num, 1, ...]
    u_low_res, v_low_res = np.kron(u_low_res.detach().cpu().numpy(), np.ones((2, 2))), \
                           np.kron(v_low_res.detach().cpu().numpy(), np.ones((2, 2)))
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[:, 0:1]), axis=1), \
                           np.concatenate((v_low_res, v_low_res[:, 0:1]), axis=1)
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[0:1, :]), axis=0), \
                           np.concatenate((v_low_res, v_low_res[0:1, :]), axis=0)
    u_star, v_star = truth[num, 0, ...].numpy(), truth[num, 1, ...].numpy()
    u_pred, v_pred = output[num, 0, :, :].detach().cpu().numpy(), \
                     output[num, 1, :, :].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred-u_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=.5)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u-u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('u (Meas.)')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred-v_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=0.5)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v-v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 2].axis('square')
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_title('v (Meas.)')
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')

def postProcess_v2(output, truth, low_res, xmin, xmax, ymin, ymax, num, fig_save_path):
    ''' num: Number of time step
      '''
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    x_star, y_star = np.meshgrid(x, y)
    u_low_res, v_low_res = low_res[1*num, 0, ...], low_res[1*num, 1, ...]
    u_low_res, v_low_res = np.kron(u_low_res.detach().cpu().numpy(), np.ones((2, 2))), \
                           np.kron(v_low_res.detach().cpu().numpy(), np.ones((2, 2)))
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[:, 0:1]), axis=1), \
                           np.concatenate((v_low_res, v_low_res[:, 0:1]), axis=1)
    u_low_res, v_low_res = np.concatenate((u_low_res, u_low_res[0:1, :]), axis=0), \
                           np.concatenate((v_low_res, v_low_res[0:1, :]), axis=0)
    u_star, v_star = truth[1*num, 0, ...], truth[1*num, 1, ...]
    u_pred, v_pred = output[num, 0, :, :].detach().cpu().numpy(), \
                     output[num, 1, :, :].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('u (PeCRNN)')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_low_res, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('u (Meas.)')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('v (PeCRNN)')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_low_res, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 2].axis('square')
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_title('v (Meas.)')
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')

def plot_IC(model, ic_GT, init_state_low, fig_save_path):
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    init_state_bicubic = F.interpolate(init_state_low, (100, 100), mode='bicubic')
    init_state_upconv = model.UpconvBlock(init_state_low)

    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)
    x_star, y_star = np.meshgrid(x[:-1], y[:-1])

    # Ground truth
    u_star, v_star = ic_GT[0, 0, ...].numpy(), ic_GT[0, 1, ...].numpy()
    # IC generator
    u_Genr, v_Genr = init_state_upconv[0, 0, :, :].detach().cpu().numpy(), init_state_upconv[0, 1, :, :].detach().cpu().numpy()
    # Interp IC
    u_Intrp, v_Intrp = init_state_bicubic[0, 0, ...].detach().cpu().numpy(), init_state_bicubic[0, 1, ...].detach().cpu().numpy()
    #
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    #
    cf = ax[0, 0].scatter(x_star, y_star, c=u_Genr-u_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=.5)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_title('Gen-GT')
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2, vmax=1.6)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_title('u (Ref.)')
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[0, 2].scatter(x_star, y_star, c=u_Intrp-u_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-0.5, vmax=.5)
    ax[0, 2].axis('square')
    ax[0, 2].set_xlim([xmin, xmax])
    ax[0, 2].set_ylim([ymin, ymax])
    ax[0, 2].set_xticks([])
    ax[0, 2].set_yticks([])
    ax[0, 2].set_title('Intrp-GT')
    fig.colorbar(cf, ax=ax[0, 2], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 0].scatter(x_star, y_star, c=v_Genr-v_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=0.5)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_title('Gen-GT')
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=1.0, edgecolors='none', cmap='RdYlBu', marker='s', s=4, vmin=-2.8, vmax=0.5)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_title('v (Ref.)')
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    cf = ax[1, 2].scatter(x_star, y_star, c=v_Intrp-v_star, alpha=1.0, edgecolors='none', cmap='RdBu', marker='s', s=4, vmin=-.5, vmax=0.5)
    ax[1, 2].axis('square')
    ax[1, 2].set_xlim([xmin, xmax])
    ax[1, 2].set_ylim([ymin, ymax])
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    ax[1, 2].set_title('Intrp-GT')
    fig.colorbar(cf, ax=ax[1, 2], fraction=0.046, pad=0.04)
    #
    plt.savefig(fig_save_path + 'IC_comparison.png')
    plt.close('all')

def summary_parameters(model):
    for i in model.parameters():
        print(i.shape)

def add_noise(truth, pec=0.05):  # BxCx101x101
    from torch.distributions import normal
    assert truth.shape[1]==2
    uv = [truth[:,0:1,:,:], truth[:,1:2,:,:]]
    uv_noi = []
    torch.manual_seed(66)
    for truth in uv:
        n_distr = normal.Normal(0.0, 1.0)
        R = n_distr.sample(truth.shape)
        std_R = torch.std(R)          # std of samples
        std_T = torch.std(truth)
        noise = R*std_T/std_R*pec
        uv_noi.append(truth+noise)
    return torch.cat(uv_noi, dim=1)


if __name__ == '__main__':

    ################# prepare the input dataset ####################
    time_steps = 400     # 400 steps for PeRCNN (1200 steps inference), 50 steps (work best) for recurrent ResNet
    dt = 0.00025
    dx = 1.0/100
    dy = 1.0/100

    ################### define the Initial conditions ####################
    import scipy.io as sio
    UV = sio.loadmat('../../2d_burgers/data/Burgers_2001x2x100x100_[dt=00025].mat')['uv']
    truth_clean = torch.from_numpy(UV[100:1901]).float()  # [1801, 2, 100 X, 100 Y]
    # Add noise 10%
    UV = add_noise(torch.tensor(truth_clean), pec=0.1)
    # Retrieve initial condition
    IC = UV[0:1, :, :, :]                                                # 1x2x100x100

    # get the ground truth (noisy) for training
    truth = UV[:1*time_steps+1]                                                    # [601, 2, 100, 100]

    ################# build the model #####################
    # define the mdel hyperparameters
    time_batch_size = time_steps
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    n_iters = 1000
    learning_rate = 1e-3
    save_path = './model/'

    # Low-res initial condition
    U0_low = IC[0, 0, ::2, ::2]
    V0_low = IC[0, 1, ::2, ::2]
    h0 = torch.cat((U0_low[None, None, ...], V0_low[None, None, ...]), dim=1)
    init_state_low = h0.cuda()

    model = RecurrentResNet(input_channels = 2,
                            hidden_channels = 4,
                            output_channels = 2,
                            init_state_low = init_state_low,
                            input_kernel_size = 5,
                            input_stride = 1,
                            input_padding = 2,
                            step = steps,
                            effective_step = effective_step).cuda()

    # train the model
    start = time.time()
    # pretrain_upscaler(model.UpconvBlock, init_state_low, epoch=8000, plotFlag=True)
    train_loss_list = train(model, truth, n_iters, time_batch_size, learning_rate, dt, dx, cont=False)
    # train_bfgs(model, input, IC, truth, init_state, n_iters, time_batch_size, dt, dx, Nepoch=500, lr=2.0e-3)
    end = time.time()

    print('The training time is: ', (end-start))


    # Do the forward inference
    output, _ = model()
    output = torch.cat(tuple(output), dim=0)

    # Padding x,y axis due to periodic boundary condition
    output = torch.cat((output, output[:, :, :, 0:1]), dim=3)
    output = torch.cat((output, output[:, :, 0:1, :]), dim=2)
    truth_clean  = torch.cat((truth_clean, truth_clean[:, :, :, 0:1]), dim=3)
    truth_clean  = torch.cat((truth_clean, truth_clean[:, :, 0:1, :]), dim=2)

    # init_state_bicubic = F.interpolate(init_state_low, (100, 100), mode='bicubic')
    low_res = truth[:, :, ::2, ::2] # .cpu().numpy()

    fig_save_path = './figures/'
    
    # post-process
    for i in range(0, 201, 5):
        print(i)
        postProcess_v2(output, truth_clean, low_res, xmin=0, xmax=1, ymin=0, ymax=1,
                     num=i, fig_save_path=fig_save_path)

    import scipy
    UV = output.detach().cpu().numpy()
    UV = np.transpose(UV, (1, 0, 2, 3))
    scipy.io.savemat('uv_2x1601x100x100_[ResNet].mat', {'uv': UV[:, :-1, :-1, :-1]})








