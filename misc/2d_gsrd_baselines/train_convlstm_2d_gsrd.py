import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import scipy.io as scio
import time
import os

torch.manual_seed(66)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
np.random.seed(66)
torch.set_default_dtype(torch.float32)

print('Training 2D-GS Equation ...')


def boundary_padding(x, p = 1):
    ''' Boundary padding for tensor'''

    x_pad = torch.cat((x[:, :, :, -p:], x, x[:, :, :, 0:p]), dim=3)
    x_pad = torch.cat((x_pad[:, :, -p:, :], x_pad, x_pad[:, :, 0:p, :]), dim=2)

    return x_pad


def initialize_weights(module):

    if isinstance(module, nn.Conv2d):
        #nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
        c = 1
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))
     
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, input_kernel_size, 
        input_stride, input_padding):

        super(ConvLSTMCell, self).__init__()

        # the initial parameters
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.hidden_kernel_size = 5
        self.input_kernel_size = input_kernel_size  

        self.input_stride = input_stride
        self.input_padding = input_padding

        # because have 4 gates
        self.num_features = 4

        # define padding to keep the dimensions unchanged
        # it is only for the hidden state
        self.padding = int((self.hidden_kernel_size - 1) / 2)

        # weights of input gate, stride = 1
        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True)

        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=0, bias=False)

        # weights of forget gate
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True)

        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=0, bias=False)

        # weights of candidate gate
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding,
            bias=True)

        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=0, bias=False)

        # weights of output gate
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, 
            self.input_kernel_size, self.input_stride, self.input_padding, 
            bias=True)

        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, 
            self.hidden_kernel_size, 1, padding=0, bias=False)       

        # init
        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        #nn.init.zeros_(self.Wxo.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):

        # periodic pading
        x = boundary_padding(x, p = 2)
        h = boundary_padding(h, p = 2)

        # forward
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)

        return ch, cc

    def init_hidden_tensor(self, prev_state):

        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda())


class convlstm(nn.Module):
    def __init__(self, input_channels, hidden_channels, 
        input_kernel_size, input_stride, input_padding, 
        num_layers, step = 1, effective_step = [1]):

        super(convlstm, self).__init__()

        # input channels of layer includes input_channels and hidden_channels of cells 
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.input_kernel_size = input_kernel_size
        self.input_stride = input_stride
        self.input_padding = input_padding
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []
        self.num_layers = num_layers
     
        # ConvLSTM
        for i in range(0, self.num_layers):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                input_channels = self.input_channels[i],
                hidden_channels = self.hidden_channels[i],
                input_kernel_size = self.input_kernel_size[i],
                input_stride = self.input_stride[i],
                input_padding = self.input_padding[i])
        
            setattr(self, name, cell)
            self._all_layers.append(cell)  

        # output layer
        self.output_layer = nn.Conv2d(32, 2, kernel_size = 5, stride = 1, padding = 0)

        # initialize weights
        self.apply(initialize_weights)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, initial_state, input):
        
        self.initial_state = initial_state
        internal_state = []
        outputs = []
        second_last_state = []

        for step in range(self.step):
            x = input
             
            # convlstm
            for i in range(0, self.num_layers):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    # output_size needs to be changed here
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state = self.initial_state[i])  
                    internal_state.append((h,c))
                
                # one-step forward
                (h, c) = internal_state[i]
                x, new_c = getattr(self, name)(x, h, c)
                internal_state[i] = (x, new_c)                               

            # output
            x = boundary_padding(x, p = 2)
            x = self.output_layer(x)

            if step == (self.step - 2):
                second_last_state = internal_state.copy()
                
            if step in self.effective_step:
                outputs.append(x)                

        return outputs, second_last_state



def train(model, input, initial_state, truth, n_iters, lr, save_path):
    
    # define some parameters
    train_loss_list = []
    val_loss_list = []
    second_last_state = []
    best_loss = 1e4

    # model
    optimizer = optim.Adam(model.parameters(), lr = lr) 
    scheduler = StepLR(optimizer, step_size = 100, gamma = 0.985)  
    mse_loss = nn.MSELoss()
    
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    ## Training loop
    for epoch in range(n_iters):
        
        optimizer.zero_grad()
        output, second_last_state = model(initial_state, input)
        output = torch.cat(tuple(output), dim=0) # [u1,..u400]
        ghost_u0 = torch.zeros(1,2,100,100).cuda()
        output = torch.cat((ghost_u0, output), dim=0)
        # todo: concatenate the interpolated IC (1x2x100x100)
        
        # Training and validation dataset split (use val dataset to tune hyperparameters)
        pred, gt = output[1:-1:20, :, ::4, ::4], truth[1::20, :, ::4, ::4].cuda()   # todo: double check the output shape match
        
        # split dataset
        idx = int(pred.shape[0] * 0.9)
        pred_tra, pred_val = pred[:idx], pred[idx:]  # prediction
        gt_tra, gt_val = gt[:idx], gt[idx:]          # ground truth
        
        # compute the loss
        loss_data = mse_loss(pred_tra, gt_tra)       # data loss
        loss_valid = mse_loss(pred_val, gt_val)
        data_loss, val_loss = loss_data.item(), loss_valid.item()
        train_loss_list.append(data_loss)
        val_loss_list.append(val_loss)
        loss_data.backward()
        
        optimizer.step()
        scheduler.step()    
        
        # print train loss and validation loss
        print('[%d/%d %d%%] train loss: %.8f, val los: %.8f' % ((epoch+1), n_iters, ((epoch+1)/n_iters*100.0), 
            data_loss, val_loss))
    
        # save model
        if data_loss < best_loss:
            #save_model(model, 'convlstm_fn', save_path)
            save_checkpoint(model, optimizer, scheduler, save_path)
            best_loss = data_loss    
    
    return train_loss_list, val_loss_list
    

def post_process(output, true, axis_lim, uv_lim, num, fig_save_path):
    ''' 
    axis_lim: [xmin, xmax, ymin, ymax]
    uv_lim: [u_min, u_max, v_min, v_max]
    num: Number of time step
    '''

    # get the limit data
    xmin, xmax, ymin, ymax = axis_lim
    u_min, u_max, v_min, v_max = uv_lim

    # grid
    x = np.linspace(-0.5, 0.5, 100+1)
    y = np.linspace(-0.5, 0.5, 100+1)
    x_star, y_star = np.meshgrid(x, y)
    
    u_star = true[num, 0, 1:-1, 1:-1]
    u_pred = output[num, 0, 1:-1, 1:-1].detach().cpu().numpy()

    v_star = true[num, 1, 1:-1, 1:-1]
    v_pred = output[num, 1, 1:-1, 1:-1].detach().cpu().numpy()

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    cf = ax[0, 0].scatter(x_star, y_star, c=u_pred, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=u_min, vmax=u_max)
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 0].set_title('u-ConvLSTM')
    fig.colorbar(cf, ax=ax[0, 0])

    cf = ax[0, 1].scatter(x_star, y_star, c=u_star, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=u_min, vmax=u_max)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    # cf.cmap.set_under('whitesmoke')
    # cf.cmap.set_over('black')
    ax[0, 1].set_title('u-Ref.')
    fig.colorbar(cf, ax=ax[0, 1])

    cf = ax[1, 0].scatter(x_star, y_star, c=v_pred, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=v_min, vmax=v_max)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[1, 0].set_title('v-ConvLSTM')
    fig.colorbar(cf, ax=ax[1, 0])

    cf = ax[1, 1].scatter(x_star, y_star, c=v_star, alpha=0.9, edgecolors='none', 
        cmap='RdYlBu', marker='s', s=4, vmin=v_min, vmax=v_max)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    cf.cmap.set_under('whitesmoke')
    cf.cmap.set_over('black')
    ax[1, 1].set_title('v-Ref.')
    fig.colorbar(cf, ax=ax[1, 1])

    # plt.draw()
    plt.savefig(fig_save_path + 'uv_comparison_'+str(num).zfill(3)+'.png')
    plt.close('all')

    return u_star, u_pred, v_star, v_pred


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''
    # save_dir = './model/AE_pretrain_fn/checkpoint100.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    #save_dir = './model/AE_pretrain_fn/checkpoint100.pt'
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler    
    
    

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


def frobenius_norm(tensor):
    return np.sqrt(np.sum(tensor ** 2))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    ################# prepare the input dataset ####################
    time_steps = 800    # 800 for training, 1700 for inference
    dt = 0.5
    dx = 1.0/100
    dy = 1.0/100

    ################### define the Initial conditions ####################
    UV = scio.loadmat('../../2d_gs_rd/data/2DRD_2x3001x100x100_[dt=05].mat')['uv']
    truth_clean = torch.from_numpy(UV[:,0:801,:,:]).float()  # [2,801, 100 X, 100 Y]
    truth_clean = np.transpose(truth_clean, (1,0,2,3))   # [801,2,100,100]
    
    # Add noise 10%
    UV = add_noise(torch.tensor(truth_clean), pec=0.1)
    # Retrieve initial condition
    IC = UV[0:1, :, :, :]                                         # 1x2x100x100

    # get the ground truth (noisy) for training
    truth = UV[:time_steps+1]                              # [401, 2, 100, 100]

    ################# build the model #####################
    # Low-res initial condition
    U0_low = IC[0, 0, ::4, ::4]
    V0_low = IC[0, 1, ::4, ::4]
    h0 = np.concatenate((U0_low[None, None, ...], V0_low[None, None, ...]), 1) # [1,2,25,25]
    h0 = np.repeat(h0, 32/2, axis=1)
    init_state_low = torch.tensor(h0).cuda() # [1,32,50,50]

    # Bicubic interpolation function,1x2x100x100
    init_state_bicubic = F.interpolate(init_state_low, (100, 100), mode='bicubic') # [1, 32, 100, 100]
    
    # initial states for convlstm
    num_layer = 1
    (h0, c0) = (init_state_bicubic, init_state_bicubic)
    initial_state = []
    for i in range(num_layer):
        initial_state.append((h0,c0))

    # build the input (x,y)
    x = np.linspace(0,1,101)
    x = x[0:-1]
    X,Y = np.meshgrid(x,x)
    input = np.concatenate((X[None,None,...],Y[None,None, ...]), 1) # [1,2,100,100]
    input = torch.tensor(input, dtype=torch.float32).cuda()

    # build the model
    time_batch_size = time_steps
    steps = time_batch_size + 1
    effective_step = list(range(0, steps))
    num_time_batch = int(time_steps / time_batch_size)
    n_iters = 5000
    lr = 8e-4
    model_save_path = './model/convlstm_2dgs/checkpoint800.pt'
    fig_save_path = './figures/convlstm_2dgs/'  

    model = convlstm(
        input_channels = 2, 
        hidden_channels = [32], 
        input_kernel_size = [5], 
        input_stride = [1], 
        input_padding = [0],  # due to boundary padding
        num_layers = num_layer,
        step = steps, 
        effective_step = effective_step).cuda()

    model_para = count_parameters(model)
    print('number of model parameters: ', model_para)


    # start = time.time()
    # train_loss, val_loss = train(model, input, initial_state, truth, n_iters, lr, model_save_path)
    # end = time.time()
    # np.save('./model/convlstm_2dgs/train_loss800', train_loss)
    # np.save('./model/convlstm_2dgs/val_loss800', val_loss)
    # print('The training time is: ', (end-start))

    # # load model
    # time_batch_size_load = time_steps
    # steps_load = time_batch_size_load + 1
    # num_time_batch = int(steps_load / time_batch_size_load)
    # effective_step = list(range(0, steps_load))

    # model = convlstm(
    #     input_channels = 2,
    #     hidden_channels = [32],
    #     input_kernel_size = [5],
    #     input_stride = [1],
    #     input_padding = [0],  # due to boundary padding
    #     num_layers = num_layer,
    #     step = steps_load,
    #     effective_step = effective_step).cuda()

    # #load_model(model, 'convlstm_fn', model_save_path)
    # model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, save_dir=model_save_path)

    # output, _ = model(initial_state, input)
    # output = torch.cat(tuple(output), dim=0)
    # #output = torch.cat((input.cuda(), output), dim=0)

    # # Padding x axis due to periodic boundary condition
    # output = torch.cat((output[:, :, :, -1:], output, output[:, :, :, 0:2]), dim=3)

    # # Padding y axis due to periodic boundary condition
    # output = torch.cat((output[:, :, -1:, :], output, output[:, :, 0:2, :]), dim=2)

    # # get the ground truth for plotting
    # # [t, c, h, w]
    # truth = truth_clean[1:time_steps+1]
    # #truth = truth[1:,:,:,:] #[400,2,100,100]

    # # [51, 2, 131, 131]
    # truth = np.concatenate((truth[:, :, :, -1:], truth, truth[:, :, :, 0:2]), axis=3)
    # truth = np.concatenate((truth[:, :, -1:, :], truth, truth[:, :, 0:2, :]), axis=2)

    # # post-process
    # ten_true = []
    # ten_pred = []
    # for i in range(0, 40):
    #     u_star, u_pred, v_star, v_pred = post_process(output, truth, [-0.5,0.5,-0.5,0.5],
    #         [0.0, 1.2, -0.1, 1.0], num = 20*i, fig_save_path=fig_save_path)







