import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

class NavStokesPINN(nn.Module):
    def __init__(self, x, y, t, u, v, layers, device):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.iter = 0
        X = np.concatenate([x, y, t], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        self.X = X

        self.x = X[:,0:1]
        self.y = X[:,1:2]
        self.t = X[:,2:3]

        self.u = u
        self.v = v

        self.layers = layers
        self.device = device

        self.linears = self.initialize_NN(layers)   # linear units array that contain the weights and biases
        self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(self.device)
        self.lambda_2 = torch.tensor([0.0], requires_grad=True).to(self.device)

        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        self.register_parameter('lambda_1', self.lambda_1)
        self.register_parameter('lambda_2', self.lambda_2)

        self.x = torch.from_numpy(self.x).float().to(self.device)
        self.y = torch.from_numpy(self.y).float().to(self.device)
        self.t = torch.from_numpy(self.t).float().to(self.device)
        self.u = torch.from_numpy(self.u).float().to(self.device)
        self.v = torch.from_numpy(self.v).float().to(self.device)
        self.lb = torch.from_numpy(self.lb).float().to(self.device)
        self.ub = torch.from_numpy(self.ub).float().to(self.device)

        
        #self.loss = self.loss()

        self.optimizer = torch.optim.LBFGS(self.parameters(), max_iter=50, max_eval=50000)
        self.optimizer_Adam = torch.optim.Adam(self.parameters())


    def initialize_NN(self, layers):
        linears = nn.ModuleList(nn.Linear(layers[i], layers[i+1]) for i in range(len(layers) - 1))
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(linears[i].weight.data, gain=1.0)
            nn.init.zeros_(linears[i].bias.data)
        
        return linears
        
    def net_NS(self, x, y, t):      # loss_PDE
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        x_grad = x.clone()
        x_grad.requires_grad = True

        y_grad = y.clone()
        y_grad.requires_grad = True

        t_grad = t.clone()
        t_grad.requires_grad = True

        psi_and_p = self.neural_net(torch.cat([x_grad, y_grad, t_grad], 1))
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]

        u = autograd.grad(psi, y_grad, torch.ones([psi.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        v = -autograd.grad(psi, x_grad, torch.ones([psi.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]

        u_t = autograd.grad(u, t_grad, torch.ones([u.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        u_x = autograd.grad(u, x_grad, torch.ones([u.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        u_y = autograd.grad(u, y_grad, torch.ones([u.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]

        u_xx = autograd.grad(u_x, x_grad, torch.ones([u_x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        u_yy = autograd.grad(u_y, y_grad, torch.ones([u_y.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]

        v_t = autograd.grad(v, t_grad, torch.ones([u.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        v_x = autograd.grad(v, x_grad, torch.ones([u.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        v_y = autograd.grad(v, y_grad, torch.ones([u.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]

        v_xx = autograd.grad(v_x, x_grad, torch.ones([u_x.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        v_yy = autograd.grad(v_y, y_grad, torch.ones([u_y.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]

        p_x = autograd.grad(p, x_grad, torch.ones([p.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]
        p_y = autograd.grad(p, y_grad, torch.ones([p.shape[0], 1]).to(self.device), retain_graph=True, create_graph=True)[0]

        f_u = u_t + lambda_1*(u*u_x + v*u_y) + p_x - lambda_2*(u_xx + u_yy)
        f_v = v_t + lambda_1*(u*v_x + v*v_y) + p_y - lambda_2*(v_xx + v_yy)
        return u, v, f_u, f_v   # NOT RETURNING P ANYMORE

        
    def neural_net(self, X):        # forward pass through the network
        if torch.is_tensor(X) != True:
            X = torch.from_numpy(X)

        X = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        a = X.float()

        for i in range(len(self.layers) - 2):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)

        return a
        
    def loss(self):
        u_pred, v_pred, f_u_pred, f_v_pred = self.net_NS(self.x, self.y, self.t)
        loss = self.loss_function(u_pred, self.u) + \
            self.loss_function(v_pred, self.v) + \
            self.loss_function(f_u_pred, torch.zeros_like(f_u_pred)) + \
            self.loss_function(f_v_pred, torch.zeros_like(f_v_pred))
        return loss


    def train(self, nIter):
        #for i in range(nIter):
        #    loss = self.loss()
        #    self.optimizer_Adam.zero_grad()
        #    loss.backward()
        #    self.optimizer_Adam.step()
        #    if i % 10:
        #        print("Adam loss:", loss)
        
        self.optimizer.step(self.closure)
            

    def closure(self):
        self.optimizer.zero_grad()
        loss = self.loss()

        loss.backward()
        self.iter += 1

        if self.iter % 10 == 0:
            print("LBFGS loss:", loss)
        return loss
