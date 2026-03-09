import os
import math
import time
import torch
import torch.nn.functional as F
import logging
import numpy as np
# from matplotlib import pyplot as plt

torch.manual_seed(0)  # set seed for reproducibility


class DGMNet(torch.nn.Module):
    """
    deep Galerkin approach to solve PDE with utility functions
    """
    def __init__(
        self,
        dgm_f_fun,
        dgm_deriv_map,
        boundary_fun=None,
        problem_name="tmp",
        dgm_zeta_map=None,
        deriv_condition_deriv_map=None,
        deriv_condition_zeta_map=None,
        dim_out=None,
        phi_fun=(lambda x: x),
        psi_fun=(lambda x: x),
        dt_order=2,
        x_lo_original=0.0,
        x_hi_original=1.0,
        overtrain_rate=1.0,
        x_low_original_boundary_fun=None,
        x_high_original_boundary_fun=None,
        t_lo=0.0,
        t_hi=1.0,
        tx_to_tx_init_ratio = 5,
        neurons=20,
        layers=5,
        dgm_lr=1e-3,
        batch_normalization=False,
        weight_decay=0,
        dgm_nb_states=1000,
        epochs=3000,
        device="cpu",
        dgm_activation="tanh",
        directory=None,
        verbose=False,
        fix_all_dim_except_first=False,
        save_as_tmp=False,
        **kwargs,
    ):
        super(DGMNet, self).__init__()
        self.f_fun = dgm_f_fun
        self.boundary_fun = boundary_fun
        self.n, self.dim_in = dgm_deriv_map.shape # self.n = 3, self.dim_in = 1
        # print(f"deriv_map has shape {dgm_deriv_map.shape} with n={self.n} and dim_in={self.dim_in}")
        
        """Working on the derivative map, we need to add time derivative at the beginning."""
        # add one more dimension of time to the left of deriv_map
        self.deriv_map = np.append(np.zeros((self.n, 1)), dgm_deriv_map, axis=-1)
        # add dt to the top of deriv_map
        self.dt_order = dt_order
        self.deriv_map = np.append(np.array([[self.dt_order] + [0] * self.dim_in]), self.deriv_map, axis=0)
        # the final deriv_map has the shape of (n + 1) x (dim + 1)
        """End of working on the derivative map."""
        
        
        """
        dgm_zeta_map = zeta_map = np.array([0, 0, 0])
        Zeta map refers to u. 
        Essentially the forward function is composed of 1 function: u.
        So you can imagine for example in a usual sense we have 
        u(t,x,y) = [u_1(t,x,y), u_2(t,x,y)]
        so if we want to get u_1, we set coordinate=0
        if we want to get u_2, we set coordinate=1
        """
        self.zeta_map = dgm_zeta_map if dgm_zeta_map is not None else np.zeros(self.n, dtype=int)
        
        """
        First loop: 
        idx = 0, c = [1,0]
        grad += \\partial_x u_1
        Second loop:
        idx = 1, c = [0,1]
        grad += \\partial_y u_2
        Hence grad = \\partial_x u_1 + \\partial_y u_2
        which is the divergence of u
        """
        self.deriv_condition_deriv_map = deriv_condition_deriv_map
        self.deriv_condition_zeta_map = deriv_condition_zeta_map
        
        self.dim_out = dim_out if dim_out is not None else self.zeta_map.max() + 1 # self.zeta_map.max() = 1, so self.dim_out = 2
        self.coordinate = np.array(range(self.dim_out)) # self.coordinate = [0,1]

        self.phi_fun = phi_fun
        self.psi_fun = psi_fun
        
        # layers: [in->hid] + (layers x [hid->hid]) + [hid->out]
        # With current config (dim_in=2, neurons=20, layers=5, dim_out=2):
        #   total Linear layers = 7
        #   1) Linear(3 -> 20)        # input is (t, x1, x2)
        #   2-6) five Linear(20 -> 20)
        #   7) Linear(20 -> 2)        # outputs (u1, u2)
        self.u_layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim_in + 1, neurons, device=device)]
            + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
            + [torch.nn.Linear(neurons, self.dim_out, device=device)]
        )

        # BN count = len(u_layer) - 1 = 6
        # One BN for each pre-output layer (the final output layer has no BN).
        # BN is only applied in forward() when self.batch_normalization == True.
        self.u_bn_layer = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(len(self.u_layer) - 1)]
        )
        
        # p_layer architecture (for pressure):
        # total Linear layers = 7 (same structure as u_layer)
        # 1) Linear(3 -> 20)        # input is (t, x1, x2)
        # 2-6) five Linear(20 -> 20)
        # 7) Linear(20 -> 1)        # output is scalar p
        self.p_layer = torch.nn.ModuleList(
            [torch.nn.Linear(self.dim_in + 1, neurons, device=device)]
            + [torch.nn.Linear(neurons, neurons, device=device) for _ in range(layers)]
            + [torch.nn.Linear(neurons, 1, device=device)]
        )

        # BN count = len(p_layer) - 1 = 6
        # One BN for each pre-output layer (the final output layer has no BN).
        # BN is only applied in forward() when self.batch_normalization == True.
        self.p_bn_layer = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(neurons, device=device) for _ in range(len(self.p_layer) - 1)]
        )
        
        self.lr = dgm_lr
        self.weight_decay = weight_decay

        self.loss = torch.nn.MSELoss()
        self.activation = {
            "tanh": torch.nn.Tanh(),
            "relu": torch.nn.ReLU(),
            "softplus": torch.nn.ReLU(),
        }[dgm_activation]
        self.batch_normalization = batch_normalization
        self.nb_states = dgm_nb_states
        self.tx_to_tx_init_ratio = tx_to_tx_init_ratio
        
        
        self.ori_x_lo = x_lo_original
        self.ori_x_hi = x_hi_original
        x_lo, x_hi = (
            x_lo_original - overtrain_rate * (x_hi_original - x_lo_original),
            x_hi_original + overtrain_rate * (x_hi_original - x_lo_original),
        )
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.x_low_original_boundary_fun = x_low_original_boundary_fun
        self.x_high_original_boundary_fun = x_high_original_boundary_fun

        self.t_lo = t_lo
        self.t_hi = t_hi
        self.epochs = epochs
        self.device = device
        self.verbose = verbose
        self.fix_all_dim_except_first = fix_all_dim_except_first

        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time stamp
        self.directory = directory if directory is not None else ""
        self.working_dir = (
            os.path.join(self.directory, "logs", "tmp")
            if save_as_tmp
            else os.path.join(self.directory, "logs", f"{timestr}-{problem_name}")
        )
        self.working_dir_full_path = os.path.join(
            os.getcwd(),
            self.working_dir,
        )
        
        self.log_config()
        self.eval()

    def gen_sample(self):
        """
        Generate two uniform sample sets on the space-time box:
        - tx: interior space-time samples
        - tx_init: initial-time samples (t fixed at t_lo)
        """
        # Number of interior points: ratio * nb_states.
        nb_tx = int(self.tx_to_tx_init_ratio * self.nb_states)

        # Sample interior time values uniformly on [t_lo, t_hi].
        t_tx = self.t_lo + (self.t_hi - self.t_lo) * torch.rand(nb_tx, device=self.device)

        # Sample interior spatial values uniformly on [x_lo, x_hi] for each spatial dimension.
        x_tx = self.x_lo + (self.x_hi - self.x_lo) * torch.rand(
            (self.dim_in, nb_tx), device=self.device
        )

        # Stack into tx with shape (dim_in + 1, nb_tx): row 0=t, rows 1..=dim_in are x components.
        tx = torch.cat((t_tx.unsqueeze(0), x_tx), dim=0)

        # Build initial-time samples: t fixed at t_lo, x sampled uniformly in the same box.
        t_init = self.t_lo * torch.ones(self.nb_states, device=self.device)
        x_init = self.x_lo + (self.x_hi - self.x_lo) * torch.rand(
            (self.dim_in, self.nb_states), device=self.device
        )

        # Stack into tx_init with shape (dim_in + 1, nb_states).
        tx_init = torch.cat((t_init.unsqueeze(0), x_init), dim=0)
        
        if self.x_low_original_boundary_fun:
            # Build boundary samples at x_lo: x fixed at ori_x_lo, t sampled uniformly in [t_lo, t_hi].
            t_boundary = self.t_lo + (self.t_hi - self.t_lo) * torch.rand(
                self.nb_states, device=self.device
            )
            x_boundary = self.ori_x_lo * torch.ones(
                (self.dim_in, self.nb_states), device=self.device
            )
            # Stack into tx_boundary_at_x_lo with shape (dim_in + 1, nb_states).
            tx_boundary_at_x_lo = torch.cat((t_boundary.unsqueeze(0), x_boundary), dim=0)
        else:
            tx_boundary_at_x_lo = None
            
        if self.x_high_original_boundary_fun:
            # Build boundary samples at x_hi: x fixed at ori_x_hi, t sampled uniformly in [t_lo, t_hi].
            t_boundary = self.t_lo + (self.t_hi - self.t_lo) * torch.rand(
                self.nb_states, device=self.device
            )
            x_boundary = self.ori_x_hi * torch.ones(
                (self.dim_in, self.nb_states), device=self.device
            )
            # Stack into tx_boundary_at_x_hi with shape (dim_in + 1, nb_states).
            tx_boundary_at_x_hi = torch.cat((t_boundary.unsqueeze(0), x_boundary), dim=0)
        else:
            tx_boundary_at_x_hi = None

        return tx, tx_init, tx_boundary_at_x_lo, tx_boundary_at_x_hi

    def train_and_eval(self, debug_mode=False):
        """
        generate sample and evaluate (plot) NN approximation when debug_mode=True
        """
        # initialize optimizer
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        start = time.time()
        self.train()  # training mode

        # loop through epochs
        for epoch in range(self.epochs):
            tx, tx_init, tx_boundary_at_x_lo, tx_boundary_at_x_hi = self.gen_sample()

            # clear gradients and evaluate training loss
            optimizer.zero_grad()

            loss = 0
            # self.coordinate = [0,1]
            for idx in self.coordinate: # for idx in [0,1]:
                # initial-condition loss + pde loss
                # Initial-condition loss at t=t_lo using tx_init samples.
                pred_init = self.forward(tx_init.T, coordinate=idx)
                target_init = self.phi_fun(tx_init[1:, :], total_dim=self.dim_in, coordinate=idx)
                loss = self.loss(pred_init, target_init) # torch.nn.MSELoss()
                
                tx_init_2 = tx_init.detach().clone().requires_grad_(True)
                pred_init_psi = self.nth_derivatives(
                    np.array([1] + [0] * self.dim_in),  # dt derivative
                    self.forward(tx_init_2.T, coordinate=idx),
                    tx_init_2,
                )
                target_init_psi = self.psi_fun(tx_init[1:, :], total_dim=self.dim_in, coordinate=idx)
                loss = loss + self.loss(pred_init_psi, target_init_psi)
                
                if tx_boundary_at_x_lo is not None:
                    pred_boundary_at_x_lo = self.forward(tx_boundary_at_x_lo.T, coordinate=idx)
                    target_boundary_at_x_lo = self.x_low_original_boundary_fun(
                        tx_boundary_at_x_lo, total_dim=self.dim_in, coordinate=idx
                    )
                    loss = loss + self.loss(pred_boundary_at_x_lo, target_boundary_at_x_lo)
                
                if tx_boundary_at_x_hi is not None:
                    pred_boundary_at_x_hi = self.forward(tx_boundary_at_x_hi.T, coordinate=idx)
                    target_boundary_at_x_hi = self.x_high_original_boundary_fun(
                        tx_boundary_at_x_hi, total_dim=self.dim_in, coordinate=idx
                    )
                    loss = loss + self.loss(pred_boundary_at_x_hi, target_boundary_at_x_hi)

                pde_loss_term = self.pde_loss(tx, coordinate=idx)
                loss = loss + pde_loss_term
                    
            # update model weights
            loss.backward()
            optimizer.step()

            # print loss information every 500 epochs
            if epoch % 500 == 0 or epoch + 1 == self.epochs:
                if debug_mode:
                    grid = np.linspace(self.x_lo, self.x_hi, 100).astype(np.float32)
                    x_mid = (self.x_lo + self.x_hi) / 2
                    grid_nd = np.concatenate(
                        (
                            self.t_lo * np.ones((1, 100)),
                            np.expand_dims(grid, axis=0),
                            x_mid * np.ones((self.dim_in - 1, 100)),
                        ),
                        axis=0,
                    ).astype(np.float32)
                    self.eval()
                    for idx in self.coordinate:
                        nn = (
                            self(torch.tensor(grid_nd.T, device=self.device), coordinate=idx)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        plt.plot(grid, nn)
                        plt.title(f"DGM approximation of coordinate {idx} at epoch {epoch}.")
                        plt.show()
                    self.train()
                if self.verbose:
                    logging.info(f"Epoch {epoch} with loss {loss.detach()}")

        torch.save(
            self.state_dict(), f"{self.working_dir_full_path}/checkpoint.pt"
        )
        if self.verbose:
            logging.info(
                f"Training of neural network with {self.epochs} epochs take {time.time() - start} seconds."
            )
        self.eval()
    
    def forward(self, x, coordinate=0, all_u=False):
        """
        self(x) evaluates the neural network approximation NN(x)
        """
        # -1 corresponds to p_layer
        # else 0 to dim_out corresponds to u_layer
        layer = self.u_layer if coordinate >= 0 else self.p_layer
        bn_layer = self.u_bn_layer if coordinate >= 0 else self.p_bn_layer
        coordinate = 0 if coordinate < 0 else coordinate
        
        for idx, (f, bn) in enumerate(zip(layer[:-1], bn_layer)):
            # f is torch.nn.Linear
            # bn is torch.nn.BatchNorm1d
            tmp = f(x)
            # BN is created regardless, but used only when this flag is True.
            if self.batch_normalization:
                tmp = bn(tmp)
            tmp = self.activation(tmp)
            if idx == 0:
                x = tmp
            else:
                # resnet
                x = tmp + x

        x = layer[-1](x)
        if all_u:
            return x
        else:
            return x[:, coordinate] # returns shape (N,)

    def pde_loss(self, x, coordinate):
        """
        Why we need to detach and clone x here?
        Because x is generated by gen_sample function
        and during the process of sampling, it may have some gradient history in computation graph
        If we do not detach and clone, when we call autograd.grad later in nth_derivatives function
        it may cause error that the computation graph takes into account of the previous history
        during our sampling process.
        """
        x = x.detach().clone().requires_grad_(True)
        """
        recall that deriv_map has the shape of (n + 1) x (dim + 1)
        with deriv_map[0] representing du/dt
        zeta_map has the shape of n
        So self.deriv_map[0] = [2,0]
        self.forward(x.T, coordinate=coordinate) is the u_{coordinate} value
        nth_derivative function will differentiate with respect to cur_dim (which is 0) for cur_order (which is 2) times
        meaning it differentiate u_{coordinate} with respect to t twice
        So it is \\partial_{tt} u_{coordinate}.
        So the reason is (a) make x a clean gradient target, and (b) prevent gradients from flowing back into the sampling step.
        Make a fresh leaf for higher-order derivatives and avoid back propagating through sampling.
        """
        dtt = self.nth_derivatives(self.deriv_map[0], self.forward(x.T, coordinate=coordinate), x)
        """
        Shape notes for current run:
        - input x comes from tx, so x has shape (3, N), with N=2000 in gen_sample().
        - x.T has shape (N, 2), matching forward() input format: columns are (t, x).
        - self.forward(x.T, coordinate=coordinate) returns shape (N,) when all_u=False.
        - self.deriv_map[0] = [2, 0], so nth_derivatives computes d²/dt² of that (N,) output.
        - dtt therefore has shape (N,) (here N=2000).
        """

        fun_and_derivs = []
        for order, zeta in zip(self.deriv_map[1:], self.zeta_map):
            """
            For example, 
            order = [0,2,0] and zeta = 0 means we want to calculate \\partial_{xx} u_1
            order = [0,0,2] and zeta = 0 means we want to calculate \\partial_{yy} u_1
            order = [0,2,0] and zeta = 1 means we want to calculate \\partial_{xx} u_2
            order = [0,0,2] and zeta = 1 means we want to calculate \\partial_{yy} u_2
            order = [1,0,0] and zeta = -1 means we want to calculate \\partial_{t} p
            order = [1,0,0] and zeta = 0 means we want to calculate \\partial_{t} u_1
            So we call nth_derivatives with order and self.forward(x.T, coordinate=zeta)
            which is the function value u_1 or u_2 or p
            and x is the variable we differentiate with respect to
            """
            fun_and_derivs.append(self.nth_derivatives(order, self.forward(x.T, coordinate=zeta), x))
            # shape: (3, N)
            # row mapping (for current deriv_map + zeta_map):
            #  0: u
            #  1: d/dxx u
            #  2: d/dyy u

        fun_and_derivs = torch.stack(fun_and_derivs)
        """
        self.f_fun(fun_and_derivs):
            f(u) = -(∂xx u + ∂yy u) + u - u^3
            return f
        So the pde is \\partial_t u_{coordinate} + f = 0
        Hence the loss is MSE( \\partial_t u_{coordinate} + f , 0 )
        """
        return self.loss(dtt + self.f_fun(fun_and_derivs), torch.zeros_like(dtt)) # torch.nn.MSELoss()

    @staticmethod
    def nth_derivatives(order, y, x):
        """
        calculate the derivatives of y wrt x with order `order`

        Shape convention used in this file:
        - x: (3, N), row 0=t, row 1=x1, row 2=x2
        - y: (N,) at entry (typically output of forward(..., all_u=False))

        For each derivative step:
        - grads = autograd.grad(y.sum(), x)[0] has SAME shape as x => (3, N)
        - selecting grads[cur_dim] picks one row => shape (N,)
        - so y stays shape (N,) after each update

        Why y.sum():
        - autograd.grad expects a scalar output by default.
        - summing allows one gradient call for the batch.
        - with sample-wise independent mapping (no BN in your run), this yields per-sample derivatives.
        """
        for cur_dim, cur_order in enumerate(order):
            for _ in range(int(cur_order)):
                try:
                    # grads shape: (3, N) (same as x)
                    grads = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
                except RuntimeError:
                    # when very high order derivatives are taken for polynomial function
                    # it has 0 gradient but torch has difficulty knowing that
                    # hence we handle such error separately
                    # return shape matches current y shape (N,)
                    return torch.zeros_like(y)

                # pick derivative along dimension cur_dim:
                # grads[cur_dim] picks one row => shape is (N,)
                # so y stays shape (N,) after each update
                y = grads[cur_dim]
        return y

    def error_calculation(self, exact_u_fun, exact_p_fun, nb_pts_time=11, nb_pts_spatial=2*126+1, error_multiplier=1):
        x = np.linspace(self.ori_x_lo, self.ori_x_hi, nb_pts_spatial)
        t = np.linspace(self.t_lo, self.t_hi, nb_pts_time)
        arr = np.array(np.meshgrid(*([x]*self.dim_in + [t]))).T.reshape(-1, self.dim_in + 1)
        arr[:, [-1, 0]] = arr[:, [0, -1]]
        arr = torch.tensor(arr, device=self.device, dtype=torch.get_default_dtype())
        error = []
        nn = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            nn.append(self(arr[cur:min(cur+batch_size, last)], coordinate=0, all_u=True).detach())
            cur += batch_size
        nn = torch.cat(nn, dim=0)

        # Lejay
        logging.info("The error as in Lejay is calculated as follows.")
        overall_error = 0
        for i in range(self.dim_in):
            error.append(error_multiplier * (nn[:, i] - exact_u_fun(arr.T, i)).reshape(nb_pts_time, -1) ** 2)
            overall_error += (error[-1])
        error.append(overall_error)
        for i in range(self.dim_in):
            logging.info(f"$\\hat{{e}}_{i}(t_k)$")
            self.latex_print(error[i].max(dim=1)[0])
        logging.info("$\\hat{e}(t_k)$")
        self.latex_print(error[-1].max(dim=1)[0])
        logging.info("\\hline")

        # erru
        logging.info("\nThe relative L2 error of u (erru) is calculated as follows.")
        denominator, numerator = 0, 0
        for i in range(self.dim_in):
            denominator += exact_u_fun(arr.T, i).reshape(nb_pts_time, -1) ** 2
            numerator += (nn[:, i] - exact_u_fun(arr.T, i)).reshape(nb_pts_time, -1) ** 2
        logging.info("erru($t_k$)")
        self.latex_print((numerator.mean(dim=-1)/denominator.mean(dim=-1)).sqrt())

        del nn
        torch.cuda.empty_cache()
        grad = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            xx = arr[cur:min(cur+batch_size, last)].detach().clone().requires_grad_(True)
            tmp = []
            for i in range(self.dim_in):
                tmp.append(
                    torch.autograd.grad(
                        self(xx, coordinate=i).sum(),
                        xx,
                    )[0][:, 1:].detach()
                )
            grad.append(torch.stack(tmp, dim=-1))
            cur += batch_size
        grad = torch.cat(grad, dim=0)

        # errgu
        logging.info("\nThe relative L2 error of gradient of u (errgu) is calculated as follows.")
        denominator, numerator = 0, 0
        xx = arr.detach().clone().requires_grad_(True)
        for i in range(self.dim_in):
            exact = torch.autograd.grad(
                exact_u_fun(xx.T, i).sum(),
                xx,
            )[0][:, 1:]
            denominator += exact.reshape(nb_pts_time, -1, self.dim_in) ** 2
            numerator += (exact - grad[:, :, i]).reshape(nb_pts_time, -1, self.dim_in) ** 2
        logging.info("errgu($t_k$)")
        self.latex_print((numerator.mean(dim=(1, 2))/denominator.mean(dim=(1, 2))).sqrt())

        # errdivu
        logging.info("\nThe absolute divergence of u (errdivu) is calculated as follows.")
        numerator = 0
        for i in range(self.dim_in):
            numerator += (grad[:, i, i]).reshape(nb_pts_time, -1)
        numerator = numerator**2
        logging.info("errdivu($t_k$)")
        self.latex_print(
            ((self.ori_x_hi - self.ori_x_lo)**self.dim_in * numerator.mean(dim=-1)).sqrt()
        )

        del grad, xx
        torch.cuda.empty_cache()
        arr = arr.reshape(nb_pts_time, -1, self.dim_in + 1)[-1].detach()
        nn = []
        cur, batch_size, last = 0, 100000, arr.shape[0]
        while cur < last:
            nn.append(
                self(
                    arr[cur:min(cur+batch_size, last), :],
                    coordinate=-1,
                ).detach()
            )
            cur += batch_size
        nn = torch.cat(nn, dim=0)

        # errp
        logging.info("\nThe relative L2 error of p (errp) is calculated as follows.")
        denominator = (exact_p_fun(arr.T) - exact_p_fun(arr.T).mean()) ** 2
        numerator = (
                            nn - nn.mean() - exact_p_fun(arr.T) + exact_p_fun(arr.T).mean()
                    ) ** 2
        logging.info("errp($t_k$)")
        logging.info(
            "& --- " * (nb_pts_time - 1)
            + f"& {(numerator.mean()/denominator.mean()).sqrt().item():.2E} \\\\"
        )

    def compare_with_exact(
        self,
        exact_fun,
        return_error=False,
        nb_points=100,
        show_plot=True,
        exclude_terminal=False,
        ylim=None,
    ):
        grid = np.linspace(self.ori_x_lo, self.ori_x_hi, nb_points)
        x_mid = (self.ori_x_lo + self.ori_x_hi) / 2
        grid_d_dim = np.concatenate((
            np.expand_dims(grid, axis=0),
            x_mid * np.ones((self.dim_in - 1, nb_points))
        ), axis=0)
        grid_d_dim_with_t = np.concatenate((self.t_lo * np.ones((1, nb_points)), grid_d_dim), axis=0)

        nn_input = grid_d_dim_with_t
        error = []
        for_range = self.dim_out
        for i in range(for_range):
            f = plt.figure()
            true = exact_fun(
                self.t_lo,
                grid_d_dim,
                self.t_hi,
                i
            )
            terminal = exact_fun(self.t_hi, grid_d_dim, self.t_hi, i)
            nn = (
                self(torch.tensor(
                    nn_input.T, device=self.device, dtype=torch.get_default_dtype()
                ), coordinate=i)
                .detach()
                .cpu()
                .numpy()
            )
            error.append(np.abs(true - nn).mean())
            plt.plot(grid, nn, label="NN")
            plt.plot(grid, true, label="True solution")
            if not exclude_terminal:
                plt.plot(grid, terminal, label="Terminal solution")
            plt.xlabel("$x_1$")
            plt.ylabel(f"$u_{i+1}$")
            plt.legend()
            if ylim is not None and i == 0:
                # only change ylim for u0
                plt.ylim(*ylim)
            f.savefig(
                f"{self.working_dir_full_path}/plot/dgm_u{i}_comparison_with_exact.png", bbox_inches="tight"
            )
            if show_plot:
                plt.show()
            plt.close()

            data = np.stack((grid, true, terminal, nn)).T
            np.savetxt(
                f"{self.working_dir_full_path}/data/dgm_u{i}_comparison_with_exact.csv",
                data,
                delimiter=",",
                header="x,true,terminal,branch",
                comments=""
            )
        if return_error:
            return np.array(error)
    
    def log_config(self):
        """
        Set up configuration for log files and mkdir.
        """
        if not os.path.isdir(self.working_dir_full_path):
            os.makedirs(self.working_dir_full_path)
            os.mkdir(f"{self.working_dir_full_path}/plot")
            os.mkdir(f"{self.working_dir_full_path}/data")
        formatter = "%(asctime)s | %(name)s |  %(levelname)s: %(message)s"
        logging.getLogger().handlers = []  # clear previous loggers if any
        logging.basicConfig(
            filename=f"{self.working_dir_full_path}/run.log",
            filemode="w",
            level=logging.DEBUG,
            format=formatter,
        )
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
        logging.info(f"Logs are saved in {os.path.abspath(self.working_dir_full_path)}")
        logging.debug(f"Current configuration: {self.__dict__}")

    @staticmethod
    def latex_print(tensor):
        mess = ""
        for i in tensor[:-1]:
            mess += f"& {i.item():.2E} "
        mess += "& --- \\\\"
        logging.info(mess)


if __name__ == "__main__":
    # configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    """
    Zeta map refers to u. 
    Essentially the forward function is composed of 1 functions: u.
    So you can imagine for example in a usual sense we have 
    u(t,x,y) = [u_1(t,x,y), u_2(t,x,y)]
    so if we want to get u_1, we set coordinate=0
    if we want to get u_2, we set coordinate=1
    """
    zeta_map = np.array([0, 0, 0])
    """
    To accompany derivative map you have to look at it together with zeta_map.
    For example, derivative map with an entry 
    [[2,0],[0,2],[2,0],[0,2]] and corresponding zeta_map entry [0,0,1,1]
    means \\partial_{xx} u_1 and \\partial_{yy} u_1 and \\partial_{xx} u_2 and \\partial_{yy} u_2
    """
    # deriv_map is n x d array defining lambda_1, ..., lambda_n
    deriv_map = np.array(
        [
            [0],  # u
            [2],  # \partial_{xx} u
        ]
    )
    _, dim = deriv_map.shape # dim = 2

    def f_example(y):
        """
        y has shape (2, N), with rows:
        y[0]=u, y[1]=∂xx u
        dim=1. 
        """
        f = y[1] + y[0] - y[0]**3 # f(u) = ∂xx u + u - u^3
        return f

    def phi_example(x, total_dim, coordinate):
        output = 1
        for d in range(total_dim):
            output *= torch.sin(math.pi * x[d])
        return output
        
    def psi_example(x, total_dim, coordinate):
        output = -1
        for d in range(total_dim):
            output *= torch.sin(math.pi * x[d])
        return output
    
    def x_low_original_boundary_function_example(tx, total_dim, coordinate):
        t = tx[0]
        x = tx[1]
        return torch.zeros_like(t)

    def x_high_original_boundary_function_example(tx, total_dim, coordinate):
        t = tx[0]
        x = tx[1]
        return torch.zeros_like(t)


    # boundary_fun=None & overtrain_rate=.1
    # boundary_fun=given & overtrain_rate=.0
    # initialize model and training
    model = DGMNet(
        dgm_f_fun=f_example,
        phi_fun=phi_example,
        psi_fun=psi_example,
        dgm_deriv_map=deriv_map,
        dgm_zeta_map=zeta_map,
        dt_order=2,
        t_lo=0.0,
        t_hi=2.0,
        tx_to_tx_init_ratio=5,
        x_lo_original=0.0,
        x_hi_original=1.0,
        overtrain_rate=2.0,
        x_low_original_boundary_fun=x_low_original_boundary_function_example,
        x_high_original_boundary_fun=x_high_original_boundary_function_example,
        device=device,
        verbose=True,
        save_as_tmp=True,
        dgm_nb_states=2000,
        epochs=5000,
        dgm_lr=1e-3,
        weight_decay=1e-6,
        directory=None
    )
    model.train_and_eval(debug_mode=False)

    # Build the requested evaluation grid:
    # t = 0.00, 0.05, ..., 1.00 and x = 0.00, 0.01, ..., 1.00.
    t_values = np.arange(0.0, 1.0 + 1e-12, 0.05, dtype=np.float32)
    x_values = np.arange(0.0, 1.0 + 1e-12, 0.01, dtype=np.float32)

    # Each row corresponds to one t in t_values, each column corresponds to one x in x_values.
    predictions = np.zeros((len(t_values), len(x_values)), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_values, device=device)
        for row_idx, t_val in enumerate(t_values):
            t_tensor = torch.full((len(x_values),), float(t_val), device=device)
            # Model input is (t, x) for dim_in=1.
            nn_input = torch.stack((t_tensor, x_tensor), dim=1)
            predictions[row_idx, :] = model(nn_input, coordinate=0).detach().cpu().numpy()

    # Save CSV under results/.
    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", "predictions_d1.csv")
    header = ",".join([f"{x:.2f}" for x in x_values])
    np.savetxt(results_path, predictions, delimiter=",", header=header, comments="")
    print(f"Saved prediction grid to {results_path}")
