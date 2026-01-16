# Simulation 13

For Simulation 13, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z_1,z_2,t)-(i)^2\Big[\frac{\partial^2 u}{\partial z^2}(z,t)\Big] = -u(z,t) + (u(z,t))^3$

$u(z,0) = \phi(z) = \sin(\pi z)$

$\frac{\partial u}{\partial t}(z,0) = \psi(z) = -\sin(\pi z)$

We set $c=i$, $\lambda = 0.25$, and sample $z \in \big(\mathbb{R} \cap [0.00, 1.00] \big)$ on a uniform 1-D grid of 101 points (same as Simulation 12 and 13).

Unlike Simulation 13, which samples $t \in [0,3]$ in steps of $0.1$, here we sample $t \in [0,3]$ in steps of $0.05$ (61 time points).
