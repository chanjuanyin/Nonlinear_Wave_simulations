# Simulation 07

For Simulation 07, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z_1,z_2,t)-(i)^2\Biggl[\frac{\partial^2 u}{\partial z_1^2}(z_1,z_2,t)+\frac{\partial^2 u}{\partial z_2^2}(z_1,z_2,t)\Biggl] = -u(z_1,z_2,t) - (u(z_1,z_2,t))^3$

$u(z_1,z_2,0) = \phi(z_1,z_2) = \sin(\pi z_1)\sin(\pi z_2)$

$\frac{\partial u}{\partial t}(z_1,z_2,0) = \psi(z_1,z_2) = -\sin(\pi z_1)\sin(\pi z_2)$

We set $c=i$, $t=0.5$, $\lambda = 0.25$, and vary $(z_1, z_2)$ in $\big(\mathbb{R}^2 \cap [0.00, 1.00] \times [0.00, 1.00]\big)$ over a $101 \times 101$ grid.
