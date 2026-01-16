# Simulation 05

For Simulation 05, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z_1,z_2,t)-\Big[\frac{\partial^2 u}{\partial z_1^2}(z_1,z_2,t)+\frac{\partial^2 u}{\partial z_2^2}(z_1,z_2,t)\Big] = (u(z_1,z_2,t))^2$

$u(z_1,z_2,0) = \phi(z_1,z_2) = 6(z_1+z_2)^{-2}$

$\frac{\partial u}{\partial t}(z_1,z_2,0) = \psi(z_1,z_2) = -12\sqrt{3}(z_1+z_2)^{-3}$

The analytical solution is:

$u(z_1,z_2,t) = 6(z_1+z_2+\sqrt{3}t)^{-2}$

We set $c=1$, $z_1=4+0i$, $z_2=4+0i$, $\lambda = 1.0$, and sample $t \in [0,4.0]$.
