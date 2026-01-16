# Simulation 03

For Simulation 03, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z,t)-\frac{\partial^2 u}{\partial z^2}(z,t) = \frac{3}{2}(u(z,t))^2 + 2(u(z,t))^3$

$u(z,0) = \phi(z) = \frac{4}{z^2-4}$

$\frac{\partial u}{\partial t}(z,0) = \psi(z) = -\frac{8\sqrt{2}z}{(z^2-4)^2}$

The analytical solution is:

$u(z,t) = \frac{4}{(z+\sqrt{2}t)^2-4}$

We set $c=1$, $z=9+0i$, $\lambda = 1.0$, and sample $t \in [0,5.0]$.
