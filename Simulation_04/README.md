# Simulation 04

For Simulation 04, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z,t)-(i)^2\Big[\frac{\partial^2 u}{\partial z^2}(z,t)\Big] = -u(z,t) + (u(z,t))^3$

$u(z,0) = \phi(z) = \tanh\Big(\frac{i}{\sqrt{6}}z\Big)$

$\frac{\partial u}{\partial t}(z,0) = \psi(z) = -\sqrt{\frac{2}{3}}sech^2\Big(\frac{i}{\sqrt{6}}z\Big)$

The analytical solution is:

$u(z,t) = \tanh\Big(\frac{1}{\sqrt{6}}(iz-2t)\Big)$

We set $c=i$, $z=-1+0i$, $\lambda = 0.4185722$, and sample $t \in [0,3.0]$.
