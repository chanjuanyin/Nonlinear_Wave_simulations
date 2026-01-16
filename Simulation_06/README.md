# Simulation 06

For Simulation 06, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z_1,z_2,t)-(i)^2\Biggl[\frac{\partial^2 u}{\partial z_1^2}(z_1,z_2,t)+\frac{\partial^2 u}{\partial z_2^2}(z_1,z_2,t)\Biggl] = -u(z_1,z_2,t) + (u(z_1,z_2,t))^3$

$u(z_1,z_2,0) = \phi(z_1,z_2) = \tanh\Big(\frac{i}{2\sqrt{3}}(z_1+z_2)\Big)$

$\frac{\partial u}{\partial t}(z_1,z_2,0) = \psi(z_1,z_2) = -\sqrt{\frac{2}{3}}sech^2\Big(\frac{i}{2\sqrt{3}}(z_1+z_2)\Big)$

The analytical solution is:

$u(z_1,z_2,t) = \tanh\Big(\frac{1}{\sqrt{6}}\big(\frac{i}{\sqrt{2}}(z_1+z_2)-2t\big)\Big)$

We set $c=i$, $z_1=-1+0i$, $z_2=-1+0i$, $\lambda = 0.3495487$, and sample $t \in [0,3.0]$ in steps of $0.1$ (31 time points).
