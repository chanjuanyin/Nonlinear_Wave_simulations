# Simulation 09

For Simulation 09, the wave equation is:

$\frac{\partial^2 u}{\partial t^2}(z_1,z_2,z_3,t)-(i)^2\Biggl[\frac{\partial^2 u}{\partial z_1^2}(z_1,z_2,z_3,t)+\frac{\partial^2 u}{\partial z_2^2}(z_1,z_2,z_3,t)+\frac{\partial^2 u}{\partial z_3^2}(z_1,z_2,z_3,t)\Biggl] = -u(z_1,z_2,z_3,t) + (u(z_1,z_2,z_3,t))^3$

$u(z_1,z_2,z_3,0) = \phi(z_1,z_2,z_3) = \tanh\Big(\frac{i}{3\sqrt{2}}(z_1+z_2+z_3)\Big)$

$\frac{\partial u}{\partial t}(z_1,z_2,z_3,0) = \psi(z_1,z_2,z_3) = -\sqrt{\frac{2}{3}}sech^2\Big(\frac{i}{3\sqrt{2}}(z_1+z_2+z_3)\Big)$

The analytical solution is:

$u(z_1,z_2,z_3,t) = \tanh\Big(\frac{1}{\sqrt{6}}\big(\frac{i}{\sqrt{3}}(z_1+z_2+z_3)-2t\big)\Big)$

We set $c=i$, $z_1=-1+0i$, $z_2=-1+0i$, $z_3=-1+0i$, $\lambda = 0.3038419$, and sample $t \in [0,3.0]$ in steps of $0.1$ (31 time points).
