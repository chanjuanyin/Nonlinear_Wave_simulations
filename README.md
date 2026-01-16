# Nonlinear_Wave

This repository contains numerical experiments for “Probabilistic solution and quantitative estimates for wave equations with polynomial nonlinearities.”

- Simulations 4, 6, 9 → section 7.1 experiments.
- Simulations 7, 10 → section 7.2 experiments.
- Simulation 14 → comparison with Mathematica, Galerkin, and an implicit scheme (section 7.3).
- All runs use $10^7$ samples with multithreaded C++.

## Simulations 4, 6, 9

Wave equation: $\partial_{tt}u + \Delta u = -u + u^3$

| $d$ | $\phi(z)$                                         | $\psi(z)$                                                              | analytical solution                                                                 |
|-----|---------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 1   | $\tanh\Big(\frac{i}{\sqrt{6}}z\Big)$            | $-\sqrt{\frac{2}{3}}sech^2\Big(\frac{i}{\sqrt{6}}z\Big)$            | $\tanh\Big(\frac{1}{\sqrt{6}}(iz-2t)\Big)$                                         |
| 2   | $\tanh\Big(\frac{i}{2\sqrt{3}}(z_1+z_2)\Big)$   | $-\sqrt{\frac{2}{3}}sech^2\Big(\frac{i}{2\sqrt{3}}(z_1+z_2)\Big)$   | $\tanh\Big(\frac{1}{\sqrt{6}}\big(\frac{i}{\sqrt{2}}(z_1+z_2)-2t\big)\Big)$                |
| 3   | $\tanh\Big(\frac{i}{3\sqrt{2}}(z_1+z_2+z_3)\Big)$ | $-\sqrt{\frac{2}{3}}sech^2\Big(\frac{i}{3\sqrt{2}}(z_1+z_2+z_3)\Big)$ | $\tanh\Big(\frac{1}{\sqrt{6}}\big(\frac{i}{\sqrt{3}}(z_1+z_2+z_3)-2t\big)\Big)$            |

Evaluation points:

| $d$ | $z$                         |
|-----|-----------------------------|
| 1   | $-1+0i$                     |
| 2   | $(-1+0i,\,-1+0i)$           |
| 3   | $(-1+0i,\,-1+0i,\,-1+0i)$   |

Sample $t \in [0,3]$ in steps of $0.1$ (31 time points).

## Simulations 7, 10

Wave equation ($d=2$): $\partial_{tt}u - c^2\Delta u = -u + u^3$

Initial data: $u(z,0)=\sin(\pi z_1)\sin(\pi z_2)$, $\partial_t u(z,0)=-\sin(\pi z_1)\sin(\pi z_2)$.

- Simulation 7: $c=i$
- Simulation 10: $c=1$

Both use $t=0.5$, $\lambda=0.25$, sampling $(z_1,z_2)\in[0,1]^2$ on a $101\times101$ grid.

## Simulations 11–14

Wave equation ($d=1$): $\partial_{tt}u - (i)^2\Delta u = -u + u^3$

Initial data: $u(z,0)=\sin(\pi z)$, $\partial_t u(z,0)=-\sin(\pi z)$.

Base setup: $c=i$, $\lambda=0.25$, $z\in[0,1]$ on a uniform 1-D grid of 101 points.

- Simulation 14: $t \in [0,3]$ in steps of $0.05$ (61 time points) ⇒ $101 \times 61$ space–time samples.
- Simulation 13: same as 14, but $t$ step $0.1$ (31 time points).
- Simulation 12: same as 14, but fixed $t=0.5$.
- Simulation 11: same as 12, but $\psi(z)=\sin(\pi z)$.

## Simulations 1, 2, 3, 5, 8

These exploratory runs use $t$ sampled in $0.1$ increments.

| Simulation | $d$ | wave equation                              | $\phi(z)$               | $\psi(z)$                          | analytical solution                 | evaluation $z$         | $\lambda$ | $t$ range |
|------------|-----|--------------------------------------------|-------------------------|------------------------------------|--------------------------------------|------------------------|-----------|-----------|
| 01         | 1   | $\partial_{tt}u - \Delta u = u^2$          | $6z^{-2}$               | $-12\sqrt{2}z^{-3}$                | $6(z+\sqrt{2}t)^{-2}$                | $3+0i$                 | 0.25      | $[0,2]$   |
| 02         | 1   | $\partial_{tt}u - \Delta u = u^3$          | $\sqrt{2}z^{-1}$        | $-2z^{-2}$                         | $\sqrt{2}(z+\sqrt{2}t)^{-1}$         | $6+0i$                 | 0.25      | $[0,5]$   |
| 03         | 1   | $\partial_{tt}u-\Delta u=\frac{3}{2}u^2+2u^3$ | $\frac{4}{z^2-4}$       | $-\frac{8\sqrt{2}z}{(z^2-4)^2}$    | $\frac{4}{(z+\sqrt{2}t)^2-4}$        | $9+0i$                 | 1         | $[0,5]$   |
| 05         | 2   | $\partial_{tt}u - \Delta u = u^2$          | $6(z_1+z_2)^{-2}$       | $-12\sqrt{3}(z_1+z_2)^{-3}$        | $6(z_1+z_2+\sqrt{3}t)^{-2}$          | $(4+0i,\,4+0i)$        | 1         | $[0,4]$   |
| 08         | 3   | $\partial_{tt}u - \Delta u = u^2$          | $6(z_1+z_2+z_3)^{-2}$   | $-24(z_1+z_2+z_3)^{-3}$            | $6(z_1+z_2+z_3+2t)^{-2}$             | $(4+0i,\,4+0i,\,4+0i)$ | 1         | $[0,4]$   |