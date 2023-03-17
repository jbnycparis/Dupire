from __future__ import print_function
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

#
# solving Dupire's PDE using 'flat' boundary conditions (i.e. u''=0)
#

#params
S0 = 100.
sigma = 0.2
r = 0.05
d = 0.02
T = 1.

def BS_reduced(x,T,sig):#Black-Scholes for C/S0 at t=0 in reduced variables (x,T); x = ln(S0/K)
    d1 = (x + (r - d + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = (x + (r - d - 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    val = np.exp(-d*T) * scipy.stats.norm.cdf(d1) - np.exp(-x-r*T) * scipy.stats.norm.cdf(d2)
    return val

# discretization params
xmin = -3.
xmax = 3.
nx = 100
nt = 100
dt = T / nt
dx = (xmax - xmin) / (nx - 1)

x = np.zeros(nx)
for k in range(nx):
    x[k] = xmin + k * dx
old_u = np.zeros(nx) # u is the 'reduced' option price: C(K,T;S0,t=0) / S0
for k in range(nx):
    old_u[k] = np.maximum(1. - np.exp(-x[k]),0.) # payoff (1 - exp(-x))_+
new_u = np.zeros(nx)
rhs = np.zeros(nx)
# LHS matrix: b diagonal; a lower diag; c upper diag
b = (2./dt) + d + sigma**2 / (dx**2)
a = -sigma**2 / (2. * dx**2) + (r - d + 0.5 * sigma**2) / (2. * dx)
c = -sigma**2 / (2. * dx**2) - (r - d + 0.5 * sigma**2) / (2. * dx)
LHS_banded = np.zeros((3,nx))
LHS_banded[0,2:] = c
LHS_banded[2,:-2] = a
LHS_banded[1,1:-1] = b
LHS_banded[1,0] = (2./dt) + d
LHS_banded[1,-1] = (2./dt) + d
b_rhs = (2./dt) - d - sigma**2 / (dx**2)
for t in range(nt):
    # apply RHS matrix
    rhs[0] = ((2./dt) - d) * old_u[0]
    rhs[nx-1] = ((2./dt) - d) * old_u[nx-1]
    for k in range(1,nx-1):
        rhs[k] = b_rhs * old_u[k] - a * old_u[k-1] - c * old_u[k+1]
    new_u = solve_banded((1, 1), LHS_banded, rhs)
    (new_u,old_u) = (old_u,new_u) # swap

u_theo = np.zeros(nx)
for k in range(nx):
    u_theo[k] = BS_reduced(x[k],T,sigma)
error = new_u - u_theo

# back to actual option prices
x_strikes = np.zeros(nx)
price_theo_grid = np.zeros(nx)
price_num_grid = np.zeros(nx)
for k in range(nx):
    x_strikes[k] = S0 * np.exp(-x[k])
    price_theo_grid[k] = S0 * BS_reduced(x[k],T,sigma)
    price_num_grid[k] = S0 * new_u[k]

strikes = np.arange(50,200,10) # useful range
nstrikes = len(strikes)
price_theo = np.zeros(nstrikes)
price_num = np.zeros(nstrikes)
for k in range(nstrikes):
    price_theo[k] = S0 * BS_reduced(np.log(S0/strikes[k]),T,sigma)
    price_num[k] = np.interp(strikes[k], x_strikes[::-1], price_num_grid[::-1])

plt.figure()
plt.plot(x,u_theo)
plt.plot(x,new_u)
plt.legend(['theo','num'])
plt.title("Dupire (x variable)")

plt.figure()
plt.plot(x,error)
#plt.ylim([-1e-3,1e-3])
plt.title('Error: num Dupire - theo Dupire (x variable)')

plt.figure()
plt.plot(strikes,price_theo)
plt.plot(strikes,price_num)
plt.legend(['theo','num'])
plt.title("Dupire (function of strike)")

plt.figure()
plt.plot(strikes,price_num - price_theo)
plt.title("Error (function of strike)")

plt.show()


















