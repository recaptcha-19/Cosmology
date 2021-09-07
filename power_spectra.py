#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.misc import derivative
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline as IUS, interp1d
from scipy.optimize import fsolve

rc('text', usetex = True)
#%%
#Evolution of the scalar field
kp = 0.05

def V(phi):
    a = 1
    b = 1.4349
    V0 = 4*10**-10
    v = np.sqrt(0.108)
    x = phi/v
    return V0*(6*x**2 - 4*a*x**3 + 3*x**4)/(1 + b*x**2)**2

def V_phi(phi):
    return derivative(V, phi, dx = 10**-6)

N_eval = np.linspace(0, 63.2, 10**6)
phi_i = 3.614
ei = 10**-4
phi_Ni = -np.sqrt(2*ei)

def solver(N, X):
    phi, g = X
    return [g, (g**2/2 - 3)*(V_phi(phi)/V(phi) + g)]

def HSR_1_check(N, X):
    phi, g = X
    return g**2/2 - 1

HSR_1_check.terminal = False

X0 = [phi_i, phi_Ni]
sol = solve_ivp(solver, (0,66), X0, method = 'BDF', t_eval = np.linspace(0, 62.8, 10**6))
phi = sol.y[0]
phi_N = sol.y[1]
N = sol.t

plt.plot(phi, phi_N)
plt.xlabel("$\phi$")
plt.ylabel("$\phi'$")
plt.title("Phase plot")
plt.show()

# %%
#Evolution of Hubble scale

H = (2*V(phi)/(6 - phi_N**2))**0.5
plt.plot(N, H, label = "$H(N)$")
plt.legend()
plt.show()

phi_N_2_check = interp1d(phi_N**2, N)
phi_N = IUS(N, phi_N)
N_end = phi_N_2_check(1)
print(N_end)
H = IUS(N, H)
H_N = H.derivative()
ai = kp/np.exp(N_end - 50)*H(N_end - 50)

a = IUS(N, ai*np.exp(N))
phi_NN = phi_N.derivative()
z = IUS(N, ai*np.exp(N)*(phi_NN(N)))
z_N = z.derivative()
z_NN = z.derivative(2)
mus2 = IUS(N, (a(N)*H(N))**2*(z_NN(N)/z(N) + z_N(N)/z(N) + z_N(N)*H_N(N)/(z(N)*H(N))))
mut2 = IUS(N, (2 + H_N(N)/H(N))*(a(N)*H(N))**2)

# %%
#Evaluating the scalar power spectrum

plt.plot(N, np.sqrt(np.abs(mus2(N))))
plt.yscale('log')
plt.show()

Ni_check = IUS(10**2*np.sqrt(np.abs(mut2(N))) - 10**-3, N)
print(Ni_check(0))
# N_sol = fsolve(lambda N: 10**2*np.sqrt(np.abs(mut2(N))) - 10**-3, 1)
# plt.plot(N, 10**2*np.sqrt(np.abs(mus2(N))) - 10**-3)
# plt.yscale('log')
# plt.show()
# print(N_sol)
# %%
