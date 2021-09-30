#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.misc import derivative
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline as IUS, interp1d
from scipy.optimize import fsolve
from time import time

rc('text', usetex = True)
# %%
#Evolution of scalar field
kp = 0.05
m = 10**-5

def V(phi):
    return 1/2*m**2*phi**2

def V_phi(phi):
    return derivative(V, phi, dx = 10**-6)

N_eval = np.linspace(0, 68, 10**6)
phi_i = 18
ei = 10**-4
phi_Ni = -(2*ei)**0.5

def solver(N, X):
    phi, g = X
    return [g, (g**2/2 - 3)*(V_phi(phi)/V(phi) + g)]

X0 = [phi_i, phi_Ni]
sol = solve_ivp(solver, (0,68), X0, method = 'BDF', t_eval = N_eval)
phi = sol.y[0]
phi_N = sol.y[1]
N = sol.t

plt.plot(N, phi)
plt.xlabel("$\phi$")
plt.ylabel("N")
plt.title("$\phi(N)$")
plt.show()

plt.plot(N, phi_N)
plt.xlabel("$\phi$")
plt.ylabel("N")
plt.title("$\phi'(N)$")
plt.show()

plt.plot(phi, phi_N)
plt.xlabel("$\phi$")
plt.ylabel("$\phi'$")
plt.title("Phase plot")
plt.show()

phi = IUS(N, phi)
phi_N = IUS(N, phi_N)
# %%
#Evolution of Hubble scale

def H(N):
    return (2*V(phi(N))/(6 - (phi_N(N))**2))**0.5

def H_N(N):
    return derivative(H, N, dx = 10**-6)

def infl_end(N):
    return phi_N(N)**2/2 - 1

plt.plot(N, H(N), label = "$H(N)$")
plt.legend()
plt.show()

# phi_NN = phi_N.derivative()
# end = fsolve(infl_end, [68])
# N_end = end[0]
# print(N_end)
N_end = 68
ai = kp/(np.exp(N_end - 56.2)*H(N_end - 56.2))

def a(N):
    return ai*np.exp(N)

def z(N):
    return a(N)*phi_N(N)

def z_N(N):
    return derivative(z, N, dx = 10**-3)

def z_NN(N):
    return derivative(z, N, dx = 10**-3, n = 2)

def mus2(N):
    return (z_NN(N)/z(N) + z_N(N)/z(N) + z_N(N)*H_N(N)/(z(N)*H(N)))*(a(N)*H(N))**2

def mut2(N):
    return (2 + H_N(N)/H(N))*(a(N)*H(N))**2

plt.plot(N, np.sqrt(np.abs(mus2(N))))
plt.yscale('log')
plt.show()
#%%
#Evaluating the scalar power spectrum

def N_finder_s(N, k, scale):
    return scale - k/np.sqrt(np.abs(mus2(N)))

Nic = fsolve(N_finder_s, [1], args = (kp, 10**2))
Nshs = fsolve(N_finder_s, [1], args = (kp, 10**-2))
Nic = Nic[0]
Nshs = Nshs[0]
print(Nic)
print(Nshs)
N_eval = np.linspace(Nshs - 1, Nshs + 1, 1000)

def kp_power_spectra_solver(N, X):
    R, pi_s = X
    return [pi_s/(a(N)*H(N)*z(N)**2), -kp**2*z(N)**2*R/(a(N)*H(N))]

R_i = 1/(z(Nic)*(2*kp)**0.5)
pi_s_i = -1j*z(Nic)*(kp/2)**0.5 - a(Nic)*H(Nic)*z_N(Nic)/((2*kp)**0.5)
print(R_i)
print(pi_s_i)
sol = solve_ivp(kp_power_spectra_solver, (Nic, Nshs), [R_i, pi_s_i], t_eval = [Nshs])
R_p = sol.y[0]
#R = IUS(N_eval, R)
print("Pivot scale")
print(kp)
print(kp**3/(2*np.pi**2)*(np.abs(R_p[0]))**2)

def sps_solver(N, X, k):
    R, pi_s = X
    return [pi_s/(a(N)*H(N)*z(N)**2), -k**2*z(N)**2*R/(a(N)*H(N))]

y_list = np.arange(-5.0, 20.0, 1)
k_list = 10**y_list
sps = []
start_time = time()
for k in k_list:
    Nic = fsolve(N_finder_s, [1], args = (k, 10**2))
    Nshs = fsolve(N_finder_s, [1], args = (k, 10**-5))
    Nic = Nic[0]
    Nshs = Nshs[0]
    N_eval = np.linspace(Nshs - 1, Nshs + 1, 1000)

    R_i = 1/(z(Nic)*(2*k)**0.5)
    pi_s_i = -1j*z(Nic)*(k/2)**0.5 - a(Nic)*H(Nic)*z_N(Nic)/((2*k)**0.5)

    sol = solve_ivp(sps_solver, (Nic, Nshs), [R_i, pi_s_i], t_eval = [Nshs], args = (k,))
    R = sol.y[0]
    #R = IUS(N_eval, R)
    sps_k = k**3/(2*np.pi**2)*(np.abs(R[0]))**2
    sps.append(sps_k)
    print(k)
    print(sps_k)
    print("solved")

sps = np.asarray(sps)
print("done")
end_time = time()
print(f"{end_time - start_time} seconds")

# %%
plt.plot(k_list, sps)
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
#Evaluating the tensor power spectrum

def N_finder_t(N, k, scale):
    return scale - k/np.sqrt(np.abs(mus2(N)))

def tps_solver(N, X, k):
    h, pi_t = X
    return [pi_t/(a(N)**3*H(N)), -k**2*a(N)*h/H(N)]

tps = []
start_time = time()
for k in k_list:
    Nic = fsolve(N_finder_t, [1], args = (k, 10**2))
    Nshs = fsolve(N_finder_t, [1], args = (k, 10**-5))
    Nic = Nic[0]
    Nshs = Nshs[0]
    N_eval = np.linspace(Nshs - 1, Nshs + 1, 1000)

    h_i = 1/(a(Nic)*(2*k)**0.5)
    pi_t_i = -1j*a(Nic)*(k/2)**0.5 - a(Nic)**2*H(Nic)/((2*k)**0.5)

    sol = solve_ivp(tps_solver, (Nic, Nshs), [h_i, pi_t_i], t_eval = [Nshs], args = (k,))
    h = sol.y[0]
    tps_k = 8*k**3/(2*np.pi**2)*(np.abs(h[0]))**2
    tps.append(tps_k)
    print(k)
    print(tps_k)
    print("solved")

tps = np.asarray(tps)
print("done")
end_time = time()
print(f"{end_time - start_time} seconds")
# %%
plt.plot(k_list, tps)
plt.xscale('log')
plt.yscale('log')
plt.show()
# %%
