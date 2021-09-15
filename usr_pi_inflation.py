#%%
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.misc import derivative
rc('text', usetex = True)

#%%
#USR1
a = 1
b = 1.4349
V0 = 4*10**-10
v = np.sqrt(0.108)

def V_USR1(phi):
    x = phi/v
    return V0*(6*x**2 - 4*a*x**3 + 3*x**4)/(1 + b*x**2)**2

def V_phi_USR1(phi):
    return derivative(V_USR1, phi, dx = 1e-6) 

def V_phi_phi_USR1(phi):
    return derivative(V_phi_USR1, phi, dx = 1e-6) 

phi = np.linspace(-1, 6, 1000)
plt.plot(phi, V_USR1(phi)/V0, label = "$V$")
plt.legend()
plt.show()

plt.plot(phi, V_phi_USR1(phi)/V0, label = "$V_\phi$")
plt.legend()
plt.show()

#%%
N = np.linspace(0, 66, 10**6)
phi_i_USR1 = 3.614
ei = 10**-4
phi_Ni_USR1 = -np.sqrt(2*ei)

def USR1(N, X):
    phi, g = X
    return [g, (g**2/2 - 3)*(V_phi_USR1(phi)/V_USR1(phi) + g)]
    #return [g, (v*g**2/2 - 3/v)*((1/v**2)*(g + 12*x - 12*a*x**2 + 12*x**2)/(6*x**2 - 4*a*x**3 + 3*x**4) - (1/v)*4*b*x/(1 + b*x**2))]

X0 = [phi_i_USR1, phi_Ni_USR1]
sol = solve_ivp(USR1, (0,66), X0, method = 'Radau')
phi_USR1 = sol.y[0]
phi_N_USR1 = sol.y[1]
N_USR1 = sol.t
#print(phi_USR1)
plt.plot(phi_USR1[:100], phi_N_USR1[:100])
plt.xlabel("$\phi$")
plt.ylabel("$\phi'$")
plt.title("Phase plot for USR1")
plt.show()

phi_NN_USR1 = (phi_N_USR1**2/2 - 3)*(V_phi_USR1(phi_USR1)/V_USR1(phi_USR1) + phi_N_USR1)

e1_USR1 = phi_N_USR1**2/2
plt.plot(N_USR1[4:58], e1_USR1[4:58], label = "$\epsilon_1$")
plt.yscale('log')
plt.legend()
plt.show()

e2_USR1 = 2*phi_NN_USR1/phi_N_USR1
plt.plot(N_USR1[4:58], e2_USR1[4:58], label = "$\epsilon_2$")
#plt.yscale('log')
plt.legend()
plt.show()

phi_NN_factor = phi_NN_USR1*((1+6/phi_N_USR1**2)*(V_phi_USR1(phi_USR1)/V_USR1(phi_USR1)+phi_N_USR1) + (phi_N_USR1-6/phi_N_USR1)) 
other = (phi_N_USR1**2-6)*(V_phi_phi_USR1(phi_USR1)/V_USR1(phi_USR1)-(V_phi_USR1(phi_USR1)/V_USR1(phi_USR1))**2)
e3_USR1 = (phi_NN_factor + other)/e2_USR1
plt.plot(N_USR1[4:58], e3_USR1[4:58], label = "$\epsilon_3$")
plt.ylim(-100,100)
#plt.yscale('log')
plt.legend()
plt.show()
#%%
#USR2
V0 = 2*10**-10
A = 0.130383
f = 0.129576

def V_USR2(phi):
    return V0*(np.tanh(phi/np.sqrt(6)) + A*np.sin(1/f*np.tanh(phi/np.sqrt(6))))**2

def V_phi_USR2(phi):
    return derivative(V_USR2, phi, dx = 10**-9)

def V_phi_phi_USR2(phi):
    return derivative(V_USR2, phi, dx = 10**-4, n = 2) 

phi = np.linspace(-1, 6, 1000)
plt.plot(phi, V_USR2(phi)/V0, label = "$V$")
plt.legend()
plt.show()

plt.plot(phi, V_phi_USR2(phi)/V0, label = "$V_\phi$")
plt.legend()
plt.show()

# %%

def USR2(N, X):
    phi, g = X
    return [g, (g**2/2 - 3)*(V_phi_USR2(phi)/V_USR2(phi) + g)]

X = [6.1, 0.0141]
sol = solve_ivp(USR2, (0,66), X, method = 'LSODA', t_eval = np.linspace(0,58,10000), dense_output = True)
phi_ = sol.y[0]
phi_N_ = sol.y[1]
N_ = sol.t

X0 = [5, -1.4]
sol0 = solve_ivp(USR2, (0,66), X0, method = 'BDF', t_eval = np.linspace(0,65,10000), dense_output = True)
phi_0 = sol0.y[0]
phi_N_0 = sol0.y[1]
N_0 = sol0.t

X1 = [5, 1.4]
sol1 = solve_ivp(USR2, (0,66), X1, method = 'BDF', t_eval = np.linspace(0,20,10000), dense_output = True)
phi_1 = sol1.y[0]
phi_N_1 = sol1.y[1]
N_1 = sol1.t

X2 = [4, -1.4]
sol2 = solve_ivp(USR2, (0,66), X2, method = 'BDF', t_eval = np.linspace(0,15,10000), dense_output = True)
phi_2 = sol2.y[0]
phi_N_2 = sol2.y[1]
N_2 = sol1.t

X3 = [4, 1.4]
sol3 = solve_ivp(USR2, (0,66), X3, method = 'BDF', t_eval = np.linspace(0,65,10000), dense_output = True)
phi_3 = sol3.y[0]
phi_N_3 = sol3.y[1]
N_3 = sol3.t

X4 = [3, -1.4]
sol4 = solve_ivp(USR2, (0,66), X4, method = 'BDF', t_eval = np.linspace(0,10,10000), dense_output = True)
phi_4 = sol4.y[0]
phi_N_4 = sol4.y[1]
N_4 = sol4.t

X5 = [3, 1]
sol5 = solve_ivp(USR2, (0,66), X5, method = 'BDF', t_eval = np.linspace(0,64,10000), dense_output = True)
phi_5 = sol5.y[0]
phi_N_5 = sol5.y[1]
N_5 = sol5.t

X6 = [2, 1]
sol6 = solve_ivp(USR2, (0,66), X6, method = 'BDF', t_eval = np.linspace(0,3,10000), dense_output = True)
phi_6 = sol6.y[0]
phi_N_6 = sol6.y[1]
N_6 = sol6.t

X7 = [2, -1.4]
sol7 = solve_ivp(USR2, (0,66), X7, method = 'BDF', t_eval = np.linspace(0,3,10000), dense_output = True)
phi_7 = sol7.y[0]
phi_N_7 = sol7.y[1]
N_7 = sol7.t

X8 = [0.8, 0.4]
sol8 = solve_ivp(USR2, (0,66), X8, method = 'BDF', t_eval = np.linspace(0,2,10000), dense_output = True)
phi_8 = sol8.y[0]
phi_N_8 = sol8.y[1]
N_8 = sol8.t

X9 = [1, -1.4]
sol9 = solve_ivp(USR2, (0,66), X9, method = 'BDF', t_eval = np.linspace(0,3,10000), dense_output = True)
phi_9 = sol9.y[0]
phi_N_9 = sol9.y[1]
N_9 = sol9.t

plt.plot(phi_, phi_N_)
plt.plot(phi_0, phi_N_0)
plt.plot(phi_1, phi_N_1)
plt.plot(phi_2, phi_N_2)
plt.plot(phi_3, phi_N_3)
plt.plot(phi_4, phi_N_4)
plt.plot(phi_5, phi_N_5)
plt.plot(phi_6, phi_N_6)
plt.plot(phi_7, phi_N_7)
plt.plot(phi_8, phi_N_8)
plt.plot(phi_9, phi_N_9)
plt.xlabel("$\phi$")
plt.ylabel("$\phi'$")
plt.title("Phase plot for USR2")
plt.show()

phi_NN_ = (phi_N_**2/2 - 3)*(V_phi_USR2(phi_)/V_USR2(phi_) + phi_N_)

e1_USR2 = phi_N_**2/2
plt.plot(N_[500:], e1_USR2[500:], label = "$\epsilon_1$")
plt.yscale('log')
plt.legend()
plt.show()

e2_USR2 = 2*phi_NN_/phi_N_
plt.plot(N_[500:], e2_USR2[500:], label = "$\epsilon_2$")
#plt.yscale('log')
plt.legend()
plt.show()

phi_NN_factor = phi_NN_*((1+6/phi_N_**2)*(V_phi_USR2(phi_)/V_USR2(phi_)+phi_N_) + (phi_N_-6/phi_N_)) 
other = (phi_N_**2-6)*(V_phi_phi_USR2(phi_)/V_USR2(phi_)-(V_phi_USR2(phi_)/V_USR2(phi_))**2)
e3_USR2 = (phi_NN_factor + other)/e2_USR2

plt.plot(N_[500:], e3_USR2[500:], label = "$\epsilon_3$")
plt.ylim(-100,100)
#plt.yscale('log')
plt.legend()
plt.show()
# %%
fig = plt.figure()
gs = fig.add_gridspec(3, hspace = 0)
ax1, ax2, ax3 = gs.subplots(sharex = True)
fig.suptitle("Variation of $\epsilon_1$, $\epsilon_2$ and $\epsilon_3$ for USR1 and USR2")

ax1.set_yscale('log')
ax1.plot(N_USR1[4:58], e1_USR1[4:58], c = 'r')
ax1.plot(N_[500:], e1_USR2[500:], c = 'r', linestyle = 'dashed')
ax1.set_ylabel("$\epsilon_1$")

ax2.plot(N_USR1[4:58], e2_USR1[4:58], c = 'b')
ax2.plot(N_[500:], e2_USR2[500:], c = 'b', linestyle = 'dashed')
ax2.set_ylabel("$\epsilon_2$")

ax3.plot(N_USR1[4:58], e3_USR1[4:58], c = 'g')
ax3.plot(N_[500:], e3_USR2[500:], c = 'g', linestyle = 'dashed')
ax3.set_ylim(-100,100)
ax3.set_xlabel("N")
ax3.set_ylabel("$\epsilon_3$")
fig.savefig("slow_roll_params_USR.png")
plt.show()

#%%
#PI1
V0 = 8*10**-13
B = 0.5520

def V_PI1(phi):
    return V0*(1 + B*phi**4)

def V_phi_PI1(phi):
    return 4*V0*B*phi**3

def V_phi_phi_PI1(phi):
    return 12*V0*B*phi**2

phi = np.linspace(-1, 6, 1000)
plt.plot(phi, V_PI1(phi)/V0, label = "$V$")
plt.legend()
plt.show()

plt.plot(phi, V_phi_PI1(phi)/V0, label = "$V_\phi$")
plt.legend()
plt.show()

#%%

N = np.linspace(0, 70, 10**3)
phi_i_PI1 = 17
ei = 10**-4
phi_Ni_PI1 = np.sqrt(2*ei)

def PI1(N, X):
    phi, g = X
    return [g, (g**2/2 - 3)*(V_phi_PI1(phi)/V_PI1(phi) + g)]

X0 = [phi_i_PI1, phi_Ni_PI1]
sol = solve_ivp(PI1, (0,70), X0, method = 'LSODA', t_eval = N, dense_output = True)
phi_PI1 = sol.y[0]
phi_N_PI1 = sol.y[1]
N_PI1 = sol.t
#print(N_PI1)
plt.plot(phi_PI1, phi_N_PI1)
plt.xlabel("$\phi$")
plt.ylabel("$\phi'$")
plt.title("Phase plot for PI1")
plt.show()

#%%
phi_NN_PI1 = (phi_N_PI1**2/2 - 3)*(V_phi_PI1(phi_PI1)/V_PI1(phi_PI1) + phi_N_PI1)

e1_PI1 = phi_N_PI1**2/2
plt.plot(N_PI1[100:], e1_PI1[100:], label = "$\epsilon_1$")
plt.yscale('log')
plt.xlabel("$N$")
plt.ylabel("$\epsilon_1(N)$")
plt.legend()
plt.show()

e2_PI1 = 2*phi_NN_PI1/phi_N_PI1
plt.plot(N_PI1[100:], e2_PI1[100:], label = "$\epsilon_2$")
plt.xlabel("$N$")
plt.ylabel("$\epsilon_2(N)$")
plt.legend()
plt.show()

a = phi_N_PI1*phi_NN_PI1*(V_phi_PI1(phi_PI1)/V_PI1(phi_PI1) + phi_N_PI1)
b = phi_N_PI1**2/2 - 3
c = phi_N_PI1*(V_phi_phi_PI1(phi_PI1)/V_PI1(phi_PI1) - (V_phi_PI1(phi_PI1)/V_PI1(phi_PI1))**2) + phi_NN_PI1
phi_NNN_PI1 = a + b*c
e3_PI1 = phi_NNN_PI1/phi_NN_PI1 - phi_NN_PI1/phi_N_PI1
plt.plot(N_PI1[500:], e3_PI1[500:], label = "$\epsilon_3$")
plt.ylim(-100,100)
#plt.yscale('log')
plt.legend()
plt.show()
#%%
#PI2

m = 1.8*10**-6
phi_0 = 1.9777

def V_PI2(phi):
    return m**2*phi**2/2 - 2*m**2*phi**3/(3*phi_0) + m**2*phi**4/(4*phi_0**2)

def V_phi_PI2(phi):
    return m**2*phi - 2*m**2*phi**2/phi_0 + m**2*phi**3/phi_0**2

def V_phi_phi_PI2(phi):
    return m**2 - 4*m**2*phi/phi_0 + m**2/phi_0**2*(3*phi**2)

phi = np.linspace(-20, 20, 1000)
plt.plot(phi, V_PI1(phi)/V0, label = "$V$")
plt.legend()
plt.show()

plt.plot(phi, V_phi_PI1(phi)/V0, label = "$V_\phi$")
plt.legend()
plt.show()

#%%
N = np.linspace(0, 70, 10**4)
phi_i_PI2 = 20
ei = 10**-4
phi_Ni_PI2 = -np.sqrt(2*ei)

def PI2(N, X):
    phi, g = X
    return [g, (g**2/2 - 3)*(V_phi_PI2(phi)/V_PI2(phi) + g)]

X0 = [phi_i_PI2, phi_Ni_PI2]
sol = solve_ivp(PI2, (0,70), X0, method = 'LSODA', t_eval = N, dense_output = True)
phi_PI2 = sol.y[0]
phi_N_PI2 = sol.y[1]
N_PI2 = sol.t
#print(N_PI1)
plt.plot(phi_PI2, phi_N_PI2)
plt.xlabel("$\phi$")
plt.ylabel("$\phi'$")
plt.title("Phase plot for PI2")
plt.show()

#%%
phi_NN_PI2 = (phi_N_PI2**2/2 - 3)*(V_phi_PI2(phi_PI2)/V_PI2(phi_PI2) + phi_N_PI2)

e1_PI2 = phi_N_PI2**2/2
plt.plot(N_PI2[100:], e1_PI2[100:], label = "$\epsilon_1$")
plt.yscale('log')
plt.xlabel("$N$")
plt.ylabel("$\epsilon_1(N)$")
plt.legend()
plt.show()

e2_PI2 = 2*phi_NN_PI2/phi_N_PI2
plt.plot(N_PI2[100:], e2_PI2[100:], label = "$\epsilon_2$")
plt.xlabel("$N$")
plt.ylabel("$\epsilon_2(N)$")
plt.legend()
plt.show()

a = phi_N_PI2*phi_NN_PI2*(V_phi_PI2(phi_PI2)/V_PI2(phi_PI2) + phi_N_PI2)
b = phi_N_PI2**2/2 - 3
c = phi_N_PI2*(V_phi_phi_PI2(phi_PI2)/V_PI1(phi_PI2) - (V_phi_PI2(phi_PI2)/V_PI2(phi_PI2))**2) + phi_NN_PI2
phi_NNN_PI2 = a + b*c
e3_PI2 = phi_NNN_PI2/phi_NN_PI2 - phi_NN_PI2/phi_N_PI2
plt.plot(N_PI2[500:], e3_PI2[500:], label = "$\epsilon_3$")
plt.ylim(-100,100)
#plt.yscale('log')
plt.legend()
plt.show()

# %%
#PI3
V0 = 2.1*10**-10
c0 = 0.16401
c1 = 0.3
c2 = -1.426
c3 = 2.20313
a = 1

def V_PI3(phi):
    x = phi/np.sqrt(6*a)
    t = np.tanh(x)
    return V0*(c0 + c1*t + c2*t**2 + c3*t**3)**2

def V_phi_PI3(phi):
    return derivative(V_PI3, phi, dx = 10**-4)

def V_phi_phi_PI3(phi):
    return derivative(V_PI3, phi, dx = 10**-4, n = 2)

phi = np.linspace(-1, 6, 1000)
plt.plot(phi, V_PI3(phi)/V0, label = "$V$")
plt.legend()
plt.show()

plt.plot(phi, V_phi_PI3(phi)/V0, label = "$V_\phi$")
plt.legend()
plt.show()

# %%

def PI3(N, X):
    phi, g = X
    return [g, (g**2/2 - 3)*(V_phi_PI3(phi)/V_PI3(phi) + g)]

X = [7.4, 0.0447]
sol = solve_ivp(PI3, (0,68), X, method ='BDF', t_eval = np.linspace(0,62,10**6), dense_output = True)
phi_PI3 = sol.y[0]
phi_N_PI3 = sol.y[1]
N_PI3 = sol.t

X0 = [5, -1.4]
sol0 = solve_ivp(PI3, (0,68), X0, method = 'BDF', t_eval = np.linspace(0,16,10**6), dense_output = True)
phi_0 = sol0.y[0]
phi_N_0 = sol0.y[1]
N_0 = sol0.t

X1 = [5, 1.4]
sol1 = solve_ivp(PI3, (0,68), X1, method = 'BDF', t_eval = np.linspace(0,17,10**6), dense_output = True)
phi_1 = sol1.y[0]
phi_N_1 = sol1.y[1]
N_1 = sol1.t

X2 = [4, -1.4]
sol2 = solve_ivp(PI3, (0,68), X2, method = 'BDF', t_eval = np.linspace(0,14,10**6), dense_output = True)
phi_2 = sol2.y[0]
phi_N_2 = sol2.y[1]
N_2 = sol1.t

X3 = [4, 1.4]
sol3 = solve_ivp(PI3, (0,68), X3, method = 'BDF', t_eval = np.linspace(0,11,10**6), dense_output = True)
phi_3 = sol3.y[0]
phi_N_3 = sol3.y[1]
N_3 = sol3.t

X4 = [3, -1.4]
sol4 = solve_ivp(PI3, (0,68), X4, method = 'BDF', t_eval = np.linspace(0,9,10**6), dense_output = True)
phi_4 = sol4.y[0]
phi_N_4 = sol4.y[1]
N_4 = sol4.t

X5 = [3, 1.4]
sol5 = solve_ivp(PI3, (0,68), X5, method = 'BDF', t_eval = np.linspace(0,9,10**6), dense_output = True)
phi_5 = sol5.y[0]
phi_N_5 = sol5.y[1]
N_5 = sol5.t

X6 = [2, 1.4]
sol6 = solve_ivp(PI3, (0,66), X6, method = 'BDF', t_eval = np.linspace(0,3,10**6), dense_output = True)
phi_6 = sol6.y[0]
phi_N_6 = sol6.y[1]
N_6 = sol6.t

X7 = [2, -1.4]
sol7 = solve_ivp(PI3, (0,66), X7, method = 'BDF', t_eval = np.linspace(0,3,10**6), dense_output = True)
phi_7 = sol7.y[0]
phi_N_7 = sol7.y[1]
N_7 = sol7.t

X8 = [1, 1.4]
sol8 = solve_ivp(PI3, (0,66), X8, method = 'BDF', t_eval = np.linspace(0,2,10**6), dense_output = True)
phi_8 = sol8.y[0]
phi_N_8 = sol8.y[1]
N_8 = sol8.t

X9 = [1, -1.4]
sol9 = solve_ivp(PI3, (0,66), X9, method = 'BDF', t_eval = np.linspace(0,1,10000), dense_output = True)
phi_9 = sol9.y[0]
phi_N_9 = sol9.y[1]
N_9 = sol9.t

plt.plot(phi_PI3, phi_N_PI3)
plt.plot(phi_0, phi_N_0)
plt.plot(phi_1, phi_N_1)
plt.plot(phi_2, phi_N_2)
plt.plot(phi_3, phi_N_3)
plt.plot(phi_4, phi_N_4)
plt.plot(phi_5, phi_N_5)
plt.plot(phi_6, phi_N_6)
plt.plot(phi_7, phi_N_7)
plt.plot(phi_8, phi_N_8)
plt.plot(phi_9, phi_N_9)
plt.xlabel("$\phi$")
plt.ylabel("$\phi'$")
plt.title("Phase plot for PI3")
plt.show()

#%%
# X_ = [5, 0.0141]
# sol = solve_ivp(PI3, (0,19), X_, method = 'LSODA', t_eval = np.linspace(0,19,10**6), dense_output = True)
# phi_ = sol.y[0]
# phi_N_ = sol.y[1]
# N_ = sol.t
# plt.plot(phi_, phi_N_)
# plt.show()

phi_NN_PI3 = (phi_N_PI3**2/2 - 3)*(V_phi_PI3(phi_PI3)/V_PI3(phi_PI3) + phi_N_PI3)

e1_PI3 = phi_N_PI3**2/2
plt.plot(N_PI3[10000:], e1_PI3[10000:], label = "$\epsilon_1$")
plt.yscale('log')
plt.xlabel("$N$")
plt.ylabel("$\epsilon_1(N)$")
plt.legend()
plt.show()

e2_PI3 = 2*phi_NN_PI3/phi_N_PI3
plt.plot(N_PI3[10000:], e2_PI3[10000:], label = "$\epsilon_2$")
plt.xlabel("$N$")
plt.ylabel("$\epsilon_2(N)$")
plt.legend()
plt.show()

ius = IUS(N_PI3, e2_PI3).derivative()
d_e2_PI3_dN = ius(N_PI3)
e3_PI3 = d_e2_PI3_dN/e2_PI3
# a_term = phi_N_PI3**2 + V_phi_PI3(phi_PI3)/V_PI3(phi_PI3)*(phi_N_PI3/2 + 3/phi_N_PI3)
# b_term = (phi_N_PI3**3 - 6*phi_N_PI3)/(2*phi_NN_PI3)*(V_phi_phi_PI3(phi_PI3)/V_PI3(phi_PI3) - (V_phi_PI3(phi_PI3)/V_PI3(phi_PI3))**2)
# e3_PI3 = a_term + b_term
plt.plot(N_PI3[10000:], e3_PI3[10000:], label = "$\epsilon_3$")
plt.ylim(-100,100)
plt.xlabel("$N$")
plt.ylabel("$\epsilon_3(N)$")
plt.legend()
plt.show()
# %%
fig = plt.figure()
gs1 = fig.add_gridspec(3, hspace = 0)
ax1, ax2, ax3 = gs1.subplots(sharex = True)
fig.suptitle("Variation of $\epsilon_1$, $\epsilon_2$ and $\epsilon_3$ for PI1, PI2 and PI3")

ax1.set_yscale('log')
ax1.plot(N_PI1[100:], e1_PI1[100:], c = 'r')
ax1.plot(N_PI2[100:], e1_PI2[100:], c = 'r', linestyle = 'dashed')
ax1.plot(N_PI3[10000:], e1_PI3[10000:], c = 'r', linestyle = 'dotted')
ax1.set_ylabel("$\epsilon_1$")

ax2.plot(N_PI1[100:], e2_PI1[100:], c = 'b')
ax2.plot(N_PI2[100:], e2_PI2[100:], c = 'b', linestyle = 'dashed')
ax2.plot(N_PI3[10000:], e2_PI3[10000:], c = 'b', linestyle = 'dotted')
ax2.set_ylabel("$\epsilon_2$")

ax3.plot(N_PI1[100:], e3_PI1[100:], c = 'g')
ax3.plot(N_PI2[100:], e3_PI2[100:], c = 'g', linestyle = 'dashed')
ax3.plot(N_PI3[10000:], e3_PI3[10000:], c = 'g', linestyle = 'dotted')
ax3.set_ylim(-100,100)
ax3.set_xlabel("N")
ax3.set_ylabel("$\epsilon_3$")
fig.savefig("slow_roll_params_PI.png")
# %%
# fig = plt.figure()
# gs = fig.add_gridspec(3, hspace = 0)
# ax1, ax2, ax3 = gs.subplots(sharex = True)
# fig.suptitle("Variation of $\epsilon_1$ and $\epsilon_2$ for USR1 and USR2")

# ax1.set_yscale('log')
# ax1.plot(N_USR1[4:58], e1_USR1[4:58], c = 'r')
# ax1.plot(N_[500:], e1_USR2[500:], c = 'r', linestyle = 'dashed')
# ax1.set_ylabel("$\epsilon_1$")

# ax2.plot(N_USR1[4:58], e2_USR1[4:58], c = 'b')
# ax2.plot(N_[500:], e2_USR2[500:], c = 'b', linestyle = 'dashed')
# ax2.set_ylabel("$\epsilon_2$")

# ax3.plot(N_USR1[4:58], e3_USR1[4:58], c = 'g')
# ax3.plot(N_[500:], e3_USR2[500:], c = 'g', linestyle = 'dashed')
# ax3.set_ylim(-100,100)
# ax3.set_xlabel("N")
# ax3.set_ylabel("$\epsilon_3$")
# plt.show()