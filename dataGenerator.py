import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi
from scipy.integrate import solve_ivp
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

resolution = 100
# Domain length in y-direction. Needed to define 
# grid spacing in y-direction.
L_y = 1.0
kygrid = np.linspace(0, 2*pi*L_y, resolution)
dky = np.diff(kygrid)[0]/(2*pi*L_y)

# Density normalization to solve the equations with a 
# unity order parameters. 
n0 = 1e19 

# Central difference for the dPhi/dy term.
# User can select between orders 2 and 4. 
def comp_phi_dy(phi, order=4):
    dphidy = np.zeros(resolution)
    if order==2:
        dphidy[1:-1] = (phi[2:] - phi[0:-2])/(2*dky)
        dphidy[0] = (phi[1] - phi[-2])/(2*dky)
        dphidy[-1] = dphidy[0]
        #dphidy[-1] = (phi[0] - phi[-2])/(2*dky) # Periodic boundary condition
    if order==4:
        #dphidy[2:-2] = (phi[0:-4] - 8*phi[1:-3] + 8*phi[3:-1] - phi[4:])/(12*dky)
        #dphidy[0] = (phi[-2] - 8*phi[-1] + 8*phi[1] - phi[2])/(12*dky)
        #dphidy[1] = (phi[-1] - 8*phi[0] + 8*phi[2] - phi[3])/(12*dky)
        #dphidy[-2] = (phi[-4] - 8*phi[-3] + 8*phi[-1] - phi[0])/(12*dky)
        #dphidy[-1] = (phi[-3] - 8*phi[-2] + 8*phi[0] - phi[1])/(12*dky) # Periodic boundary condition
        dphidy[2:-3] = (phi[0:-5] - 8*phi[1:-4] + 8*phi[3:-2] - phi[4:-1])/(12*dky)
        dphidy[-3] = (phi[-5] - 8*phi[-4] + 8*phi[-2] - phi[0])/(12*dky)
        dphidy[0] = (phi[-3] - 8*phi[-2] + 8*phi[1] - phi[2])/(12*dky)
        dphidy[1] = (phi[-2] - 8*phi[0] + 8*phi[2] - phi[3])/(12*dky)
        dphidy[-2] = (phi[-4] - 8*phi[-3] + 8*phi[0] - phi[1])/(12*dky)
        #dphidy[-1] = (phi[-3] - 8*phi[-2] + 8*phi[0] - phi[1])/(12*dky) # Periodic boundary condition
        dphidy[-1] = dphidy[0]
    return dphidy

# Compute potential through Boltzmann delta_n/n = ePhi/T. 
# Since T is already given as eV, it is sufficient to simply
# scale T with delta_n/n to get the potential.
def comp_phi(delta_n, T=100, n=1.0):
    phi = T*delta_n/(n*n0)
    return phi

# This function defines the continuity equation.
def fun(t, delta_n, T, n, Ln, B):
    # Solve potential.
    phi = comp_phi(delta_n, T=T, n=n)
    # Compute electric field in the y-direction.
    dphidy = comp_phi_dy(phi)
    # dn/dt = -v_ExB*grad_n 
    # grad_n = n*n0/Ln (n0 is the normalization).
    # v_ExB = (1/B)*(-dphidy)
    dndt = (1/B)*dphidy*n*n0/Ln
    dndt[-1] = dndt[0] #Periodic boundary condition
    return dndt

def datafun(Ln,ky):
    T  = 100  # Temperature in eV.
    n  = 1.0  # Density in units of 1e19/m3.
    B  = 1.0  # Magnetic field strength in units of T.
    ntilde = 0.01*np.sin(ky*kygrid)*n0
    sol = solve_ivp(fun, (0, 1e-3), ntilde, vectorized=False, 
                    method='Radau', rtol=1e-4, max_step=1e-6, args=(T, n, Ln, B))
    return(sol)
    

if __name__=='__main__':
    T  = 100  # Temperature in eV.
    n  = 1.0  # Density in units of 1e19/m3.
    #Ln = 0.01 # Density gradient scale length in units of m.
    B  = 1.0  # Magnetic field strength in units of T.
    
    #ky = 5.0  # ky for the perturbation

    Lninp=input('Enter the value for Ln: ')
    kyinp=input('Enter the value for ky: ')
    Ln=float(Lninp)
    ky=float(kyinp)

    
    #ntilde = 0.01*np.sin(ky*kygrid)*n0
    #sol = solve_ivp(fun, (0, 1e-3), ntilde, vectorized=False, method='Radau', rtol=1e-4, max_step=1e-6, args=(T, n, Ln, B))
    sol=datafun(Ln,ky)

    # This part is for generating Figures.
    # First a movie of the drift wave propagation.
    for i in range(len(sol['y'][0,:])):
         if i > 200:
             break
         #print(i)
         plt.clf()
         plt.figure(1)
         phi = comp_phi(sol['y'][:,i])
         dphidy = comp_phi_dy(phi)
         plt.plot(kygrid, sol['y'][:,i]/(0.01*n0),'k--', label='n-tilde/n0 [%]')
         plt.plot(kygrid, -dphidy/np.max(np.abs(dphidy)), 'r--', 
                  label='E-field/max(E-field)')
         plt.legend(loc='upper left')
         plt.ylim(-1.1,1.1)
         plt.pause(0.10)

    # Figure 2 illustrates the time evolution of the drift wave.
    plt.figure(2)
    phivec = np.max(comp_phi(sol['y'][:]),axis=0)
    plt.plot(sol['t'], phivec/phivec[0], 'k', label='Phi/Phi0')
    plt.legend()

    # Figure 3 illustrates the frequency spectrum of the wave and 
    # demonstrates that the peak frequency aligns with the electron
    # diamagnetic frequency.
    plt.figure(3)
    tspace = np.linspace(sol['t'][0], sol['t'][-1],100000)
    fun1 = interp1d(sol['t'][:],comp_phi(sol['y'][50,:]))
    dt1 = np.diff(tspace)[0]
    yspace = fun1(tspace)
    phift = fft(yspace/dt1)
    fr = fftfreq(len(yspace), d=dt1)
    plt.plot(fr, np.abs(phift)/np.max(np.abs(phift)),'k',label='Frequency')
    omega_e_star = ky*T/(Ln*B)
    plt.xlim(0, 100000)
    #plt.xscale('log')
    plt.vlines(omega_e_star,0, 1, 'r',label='omega e star')
    plt.legend()
    plt.show()    
