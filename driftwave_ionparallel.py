import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, pi, c
from scipy.integrate import solve_ivp
import matplotlib as mpl

# Discretization.
# Resolution is 100 grid points within length of 2pi.
resolution = 200
# Assuming and L_y = L_z. 
L = 1.0
L_y = L
L_z = L
basegrid = np.linspace(0, 2*pi*L, resolution)
kzgrid, kygrid = np.meshgrid(basegrid, basegrid)
dk = np.diff(basegrid)[0]/(2*pi*L)

# Density normalization to solve the equations with a 
# unity order parameters.
n0 = 1e19


# For velocity, cs = sqrt(T/m) normalization is used.
# Default assumption is deuterium mass = 2*m_p.
# m_p = 938 MeV/c**2 is used and T is given in units of eV. 
def get_cref(T, m=2.0):
    return c*np.sqrt(T/(m*938*1e6))

# 2nd order central differences for the dPhi/dz and dPhi/dy terms.
def comp_dz(vec, order=4):
    dvec = np.zeros((resolution, resolution))
    if order==2:
        dvec[:,1:-1] = (vec[:,2:] - vec[:,0:-2])/(2*dk)
        dvec[:,0] = (vec[:,1] - vec[:,-2])/(2*dk)
        dvec[:,-1] = dvec[:,0] # Periodic boundary condition
        #dvec[:,-1] = (vec[:,0] - vec[:,-2])/(2*dk)
    if order==4:
        phi = vec
        dvec[:,2:-3] = (phi[:,0:-5] - 8*phi[:,1:-4] + 8*phi[:,3:-2] - phi[:,4:-1])/(12*dk)
        dvec[:,-3] = (phi[:,-5] - 8*phi[:,-4] + 8*phi[:,-2] - phi[:,0])/(12*dk)
        dvec[:,0] = (phi[:,-3] - 8*phi[:,-2] + 8*phi[:,1] - phi[:,2])/(12*dk)
        dvec[:,1] = (phi[:,-2] - 8*phi[:,0] + 8*phi[:,2] - phi[:,3])/(12*dk)
        dvec[:,-2] = (phi[:,-4] - 8*phi[:,-3] + 8*phi[:,0] - phi[:,1])/(12*dk)
        #dphidy[-1] = (phi[-3] - 8*phi[-2] + 8*phi[0] - phi[1])/(12*dky) # Periodic boundary condition
        dvec[:,-1] = dvec[:,0]
    return dvec
def comp_dy(vec, order=4):
    dvec = np.zeros((resolution,resolution))
    if order==2:
        dvec[1:-1,:] = (vec[2:,:] - vec[0:-2,:])/(2*dk)
        dvec[0,:] = (vec[1,:] - vec[-2,:])/(2*dk)
        dvec[-1,:] = dvec[0,:] # Periodic boundary condition
        #dvec[-1,:] = (vec[0,:] - vec[-2,:])/(2*dk)
    if order==4:
        phi = vec
        dvec[2:-3,:] = (phi[0:-5,:] - 8*phi[1:-4,:] + 8*phi[3:-2,:] - phi[4:-1,:])/(12*dk)
        dvec[-3,:] = (phi[-5,:] - 8*phi[-4,:] + 8*phi[-2,:] - phi[0,:])/(12*dk)
        dvec[0,:] = (phi[-3,:] - 8*phi[-2,:] + 8*phi[1,:] - phi[2,:])/(12*dk)
        dvec[1,:] = (phi[-2,:] - 8*phi[0,:] + 8*phi[2,:] - phi[3,:])/(12*dk)
        dvec[-2,:] = (phi[-4,:] - 8*phi[-3,:] + 8*phi[0,:] - phi[1,:])/(12*dk)
        #dphidy[-1] = (phi[-3] - 8*phi[-2] + 8*phi[0] - phi[1])/(12*dky) # Periodic boundary condition
        dvec[-1,:] = dvec[0,:]
    return dvec

# Compute potential through Boltzmann delta_n/n = ePhi/T. 
# Since T is already given as eV, it is sufficient to simply
# scale T with delta_n/n to get the potential.
def comp_phi(delta_n, T=100, n=1.0e19):
    phi = T*delta_n*n0/n
    return phi

# Compute the x-directional ExB velocity. Velocity is return in 
# units of m/s.
def comp_ve(phi, B=1.0):
    dphidy = comp_dy(phi)
    # E = -dphi/dy -> V_E = ExB/(B*B) = - (1/B)*dphi/dy
    ve = -(1/B)*dphidy
    return ve    
    
# This function defines the equations that we solve.
#     solution is a matrix [kygrid, kzgrid, equation_number]:
#         equation number 0 is density
#         equation number 1 is parallel momentum
#         equation number 2 is ion energy
def fun(t, solution, T, n, Ln, Lt, B, m, q):
    solution = solution.reshape((resolution, resolution, 3))
    # Check the perturbation level
    if np.max(np.abs(solution[:,:,0])*n0)>0.1*n or np.max(np.abs(solution[:,:,2]))>0.1*T:
        return
    # Evaluate potential
    phi = comp_phi(solution[:,:,0], T=T, n=n)
    # Evaluate ExB velocity in the x-direction
    ve = comp_ve(phi, B=B)
    
    # Evaluate parallel (z-direction) potential gradient
    dphidz = comp_dz(phi)
    # Parallel gradients of density, velocity, and pressure
    dnz = comp_dz(solution[:,:,0])
    duz = comp_dz(solution[:,:,1])
    dTz = comp_dz(solution[:,:,2])
    
    # Sound speed for normalization
    cs = get_cref(T)
    # Continuity equation. ExB is in units m/s, while duz is in units 1/cs.
    # Therefore, duz is multiplied by cs here.
    # Finally dndt is provided in normalized space (through dividing with n0)). 
    dndt = -n*(ve/Ln + cs*duz)/n0
    # Momentum equation along the magnetic field line.
    # dudt is normalized to (1/cs) units.  
    q_Ez = q*n*dphidz
    dudt_abs = -cs**2*(dTz/T + dnz/n + q_Ez/(T*n))
    dudt =  (1/cs)*dudt_abs
    # Ion energy balance equation.
    dpdt = -(5.0/3.0)*T*cs*duz - ve*T*(1/Ln + 1/Lt) # This is dp/(n*dt)
    dTdt = dpdt - T*dndt
    #dTdt = -T*(2*cs*duz + ve*T/Lt) # This equation was not quite right. 
    # Force periodic boundary condition
    dndt[:,-1] = dndt[:,0]
    dndt[-1,:] = dndt[0,:]
    dudt[:,-1] = dudt[:,0]
    dudt[-1,:] = dudt[0,:]
    dTdt[:,-1] = dTdt[:,0]
    dTdt[-1,:] = dTdt[0,:]
    # Stack the solution and ravel as 1-D vector for output.
    dvec = np.stack((dndt, dudt, dTdt), axis=2)
    return dvec.ravel()

def forward_dt(solution, T, n, Ln, Lt, B, m, q, dt=1e-5, method='rk4'):
    if method=='rk4':
        k1 = fun(0,solution, T, n, Ln, Lt, B, m, q)
        k2 = fun(0,solution + k1*dt/2, T, n, Ln, Lt, B, m, q)
        k3 = fun(0,solution + k2*dt/2, T, n, Ln, Lt, B, m, q)
        k4 = fun(0,solution + k3*dt, T, n, Ln, Lt, B, m, q)
        ds = (k1 + 2*k2 + 2*k3 + k4)*dt/6
        solution = solution + ds
    return solution

if __name__=='__main__':
    # These stay mostly fixed.
    T  = 100     # Temperature in eV.
    n  = 1.0e19  # Density in units of 1/m3.
    B  = 1.0     # Magnetic field strength in units of T.
    m  = 2.0     # Mass in units of proton mass.
    q  = 1.0     # Charge of the ion.

    # These are varied.
    Ln = 0.01      # Density gradient scale length in units of m.
    Lt = 40.0      # Temperature gradient scale length in units of m. mem=0.4
    ky = 10.0      # ky for the perturbation
    kz = 1.0       # kz for the perturbation
 
    use_solve_ivp = True # A flag to choose between scipy.integrate.solve_ivp
                         # and the Runge-Kutta 4 implementation in forward_dt.

    plot_maxphi_time = True # If true, plots the time evolution of the maximum 
                             # potential.
    plot_contour = True    # If true, plot phi contour, otherwise line.

    # Initial state.
    n_tilde = 0.001*np.sin(ky*kygrid + kz*kzgrid)
    upar = 0.00*n_tilde.copy()
    T_tilde = 0.00*n_tilde.copy()
    S0 = np.stack((n_tilde, upar, T_tilde), axis=2)


    if use_solve_ivp:
        sol = solve_ivp(fun, (0, 1e-4), S0.ravel(), vectorized=False, 
                    method='DOP853', args=(T, n, Ln, Lt, B, m, q)) 
        time = sol['t']
        solution = sol['y'] 
        solution = solution.reshape((resolution, resolution, 3, len(sol['y'][0,:])))
        for i in range(len(sol['y'][0,:,])):
            if i > 1000:
                break
            plt.clf()
            plt.figure(2)
            phi = comp_phi(sol['y'][:,i].reshape((resolution,resolution,3))[:,:,0])
            if plot_contour:
                plt.contourf(phi, cmap=mpl.colormaps['plasma'])
                plt.xlabel('Z (parallel to B)')
                plt.ylabel('Y (diamagnetic direction)')
                plt.gca().set_aspect(1.0)
            else:
                plt.plot(phi[:,50],'k')
                plt.ylim(-1.5, 1.5)
            plt.pause(0.01)
        plt.show()
        if plot_maxphi_time:
            plt.figure(1)
            plt.plot(time, np.max(comp_phi(solution[:,:,0,:]), axis=(0,1)),'k')
            plt.yscale('log')
            plt.show()
    else:
        solution = forward_dt(S0.ravel(), T, n, Ln, Lt, B, m, q, dt=1e-8)
        nt = 10000
        phivec = []
        for i in range(nt):
            solution = forward_dt(solution, T, n, Ln, Lt, B, m, q, dt=1e-8)
            solution1 = solution.reshape((resolution,resolution,3))
            plt.clf()
            phi = comp_phi(solution1[:,:,0])
            if np.max(phi) > 1000:
                break
            phivec.append(np.max(phi))
            #plt.contourf(phi, cmap=mpl.colormaps['plasma'])
            #plt.contourf(solution1[:,:,0], cmap=mpl.colormaps['plasma'])
            #plt.plot(phi[50,:])
            if i%10==0:
                print(i)
                plt.plot(solution1[50,:,0]/np.max(solution1[50,:,0]),'r')
            #/np.max(solution1[:,50,0]),'k')
            #plt.plot(solution1[:,50,1]/np.max(solution1[:,50,1]),'r')
            #plt.plot(solution1[:,50,2]/np.max(solution1[:,50,2]),'b')
            #plt.xlabel('Z (parallel to B)')
            #plt.ylabel('Y (diamagnetic direction)')
            #plt.gca().set_aspect(1.0)
            
                plt.pause(0.01)
        #sol = solve_ivp(fun, (0, 1.0-4), S0.ravel(), vectorized=False, 
        #                method='RK45', args=(T, n, Ln, Lt, B, m, q))
        #for i in range(len(sol['y'][0,:,])):
        #    if i > 200:
        #        break
        #    print(i)
        #    plt.clf()
        #    plt.figure(1)
        #    phi = comp_phi(sol['y'][:,i].reshape((resolution,resolution,3))[:,:,0])
        #    plt.contourf(phi, cmap=mpl.colormaps['plasma'])
        #    plt.xlabel('Z (parallel to B)')
        #    plt.ylabel('Y (diamagnetic direction)')
        #    plt.gca().set_aspect(1.0)
        #    plt.pause(0.01)
        #plt.show()

