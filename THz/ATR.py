import numpy as np

def r_p(n1,n2,Theta1):
    cos_th2 = np.sqrt(np.complex128(1-((n1/n2)**2)*np.sin(Theta1)**2))
    r_coeff=(n2*np.cos(Theta1)-n1*cos_th2)/(n2*np.cos(Theta1)+n1*cos_th2)
    return r_coeff
	
def n_p(n1,Theta1,ratio):
    r_cal=r_p(n1,1,Theta1)
    rp=r_cal*ratio
    q=(rp-1)/(rp+1)
    Delta=1-q**2*(np.sin(2*Theta1))**2
    eps=(n1**2)*((1-np.sqrt(Delta))/(2*q**2*(np.cos(Theta1))**2))
    idx = np.imag(eps) < 0
    eps[idx] = (n1**2)*((1+np.sqrt(Delta))/(2*q**2*(np.cos(Theta1))**2))[idx]
    n = np.sqrt(eps)
    return n
	
def eps_water(f):
    eps1=5.16+0.06
    epsinf=3.49+0.07
    eps0=78.36
    Thaw1=(7.89+0.06)
    omega=2*np.pi*f
    Thaw2=(0.180+0.014)
    epsJepsenplus=(eps0-eps1)/(1-1j*omega*Thaw1)+(eps1-epsinf)/(1-1j*omega*Thaw2)+epsinf
    eps1=5.16+0.06
    epsinf=3.49-0.07
    eps0=78.36
    Thaw1=(7.89-0.06)
    omega=2*np.pi*f
    Thaw2=(0.180-0.014)
    epsJepsenminus=(eps0-eps1)/(1-1j*omega*Thaw1)+(eps1-epsinf)/(1-1j*omega*Thaw2)+epsinf
    return epsJepsenplus, epsJepsenminus

def dn_dphi(n1, n2, theta, T, f):
    r =   r_p(n1, 1, theta)
    phi = np.cos(theta)/n1*((1-T*r)/(1+T*r))
    out = 1/phi*(n2+1/n2*(n1**2*np.sin(theta)**2)/np.sqrt(1-(2*phi*n1*np.sin(theta))**2)) * np.cos(theta)/n1*2*r/(1+r*T)**2*T*f*2*np.pi*1j
    
    temp = 1/phi*(n2-1/n2*(n1**2*np.sin(theta)**2)/np.sqrt(1-(2*phi*n1*np.sin(theta))**2)) * np.cos(theta)/n1*2*r/(1+r*T)**2*T*f*2*np.pi*1j
    rp=r*T
    q=(rp-1)/(rp+1)
    Delta=1-q**2*(np.sin(2*theta))**2
    eps=(n1**2)*((1-np.sqrt(Delta))/(2*q**2*(np.cos(theta))**2))
    idx = np.imag(eps) > 0 
    out[idx] = temp[idx]

    
    return out


def dn_damp(n1, n2, theta, T):
    r =   r_p(n1, 1, theta)
    phi = np.cos(theta)/n1*((1-T*r)/(1+T*r))
    out = 1/phi*(n2+1/n2*(n1**2*np.sin(theta)**2)/np.sqrt(1-(2*phi*n1*np.sin(theta))**2)) * np.cos(theta)/n1*2*r/(1+r*T)**2*T/np.abs(T)
    
    temp = 1/phi*(n2-1/n2*(n1**2*np.sin(theta)**2)/np.sqrt(1-(2*phi*n1*np.sin(theta))**2)) * np.cos(theta)/n1*2*r/(1+r*T)**2*T/np.abs(T)
    rp=r*T
    q=(rp-1)/(rp+1)
    Delta=1-q**2*(np.sin(2*theta))**2
    eps=(n1**2)*((1-np.sqrt(Delta))/(2*q**2*(np.cos(theta))**2))
    idx = np.imag(eps) > 0 
    out[idx] = temp[idx]

    
    return out