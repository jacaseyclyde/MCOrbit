#!/usr/bin/env python2


'''
Created on Fri Feb  9 16:08:27 2018

@author: jacaseyclyde
'''

# =============================================================================
# =============================================================================
# # Topmatter
# =============================================================================
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5,threshold=np.inf)

datafile='../dat/CLF-Sim.csv'
outpath='../out/'

# =============================================================================
# =============================================================================
# # Global Parameters
# =============================================================================
# =============================================================================

# =============================================================================
# Conversions   
# =============================================================================
pcArcsec = 8e3 / 60 / 60 / 180 * np.pi # [pc/arcsec] @ gal. center
kgMsun = 1.98855e+30 # [kg/Msun]
secYr = 60. * 60. * 24. * 365. # [s/yr]
kmPc = 3.0857e+13 # [km/pc]
mKm = 1.e+3 # [m/km]


# =============================================================================
# Constants   
# =============================================================================
G = 6.67e-11 * kgMsun / (mKm * kmPc)**3 * secYr**2 # [pc^3 Msun^-1 yr^-2]

# =============================================================================
# Mass Data 
# =============================================================================
Mdat = np.genfromtxt('../dat/enclosed_mass_distribution.txt')
Mdist = Mdat[:,0] * pcArcsec # [pc]
Menc = Mdat[:,1] # [log(Msun)]

# =============================================================================
# Other 
# =============================================================================
tspace=1000
ttest=np.linspace(0 , 2 * np.pi , tspace)


# =============================================================================
# =============================================================================
# # Functions
# =============================================================================
# =============================================================================
    
def Orbit(x0,v0,tstep):
    '''
    Takes in an initial position and velocity vector and generates an
    integrated orbit around SgrA*. Returns 2 arrays of position and veolocity
    vectors. Currently ignoring units until a few other things are developed
    
    x0 = [pc], v0 = [km/s], tstep = [yr]
    '''
    npoints = 100 # right now this is a completely arbitraty number
    pos = np.zeros((npoints,2))
    vel = np.zeros_like(pos)
    
    pos[0] = x0 # [pc]
    vel[0] = v0 * secYr / kmPc # [pc/yr]
    
    posnorm = np.linalg.norm(pos[0]) # [pc]
    a_old = - G * MassFunc(posnorm) / posnorm**2 # [pc/yr^2]
    a_old = a_old * pos[0] / posnorm
    
    for i in range(npoints - 1):
        pos[i+1] = pos[i] + vel[i] * tstep + 0.5 * a_old * tstep**2
        
        posnorm = np.linalg.norm(pos[i+1])
        a_new = - G * MassFunc(posnorm) / posnorm**2
        a_new = a_new * pos[i+1] / posnorm
        
        vel[i+1] = vel[i] + 0.5 * (a_old + a_new) * tstep
        
        a_old = a_new
        
    return pos,(vel / secYr * kmPc) # [pc], [km/s]
    
def MassFunc(dist):
    '''
    Takes in a distance from SgrA* and returns the enclosed mass. Based on data
    from Feldmeier-Krause et al. (2016), with interpolation
    
    dist = [pc]
    '''
    return 10**np.interp(dist,Mdist,Menc) # [Msun]

def Transmat(aop,loan,inc):
    """
    Returns the Translation matrix and its derivatives.
    
    aop = [rad], loan = [rad], inc = [rad]
    """

    T = np.array([[np.cos(loan)*np.cos(aop) - np.sin(loan)*np.sin(aop)*np.cos(inc),
        - np.sin(loan)*np.cos(aop) - np.cos(loan)*np.sin(aop)*np.cos(inc), 
        np.sin(aop)*np.sin(inc)],
        
        [np.cos(loan)*np.sin(aop) + np.sin(loan)*np.cos(aop)*np.cos(inc),
        -np.sin(loan)*np.sin(aop) + np.cos(loan)*np.cos(aop)*np.cos(inc),
        - np.cos(aop)*np.sin(inc)],
         
        [np.sin(loan)*np.sin(inc), np.cos(loan)*np.sin(inc), np.cos(inc)]])
    return T

def SkytoOrb(X,Y,p):
    """
    Takes in sky coods, X,Y, and V, as well as parameters, p, in radians. Calc-
    ulates the transformation matrix T. Returns the Orbit coords x and y as
    well as their associated velocities vx and vy
    """
    (aop, loan, inc, a, e)=p
    
    T=Transmat(p)
    
    x=T[0][2]/np.cos(inc)*(-T[2][1]*Y-T[2][0]*X)+ T[0][1]*Y + T[0][0]*X
    y=T[1][2]/np.cos(inc)*(-T[2][1]*Y-T[2][0]*X)+ T[1][1]*Y + T[1][0]*X
    
    return x,y,T

def OrbitFunc(p):
    """
    Takes in coordinates ~p in radians~ and spits out f1 -- the constraint that
    the orbit must be elliptical and SgrA* lies at the focus
    """

    (aop, loan, inc, x0, v0)=p
    
    T = Transmat(p)
    
    # k is the relative velocity weight
    # Msun and G are currently just based on wikipedia and probably need a better source
    Mdyn = 4.02e+6 # +/- 0.16 +/- 0.04 x 10^6 M_sun
    Msun = 1.99e+30 # kg
#    k=1e-1
    
     # conversion factor for going from pc to km. Wikipedia
    
    #a = a * pcToKm
    
    # should probably get better numbers than general googling but this works for now
    GM = G * Mdyn * Msun / 1e+9 # last part converting from m^3 to km^3
    
    # Sanity Check
    if e >=1.:
        return np.inf
    
    b=a*np.sqrt(1-e**2.) 
    x0=a*e
    
    # reduced angular momentum
    l = (a - a * e) * np.sqrt(GM * (1 + e) / (a * (1 - e)))
    
    '''
    Generate Ellipse
    '''    
    # Parametric coordinates
    xtest=-x0+a*np.cos(ttest)
    ytest=b*np.sin(ttest)
    ztest=np.zeros_like(xtest)
    rtest = np.sqrt(xtest**2 + ytest**2)
    
    sqrtArg = ((GM * (2 * a - rtest)) / (a * rtest)) - (l / rtest)**2
    
    if ( sqrtArg < 0.).any():
        sqrtArg = 0. # just doing a hard cuttoff, since i'm about 90% sure that values less than 0 are due to roundoff
        
    xdottest = -((ytest * l) / rtest**2) + (xtest / rtest) * np.sqrt(sqrtArg)
    ydottest = ((xtest * l) / rtest**2) + (ytest / rtest) * np.sqrt(sqrtArg)
                
    Vtest = (np.sin(inc) * np.sin(aop) * xdottest) + (-np.sin(inc) * np.cos(aop) * ydottest)
    
    # Transform from Orbit to Ellipse
    rtest=np.vstack([xtest,ytest,ztest])
    rtest = rtest / kmPc
    Rtest=np.matmul(np.linalg.inv(T),rtest)
            
    Rtest[2] = Vtest
            
    return Rtest.T # returning the transpose so that each point on the ellipse
                   # can be treated discretely

def PlotFunc(p):
    my_data = np.genfromtxt(datafile, delimiter=',')
    
    X=my_data[:,0]
    Y=my_data[:,1]
    #V=my_data[:,2]
    
    
    (aop, loan, inc, a, e)=p
    b=a*np.sqrt(1-e**2)
    x0=a*e
    #Mdyn=4.02e+6# +/- 0.16 +/- 0.04 ? 10^6 M_sun

    x,y,T=SkytoOrb(X,Y,p)
    
    t=np.copy(ttest)
    
    xx=-x0+a*np.cos(t)
    yy=b*np.sin(t)
    zz=np.zeros_like(xx)
    
    rr=np.vstack([xx,yy,zz])
    RR=np.matmul(np.linalg.inv(T),rr)
    # everything above this line can be removed and replaced with a call to 
    # EllipseFunc to get RR
    
    #r = (xx**2.+yy**2.)**.5
    #rmin=a-a*e
    
    XX=RR[0,:]
    YY=RR[1,:]
    
    x0true = 0.5 * .9
    btrue = 0.5 * np.sqrt(1-0.9**2)
    
    xxtrue=-x0true+0.5*np.cos(t)
    yytrue=btrue*np.sin(t)
    zztrue=np.zeros_like(xxtrue)
    rrtrue=np.vstack([xxtrue,yytrue,zztrue])
    
    Ttrue = Transmat([np.radians(60),np.radians(135),np.radians(300),0.5,0.9])
    Rtrue = np.matmul(np.linalg.inv(Ttrue),rrtrue)
    Xtrue = Rtrue[0,:]
    Ytrue = Rtrue[1,:]
        
    p[0:3]=np.degrees(p[0:3])
    plt.figure(1)
    plt.clf()
    plt.plot(X,Y,'ro',XX,YY,'k-',Xtrue,Ytrue,'c--',[0],[0],'g*')
    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('Sky Plane')#. p = {}'.format(p))
    plt.xlabel('Offset (pc)')
    plt.ylabel('Offset (pc)')
    plt.gca().invert_xaxis()
    plt.savefig(outpath + stamp + 'skyplane.pdf',bbox_inches='tight')
    plt.show()
    plt.figure(2)
    plt.clf()
    plt.plot(x,y,'bx',xx,yy,'k-',[0],[0],'g*')
    plt.grid(True)
    plt.axes().set_aspect('equal', 'datalim')
    plt.title('Orbit Plane')#. p = {}'.format(p))
    plt.xlabel('Offset (pc)')
    plt.ylabel('Offset (pc)')
    plt.savefig(outpath + stamp + 'orbitplane.pdf',bbox_inches='tight')
    plt.show()
    p[0:3]=np.radians(p[0:3])
    
    return
