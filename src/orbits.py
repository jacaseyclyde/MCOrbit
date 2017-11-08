#!/usr/bin/env python2


'''
Developed by Julio S Rodriguez (@BlueEarOtter) for the purposes of fitting orbits
to the minispiral of Sgr A West without proper motion data. There are still issues
that need to be worked out

Modified by J. Andrew Casey-Clyde (@jacaseyclyde)
'''

# =============================================================================
# =============================================================================
# # Topmatter
# =============================================================================
# =============================================================================

print('import os')
import os
print("import np")
import numpy as np
print("import plt")
import matplotlib.pyplot as plt
print("import corner")
import corner
print("import MCMC")
import emcee
print("multiprocessing imports")
from multiprocessing import Pool
from multiprocessing import cpu_count
print("import time")
import datetime
print("import warn")
import warnings

print("imports done")

#Ignores stuff
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

np.set_printoptions(precision=5,threshold=np.inf)

global datafile
global outpath
global stamp

datafile='../dat/CLF-Sim.csv'
outpath='../out/'

# setting up global ellipse parameter
tspace=1000
ttest=np.linspace(0 , 2 * np.pi , tspace)

# =============================================================================
# =============================================================================
# # Functions
# =============================================================================
# =============================================================================
    
# =============================================================================
# Geometry Functions    
# =============================================================================

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

def EllipseFunc(p):
    """
    Takes in coordinates ~p in radians~ and spits out f1 -- the constraint that
    the orbit must be elliptical and SgrA* lies at the focus
    """

    (aop, loan, inc, a, e)=p
    
    T = Transmat(p)
    
    # k is the relative velocity weight
    # Msun and G are currently just based on wikipedia and probably need a better source
    Mdyn = 4.02e+6 # +/- 0.16 +/- 0.04 x 10^6 M_sun
    Msun = 1.99e+30 # kg
    G = 6.67e-11 # m^3 kg^-1 s^-2
#    k=1e-1
    
    pcToKm = 3.0857e+13 # conversion factor for going from pc to km. Wikipedia
    
    a = a * pcToKm
    
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
    
    xdottest = -((ytest * l) / rtest**2) + (xtest / rtest) \
                * np.sqrt((GM * (2 * a - rtest) / (a * rtest)) - (l**2 / rtest**2))
    ydottest = -((xtest * l) / rtest**2) + (ytest / rtest) \
                * np.sqrt((GM * (2 * a - rtest) / (a * rtest)) - (l**2 / rtest**2))
                
    Vtest = (np.sin(inc) * np.sin(aop) * xdottest) \
            + (-np.sin(inc) * np.cos(aop) * ydottest)
    
    # Transform from Orbit to Ellipse
    rtest=np.vstack([xtest,ytest,ztest])
    rtest = rtest / pcToKm
    Rtest=np.matmul(np.linalg.inv(T),rtest)
            
    Rtest[2] = Vtest
            
    return Rtest.T # returning the transpose so that each point on the ellipse
                   # can be treated discretely

def Transmat(p):
    """
    Returns the Translation matrix and its derivatives. p must be given in
    radians.
    """
    (aop, loan, inc, a, e)=p
    T = np.array([[np.cos(loan)*np.cos(aop) - np.sin(loan)*np.sin(aop)*np.cos(inc),
        - np.sin(loan)*np.cos(aop) - np.cos(loan)*np.sin(aop)*np.cos(inc), 
        np.sin(aop)*np.sin(inc)],
        
        [np.cos(loan)*np.sin(aop) + np.sin(loan)*np.cos(aop)*np.cos(inc),
        -np.sin(loan)*np.sin(aop) + np.cos(loan)*np.cos(aop)*np.cos(inc),
        - np.cos(aop)*np.sin(inc)],
         
        [np.sin(loan)*np.sin(inc), np.cos(loan)*np.sin(inc), np.cos(inc)]])
    return T

def PlotFunc(p):
    my_data = np.genfromtxt(datafile, delimiter=',')
    
    X=my_data[:,0]
    Y=my_data[:,1]
    V=my_data[:,2]
    
    
    (aop, loan, inc, a, e)=p
    b=a*np.sqrt(1-e**2)
    x0=a*e
    Mdyn=4.02e+6# +/- 0.16 +/- 0.04 ? 10^6 M_sun

    x,y,T=SkytoOrb(X,Y,p)
    
    t=np.copy(ttest)
    
    xx=-x0+a*np.cos(t)
    yy=b*np.sin(t)
    zz=np.zeros_like(xx)
    
    rr=np.vstack([xx,yy,zz])
    RR=np.matmul(np.linalg.inv(T),rr)
    # everything above this line can be removed and replaced with a call to 
    # EllipseFunc to get RR
    
    r = (xx**2.+yy**2.)**.5
    rmin=a-a*e
    
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

# =============================================================================
# Probability functions
# =============================================================================

def PointPointProb(d,e):
    '''
    Returns the probability of point d being generated by a multivariate
    gaussian distribution centered on point e, with a covariance matrix cov.
    See the definition of multivariate gaussian distributions in
    Statistics, Data Mining, and Machine Learning in Astronomy, Ivezic et al. 2014
    '''

    M = len(d)
    H = np.linalg.inv(cov)
    
    x = d - e
    
    coeff = ( 1 / ((2 * np.pi)**(M / 2.)) * np.sqrt(np.linalg.det(cov)))
    
    exp = np.exp(-0.5 * np.matmul(x, np.matmul(H,x)))
    
    return coeff * exp
            
def PointModelProb(d,E):
    '''
    Returns the marginalized probability of a point d being generated by a 
    given model E
    '''
    
    pdE = 0.    # probability of getting point d, given model E
    
    for e in E:
        pdE += PointPointProb(d,e)
    
    return pdE
    
def lnLike(theta):
    '''
    Returns the log-likelihood that the dataset D was generated by the model
    definted by parameters theta. i.e.
    lnL = ln(p(D|theta))
    D should be formatted as [[x,y]]
    '''
    E = EllipseFunc(theta)
    
    lnlike = 0.
    
    for d in data:
        lnlike += np.log(PointModelProb(d,E))
    
    return lnlike

def lnPrior(theta):
    '''
    The log-likelihood of the prior. Currently assuming uniform
    '''
    #aop,loan, inc, a, e = theta
    if ((theta >= pos_min).all() and (theta < pos_max).all()):
        return 0.0
    return -np.inf

def lnProb(theta):
    '''
    Returns the total log probability that dataset D could have been generated
    by the elliptical model with parameters theta
    '''
    lp = lnPrior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = lnLike(theta)
    return lp + ll
    
# %%

# =============================================================================
# =============================================================================
# # Main Program
# =============================================================================
# =============================================================================

# initialize stamp
stamp = '{:%Y%m%d%H%M%S}/'.format(datetime.datetime.now())
    
# create output folder
os.makedirs(outpath + stamp)

# load data
my_data = np.genfromtxt(datafile, delimiter=',')

X=my_data[:,0]
Y=my_data[:,1]
V=my_data[:,2]
Xerr=my_data[:,3]
Yerr=my_data[:,4]
Verr=my_data[:,5]
Verr[Verr==0]=4e-2

data = (np.array([X,Y,V]).T)
cov = np.cov( data.T )

# =============================================================================
# MC MC
# =============================================================================

# Now, let's setup some parameters that define the MCMC
ndim = 5
nwalkers = 100
n_max = 1000

priors = np.array([[0.,0.,0,.1,.5],[2 * np.pi, 2 * np.pi, 2 * np.pi, 2, .999]])
prange = np.ndarray.tolist(priors.T)

# Initialize the chain
# Choice 1: chain uniformly distributed in the range of the parameters
print("initializing parameters")
pos_min = priors[0,:]
pos_max = priors[1,:]
psize = pos_max - pos_min
pos = [pos_min + psize*np.random.rand(ndim) for i in range(nwalkers)]

#print "priors" (really only needed if there's reasons to suspect initialization causing problems)
## Visualize the initialization
#fig = corner.corner(pos, labels=["$aop$","$loan$","$inc$","$a$", "$e$"],
#                    range=prange)
#fig.set_size_inches(10,10)
#
#
#plt.savefig(outpath + stamp + 'priors.pdf',bbox_inches='tight')
#plt.show()

# Set up backend so we can save chain in case of catastrophe
# note that this requires h5py and the latest version of emcee on github
filename = 'chain.h5'
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim) # uncomment this line to start the simulation from scratch


with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnProb, pool=pool, backend=backend)

    ncpu = cpu_count()
    print("Running MCMC on {0} CPUs".format(ncpu))
    old_tau = np.inf
    for sample in sampler.sample(pos, iterations=n_max, progress=True):
        if sampler.iteration % 100:
            continue
        
        #check convergence
        tau = sampler.get_autocorr_time(tol=0)
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            break
        old_tau = tau

tau = [91.64417, 104.07634, 94.89677, 96.07088, 123.73447] # sampler.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
log_prob_samples = sampler.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = sampler.get_blobs(discard=burnin, flat=True, thin=thin)

print("burn-in: {0}".format(burnin))
print("thin: {0}".format(thin))
print("flat chain shape: {0}".format(samples.shape))
print("flat log prob shape: {0}".format(log_prob_samples.shape))
print("flat log prior shape: {0}".format(log_prior_samples.shape))


# let's plot the results
all_samples = np.concatenate((
    samples, log_prob_samples[:, None], log_prior_samples[:, None]
), axis=1)

fig = corner.corner(samples, labels=["$aop$","$loan$","$inc$","$a$", "$e$"],
                    range=prange)
fig.set_size_inches(10,10)
plt.savefig(outpath + stamp + 'results_{0}.pdf'.format(nwalkers),bbox_inches='tight')
plt.show()

aop, loan, inc, a, e = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
            zip(*np.percentile(samples, [16, 50, 84],
            axis=0)))
aop=aop[0]
loan=loan[0]
inc=inc[0]
a=a[0]
e=e[0]
pbest=np.array([aop, loan, inc, a, e])

print(pbest)

PlotFunc(pbest)
