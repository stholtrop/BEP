from model.variationalmontecarlo import *
from model.auxilary import *

def wavefunction2(alpha):
    def psi2(x):
        x1 = x[:,0,:]; x2 = x[:,1,:]
        r1 = np.linalg.norm(x1, axis=1); r2 = np.linalg.norm(x2,axis=1)
        r12 = np.linalg.norm(x1 - x2, axis=1)
        return np.exp(-4*r1 - 4*r2 - r12/(1 + alpha * r12))
    return psi2

def local_energy(alpha):
    def EL(x):
        x1 = x[:,0,:]; x2 = x[:,1,:]
        r12 = np.linalg.norm(x1 - x2, axis=1)
        r1r2 = np.sum((x1/np.linalg.norm(x1, axis=1)[:,None] - x2/np.linalg.norm(x2, axis=1)[:,None])*(x1 - x2), axis=1)
        E = -4 + r1r2 /r12/(1 + alpha * r12)**2 - 1/r12/(1+alpha*r12)**3 - 1/4*(1+alpha*r12)**4 + 1/r12
        return -4 + r1r2 /r12/(1 + alpha * r12)**2 - 1/r12/(1+alpha*r12)**3 - 1/4*(1+alpha*r12)**4 + 1/r12
    return EL

# A helium atom with two electrons
for alpha in np.arange(0.05, 0.3, 0.05):
    psi2 = wavefunction2(alpha)
    MC = VariationalMonteCarlo(psi2, lambda x : x + 5*(np.random.rand(*x.shape) - 0.5), lambda N : 2*(np.random.rand(N, 2, 3) - 0.5))
    samples = MC.generate_samples(400, 30000, 4000)
    print(f"{alpha}:", mc_integrator(local_energy(alpha), samples.get_samples()))