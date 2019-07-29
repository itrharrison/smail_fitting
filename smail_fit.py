import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots
import emcee
import corner
from scipy.integrate import cumtrapz

def smail_distribution(z, theta):

  alpha, beta, z0 = theta

  sm = (z**alpha) * np.exp(-(z/z0)**beta)

  sm_norm = sm/cumtrapz(sm, z)[-1]

  return sm_norm

def log_prior(theta):
  alpha, beta, z0 = theta
  if 0. < alpha < 10.0 and 0.0 < beta < 10.0 and 0. < z0 < 10.:
      return 0.0
  return -np.inf

def log_likelihood(theta, x, y, yerr):
  model = smail_distribution(x, theta)
  sigma2 = yerr**2
  return -0.5*np.sum((y-model)**2/sigma2 + np.log(sigma2))

def log_probability(theta, x, y, yerr):
  lp = log_prior(theta)
  if not np.isfinite(lp):
      return -np.inf
  return lp + log_likelihood(theta, x, y, yerr)

def fit_smail(x, y, yerr, start, nwalkers=32, ndim=3):

  pos = np.asarray(start) + 1.e-4*np.random.randn(nwalkers, ndim)

  sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
  sampler.run_mcmc(pos, 5000)

  return sampler

if __name__=='__main__':

  data = np.loadtxt('./ligo-catalogue-sim.txt')
  data = data[:,1]
  counts, bins, _ = plt.hist(data, normed=True)
  bin_centres = (bins[:-1]+ bins[1:])/2

  count_errs = 1./np.sqrt(counts)

  sampler = fit_smail(bin_centres, counts, count_errs, start=[np.sqrt(2.), 2., data.mean()])

  fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)

  samples = sampler.chain
  labels = ["alpha", "beta", "z0"]
  for i in range(3):
      ax = axes[i]
      ax.plot(samples[:, :, i], "k", alpha=0.3)
      ax.set_xlim(0, len(samples))
      ax.set_ylabel(labels[i])
      #ax.yaxis.set_label_coords(-0.1, 0.5)

  axes[-1].set_xlabel("step number");
  plt.savefig('plots/chain.png', dpi=300, bbox_inches='tight')

  plt.close('all')
  flat_samples = sampler.flatchain
  fig = corner.corner(flat_samples, labels=labels)
  plt.savefig('plots/contours.png', dpi=300, bbox_inches='tight')

  median_parms = []

  for i in range(3):
      mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
      q = np.diff(mcmc)
      txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
      txt = txt.format(mcmc[1], q[0], q[1], labels[i])
      print(txt)
      median_parms.append(mcmc[1])

  plt.close('all')
  z = np.linspace(0., data.max(), 128)
  fig, axes = plt.subplots(1, figsize=(4.5, 3.75))
  axes.hist(data, normed=True, label='Data')
  axes.plot(z, smail_distribution(z, median_parms), '-', label='Best fit Smail')
  axes.legend()
  axes.set_xlabel('$z$')
  axes.set_title('alpha: {0:.3f}, beta: {1:.3f}, z0: {2:.3f}'.format(median_parms[0], median_parms[1], median_parms[2]))
  plt.savefig('plots/fit.png', dpi=300, bbox_inches='tight')

