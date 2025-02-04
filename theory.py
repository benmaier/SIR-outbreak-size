import numpy as np
import epipack as epk

from tqdm import tqdm


from scipy.optimize import newton
from scipy.interpolate import CubicSpline

import matplotlib.pyplot as pl

from bfmplot.sequences import bochenska2 as colors

from matplotlib.ticker import PercentFormatter


def new_approx(R0, s0, n_r_samples=10001):
    Omega = 1-1/R0**2
    rs = np.logspace(-4, np.log10(Omega),n_r_samples)
    ts = []
    for r in rs:
        t_ = compute_expression(r, s0, R0)
        if len(ts) > 0 and t_ < ts[-1]:
            break
        ts.append(t_)

    rs = np.concatenate(([0.], rs[:len(ts)]))

    ts = [0.] + ts
    ts = np.array(ts)
    ndx = np.argsort(ts)
    return ts[ndx], rs[ndx]


def new_worse_approx(R0, s0):
    Omega = 1-1/R0**2
    rs = np.logspace(-4, np.log10(Omega*1.01),10001)
    ts = []
    for r in rs:
        t_ = compute_expression_worse_approximation(r, s0, R0)
        ts.append(t_)
    ts = [0.] + ts
    rs = np.concatenate(([0.], rs))
    ts = np.array(ts)
    ndx = np.argsort(ts)
    return ts[ndx], rs[ndx]


def compute_expression_worse_approximation(r, s0, R):
    # Define the polynomial coefficients
    poly_coeffs = [
         R**3 * s0,               # x^3 term
        -4 * R**2 * s0,           # x^2 term
        -8 + 8 * R * s0,              # x term
         8 - 8*s0,                # constant
    ]

    # Compute the roots of the polynomial
    roots = np.roots(poly_coeffs).astype(complex)

    # Initialize the result
    result = 0

    # Iterate over the roots and compute the summation terms
    for root in roots:

        # Compute the terms in the RootSum expression
        term1 = np.log(r - root) / (-1 + R*s0 - R**2*s0 * root + 3*R**3*s0 * root**2 / 8)
        term2 = np.log(-root)    / (-1 + R*s0 - R**2*s0 * root + 3*R**3*s0 * root**2 / 8)

        # Add the computed terms to the result
        result += term1 - term2

    # Return the final computed value
    return np.real(result)

def compute_expression(r, s0, R):

    # Define the polynomial coefficients
    poly_coeffs = [
        R**4 * s0,                # x^4 term
        -6 * R**3 * s0 + 2*R**2,   # x^3 term
        18 * R**2 * s0 - 2*R**2,  # x^2 term
        -32 * R * s0 + 32,         # x term
        -32 + 32*s0,               # constant
    ]

    roots = np.roots(poly_coeffs)

    result = 0

    g = ((16 + (R*roots)**2) * np.log(roots/(roots-r))) \
            / \
         (16 - 16 * R * s0 - 2 * R**2 * roots + 18 * R**2 * s0 * roots \
          + 3 * R**2 * roots**2 - 9 * R**3 * s0 * roots**2 + 2 * R**4 * s0 * roots**3)

    return np.real(g.sum())

def find_inflection_t_and_r_from_new_approx(s0, R):

    # Define the polynomial coefficients
    poly_coeffs = [
              R**4 * s0,           # x^4 term
         -6 * R**3 * s0 + 2*R**2,  # x^3 term
         18 * R**2 * s0 - 2*R**2,  # x^2 term
        -32 * R * s0 + 32,         # x term
        -32 + 32*s0,               # constant
    ]

    roots = np.roots(poly_coeffs)

    sum_g2 = lambda r: np.real(np.sum(
            ((16 + (R*roots)**2) / (roots-r)**2) \
                    / \
               (16 - 16 * R * s0 - 2 * R**2 * roots + 18 * R**2 * s0 * roots \
                +3 * R**2 * roots**2 - 9 * R**3 * s0 * roots**2 + 2 * R**4 * s0 * roots**3)
          ))
    sum_g3 = lambda r: np.real(np.sum(
            2*((16 + (R*roots)**2) /(roots-r)**3) \
                    / \
               (16 - 16 * R * s0 - 2 * R**2 * roots + 18 * R**2 * s0 * roots \
                +3 * R**2 * roots**2 - 9 * R**3 * s0 * roots**2 + 2 * R**4 * s0 * roots**3)
        ))

    r_star = newton(sum_g2, x0=0, fprime=sum_g3)

    sum_g = np.real(np.sum(
            ((16 + (R*roots)**2) * np.log(roots/(roots-r_star))) \
                    / \
               (16 - 16 * R * s0 - 2 * R**2 * roots + 18 * R**2 * s0 * roots \
                +3 * R**2 * roots**2 - 9 * R**3 * s0 * roots**2 + 2 * R**4 * s0 * roots**3)
          ))


    return sum_g, r_star

def sigmoid_approx(t, R0, s0):
    fac = 1/R0**2/s0
    A = R0*s0 - 1
    B = np.sqrt((R0*s0-1)**2+2*s0*(1-s0)*R0**2)
    return fac * (A + B*np.tanh(0.5*B*t - np.arctanh(A/B)))

def model_exact(t, R0, s0, return_whole_result=False):
    model = epk.SIRModel(R0, recovery_rate=1)\
               .set_initial_conditions({'S':s0, 'I': 1-s0})
    res = model.integrate(t)
    if return_whole_result:
        return res
    else:
        return res['R']

def model_integrate_until_inflection(t0, R0, s0):
    model = epk.SIRModel(R0, recovery_rate=1)\
               .set_initial_conditions({'S':s0, 'I': 1-s0})
    t, res = model.integrate_until(t0, stop_condition=lambda t, y: 1/R0 - y[0])
    return t

def analyze(t, R0, s0):
    exact = model_exact(t, R0, s0)
    new = new_approx(t, R0, s0)
    sigmoid = sigmoid_approx(t, R0, s0)
    return {'exact': exact, 'sigmoid': sigmoid,'new':new}

def sigmoid_final(R0):
    return 2/R0*(1-1/R0)

def new_final(R0):
    return (1-1/R0**2)

def complicated_new_final(R0):
    #return 1 - (-(np.sqrt(108 + 2916/R0**2)/2 - 27/R0)**(1/3)/3 + (np.sqrt(108 + 2916/R0**2)/2 - 27/R0)**(-1/3))**4
    A = np.sqrt(3+81/R0**2) - 9/R0
    B = 3**(1/3)
    return 1 - B/27 * ( (A**(2/3) - B)/A**(1/3) )**4

def model_final(R0s, s0=1-1e-7):
    exact = []
    for R0 in R0s:
        _exact = model_exact([0,1e9], R0, s0)[-1]
        exact.append(_exact)
    return np.array(exact)

def plot_final(s0=0.9999999):
    R0s = np.linspace(1.001, 3,101)
    exact = []
    for R0 in R0s:
        _exact = model_exact([0,1e9], R0, s0)[-1]
        exact.append(_exact)
    new = new_final(R0s)
    sig = sigmoid_final(R0s)
    new2 = complicated_new_final(R0s)

    fig, ax = pl.subplots(1,1,figsize=(4.5,3.5))
    pl.plot(R0s, exact, lw=4, c='k', alpha=0.3, label='SIR-type models')
    handle_new, = pl.plot(R0s, new, label=r'1-1/R$_0^2$')
    handle_sig, = pl.plot(R0s, sig, '-.', label='2(1-1/R$_0$)/R$_0$')
    handle_new2, = pl.plot(R0s, new2, '--',label='Ω2')
    pl.legend(loc=[0.03,0.66])
    pl.ylabel(r'outbreak size Ω')
    pl.xlabel('basic reproduction number R$_0$')
    ax.set_xticks([1,1.5,2,2.5,3])

    ia = ax.inset_axes([0.25,0.12, 0.4,0.3])
    ia.plot([1,3],[0.00,0.06],':',lw=1,c='k')
    ia.plot([1,3],[0.00,0.02],':',lw=1,c='k')
    ia.plot(R0s[1:], 1-new[1:]/exact[1:], color=handle_new.get_color())

    ndx = np.argmax(1-new[1:]/exact[1:])
    print("new max:", R0s[ndx+1])
    print(1-new[ndx+1]/exact[ndx+1])

    ndx = np.argmax(1-new2[1:]/exact[1:])
    print("new2 max:", R0s[ndx+1])
    print(1-new2[ndx+1]/exact[ndx+1])

    ia.plot(R0s[1:], 1-sig[1:]/exact[1:], '-.', color=handle_sig.get_color())
    ia.plot(R0s[1:], 1-new2[1:]/exact[1:],':',lw=2,color=handle_new2.get_color())
    #ia.plot([2.25,3.0],[0.06,0.06],':',lw=1,c='k')

    ia.set_ylabel('rel. err.',loc='top')
    ia.set_yscale('log')

    ia.set_yticks([0.01,0.02,0.06,0.53])
    ia.set_ylim([0.01,0.53])
    ia.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    ia.yaxis.set_label_position("right")
    ia.spines["right"].set_visible(True)
    ia.spines["left"].set_visible(False)
    ia.yaxis.tick_right()

    fig.tight_layout()

    return pl.gca()

def find_time_of_inflection_approx(R0, s0=1-1e7):
    r_inflection = 1-1/R0
    t_inflection = compute_expression(r_inflection, s0, R0)
    return t_inflection

def find_true_time_of_inflection_approx(R0, s0=1-1e7):
    t, r = new_approx(R0, s0, n_r_samples=1001)
    t = t[1:]
    r = r[1:]
    #print(t, r)
    r_inflection = 1-1/R0
    a = np.diff(np.sign(np.diff(np.diff(r)/np.diff(t))))
    #pl.figure()
    #pl.plot(t, r)
    #pl.title(f"{R0=}")
    #print(np.diff(np.diff(r)))
    #pl.plot(np.diff(np.diff(r)/np.diff(t)))
    #pl.plot(np.sign(np.diff(np.diff(r))))
    ndx = np.where(a != 0)[0][0]
    r_inflection = r[ndx]
    r -= r_inflection
    t_inflection = compute_expression(r_inflection, s0, R0)
    func = lambda x: np.interp(x, t, r)
    true_t_inflection = newton(func, t_inflection)
    return true_t_inflection

def find_time_of_inflection_exact(R0, s0=1-1e7):
    return model_integrate_until_inflection(0, R0, s0)[-1]

def find_time_of_inflection_old(R0, s0=1-1e-7):
    A = R0*s0 - 1
    B = np.sqrt((R0*s0-1)**2+2*s0*(1-s0)*R0**2)
    t = (2*np.arctanh(A/B)/B)
    return t

def plot_epidemic(R0, s0=1-1e-4, axs=None):

    tnew, new = new_approx(R0, s0)

    t = np.linspace(tnew[0],tnew[-1],10001)
    exact = model_exact(t, R0, s0)
    sigmoid = sigmoid_approx(t, R0, s0)
    tworse, worse = new_worse_approx(R0, s0)

    if axs is None:
        fig, axs = pl.subplots(1,2)

    axs[0].plot(t, exact, lw=3, c='k', alpha=0.3, label='exact')
    axs[0].plot(tnew, new, label='new ')
    axs[0].plot(t, sigmoid, '-.', label='old')

    #axs[0].plot(tworse, worse)

    axs[1].plot(t[:-1], np.diff(exact)/np.diff(t), lw=3, c='k', alpha=0.3, label='exact')
    axs[1].plot(tnew[:-1], np.diff(new)/np.diff(tnew),  label='new')
    axs[1].plot(t[:-1], np.diff(sigmoid)/np.diff(t), '-.', label='old')

    axs[0].set_xlim(t[[0,-1]])
    axs[1].set_xlim(t[[0,-1]])

    axs[0].set_title(f'R$_0$ = {R0:3.1f}')
    #axs[1].plot(tworse[:-1], np.diff(worse)/np.diff(tworse))


def plot_pade_approximation(R0):
    pl.figure()

    Omega = 1-1/R0**2
    rs = np.logspace(-4, np.log10(Omega),101)
    pl.plot(rs, np.sqrt(1+rs**2*R0**2/4),lw=3)
    pl.plot(rs, (1+rs**2*R0**2/8)/(1-rs**2*R0**2/16))
    pl.plot(rs, (3*rs**2*R0**2/4+4)/(rs**2*R0**2/4+4))


def outbreak_size_reduction(R0, Re):
    return (R0**2 - Re**2) / (R0**2 -1)/Re**2

def outbreak_size_reduction_upper(R0, Re):
    return 1-(1-np.exp(-Re)/(1-np.exp(-R0)))

def plot_outbreak_reduction(R0s):
    fig, ax = pl.subplots(1,1,figsize=(3.5,3.5))
    for iR0, R0 in enumerate(R0s):
        Re = np.linspace(1, R0, 101)
        red_ex = 1-model_final(Re) / model_final([R0])[0]
        red_re = outbreak_size_reduction(R0, Re)
        #red_exp = outbreak_size_reduction_upper(R0, Re)
        label = 'exact' if iR0 == 0 else ''
        pl.plot(Re, red_ex, lw=3,c='k',alpha=0.3,label=label)
        pl.plot(Re, red_re, label=f'estimation for $R_0$ = {R0:3.1f}')
        #pl.plot(Re, red_exp,ls='-.')
    ax.set_xlabel('reduced reproduction number R$_e$')
    ax.set_ylabel(r'outbreak size reduction $\rho_\Omega$')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()
    return ax

def plot_t_inflection(s0=1-1e-4):
    fig, ax = pl.subplots(1,1,figsize=(3,4))

    #R0s = np.logspace(0.01,1.3,101)
    R0s = 1+np.logspace(-2,np.log10(99),301)

    t_sigmoid = []
    t_exact = []
    t_new = []
    t_new_half_Omega = []
    t_new_r_infty = []

    for iR0, R0 in tqdm(enumerate(R0s)):
        t_ex = find_time_of_inflection_exact(R0, s0)
        #t_nw = find_true_time_of_inflection_approx(R0, s0)
        t_nw = find_inflection_t_and_r_from_new_approx(s0,R0)[0]
        #t_nw2 = compute_expression(1/2-1/2/R0**2, s0, R0)
        #t_nw3 = compute_expression(1-1/R0, s0, R0)
        t_si = find_time_of_inflection_old(R0, s0)

        t_sigmoid.append(t_si)
        t_exact.append(t_ex)
        t_new.append(t_nw)
        #t_new_half_Omega.append(t_nw2)
        #t_new_r_infty.append(t_nw3)

    t_sigmoid = np.array(t_sigmoid)
    t_exact = np.array(t_exact)
    t_new = np.array(t_new)
    #t_new_half_Omega = np.array(t_new_half_Omega)
    #t_new_r_infty = np.array(t_new_r_infty)
    ax.plot(R0s-1, t_exact,lw=3,c='k',alpha=0.3,label='exact')
    new_handle, = ax.plot(R0s-1, t_new, label='new approx.')
    sig_handle, = ax.plot(R0s-1, t_sigmoid, '-.', label='old approx.')
    #half_om_handle, = ax.plot(R0s-1, t_new_half_Omega, label='new approx.')
    #_handle, = ax.plot(R0s-1, t_new_r_infty, label='new approx.')

    iax = ax.inset_axes([0.2,0.15,0.35,0.5])
    iax.plot(R0s-1, np.abs(1-t_new/t_exact), color=new_handle.get_color())
    iax.plot(R0s-1, np.abs(1-t_sigmoid/t_exact), '-.', color=sig_handle.get_color())
    #iax.plot(R0s-1, np.abs(1-t_new_half_Omega/t_exact))
    #iax.plot(R0s-1, np.abs(1-t_new_r_infty/t_exact))
    iax.plot([5,5],[0.0,0.13],':',c='k',lw=1,alpha=0.5)
    iax.plot([R0s[0]-1,5],[0.03,0.03],':',lw=1,c='k',alpha=0.5)
    iax.plot([R0s[0]-1,5],[0.13,0.13],':',lw=1,c='k',alpha=0.5)
    iax.set_yticks([0,0.03,0.13,0.2,0.3])
    iax.set_ylim(0,0.3)
    iax.tick_params(axis='both', which='major', labelsize='small')

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.set_xlabel('basic reproduction number $R_0-1$')
    ax.set_ylabel('time of peak prevalence')
    iax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    iax.set_xscale('log')
    iax.text(0.03,0.97,'rel.err.',transform=iax.transAxes,va='top')

    ax.legend(loc='upper right')

    fig.tight_layout()

    return ax

def plot_approximations():
    fig, ax = pl.subplots(1,1,figsize=(4,3.75))
    x = np.linspace(0.001,1,1001)
    ax.plot(x, -np.log(x),ls='-',lw=4,c='k',alpha=0.3,label=' -ln(s)')
    ax.plot(x, 1/np.sqrt(x)-np.sqrt(x),ls='-',label=r'1/√s - √s')
    ax.plot(x, 2/x**(1/4)-2*x**(1/4),ls='--',color=colors[2],label=r'2/s$^{1/4}$ - 2s$^{1/4}$')
    ax.plot(x, -np.log(x)*(1-x),ls='-',lw=1,color=colors[5],label=r' -(1-s)ln(s)')
    ax.set_xlabel('fraction susceptibles s')
    ax.set_ylim(0,4)
    ax.set_xlim(0,1)
    ax.legend()
    fig.tight_layout()
    return ax

if __name__=="__main__":
    axf = plot_final()
    axf.get_figure().savefig('./figures/./comparison_outbreak_size.pdf')
    fig, axs = pl.subplots(2,3,figsize=(7,4))
    #plot_epidemic(1.1,axs=axs)
    #plot_epidemic(1.4,axs=axs)
    #plot_epidemic(1.8,axs=axs)
    plot_epidemic(1.1,axs=axs[:,0])
    plot_epidemic(1.7,axs=axs[:,1])
    plot_epidemic(2.3,axs=axs[:,2])
    #plot_pade_approximation(1.9)

    axs[0,0].set_ylabel('share of recovereds r(t)')
    axs[1,0].set_ylabel('prevalence j(t)')
    axs[1,1].set_xlabel('time t')
    axs[0,0].legend()


    fig.tight_layout()
    fig.savefig('./figures/outbreak_dynamics.pdf')

    ax = plot_outbreak_reduction([1.1,1.7,2.3,5])
    ax.get_figure().savefig('./figures/outbreak_size_reduction.pdf')

    ax = plot_t_inflection()
    ax.get_figure().savefig('./figures/t_inflection.pdf')

    plot_epidemic(2.75)

    print(f"{outbreak_size_reduction(R0=2.5,Re=1.25)=}")
    print(f"{outbreak_size_reduction(R0=5.0,Re=2.50)=}")

    ax = plot_approximations()
    ax.get_figure().savefig('./figures/log-approx.pdf')

    pl.show()




