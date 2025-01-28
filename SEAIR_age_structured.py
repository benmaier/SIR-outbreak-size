import numpy as np
import epipack as epk

from tqdm import tqdm

from matplotlib.ticker import PercentFormatter

from get_contact_matrix import countries, load_contact_data

from scipy.optimize import newton

from bfmplot.sequences import bochenska2 as colors

#Nage = 10
#Ns = np.loadtxt('data/population.csv',skiprows=1,delimiter=',')[:,1]



#CI = np.ones((Nage, Nage))
#CA = np.ones((Nage, Nage))
#CI = np.eye(Nage)
#CA = np.eye(Nage)
#Ns = np.ones(Nage)/Nage

_i0 = 1e-7

def exact_SIR_Omega(R0):
    func = lambda r:  1 - r - np.exp(-R0*r)
    fprm = lambda r: -1 + np.exp(-R0*r)*R0
    Omega = newton(func, 0.5, fprm)
    return Omega


def model_SIR(t, R0, i00=_i0):
    s0 = 1-_i0
    rho = 1/3
    model = epk.SIRModel(R0*rho, recovery_rate=rho)\
               .set_initial_conditions({'S':s0, 'I': 1-s0})
    res = model.integrate(t)
    return res['R']

def get_relative_R_timeseries(res,N):
    Rcomp = filter(lambda s: s[0] == 'R', res.keys())
    tot = 0
    for comp in Rcomp:
        tot += res[comp]
    return tot / N

def get_relative_R_end(res,N):
    r = get_relative_R_timeseries(res,N)
    return r[-1]

def get_model_raw(R0=1,infection_rate_norm=1,i00=_i0,CI=None,CA=None,Ns=None):

    if CI is None:
        CI = np.loadtxt('contact_matrices/DE_four_age_cohorts/contact_matrix.csv',skiprows=1,delimiter=',')[:,1:]
        CA = CI
        Ns = np.loadtxt('contact_matrices/DE_four_age_cohorts/population.csv',skiprows=1,delimiter=',')[:,1]

    norm = infection_rate_norm
    s00 = 1-i00
    Nage = len(Ns)

    S = [f'S{i}' for i in range(Nage)]
    E = [f'E{i}' for i in range(Nage)]
    A = [f'A{i}' for i in range(Nage)]
    I = [f'I{i}' for i in range(Nage)]
    R = [f'R{i}' for i in range(Nage)]

    rate_E = np.ones(Nage) / 1
    rate_A = np.ones(Nage) / 1
    rate_I = np.ones(Nage) / 2

    compartments = S + E + A + I + R

    model = epk.MatrixEpiModel(compartments)

    for i in range(Nage):
        for j in range(Nage):
            model.add_transmission_processes([
                    (f'I{j}', f'S{i}', R0*CI[i,j]/Ns[j]/norm, f'I{j}', f'E{i}'),
                    (f'A{j}', f'S{i}', R0*CA[i,j]/Ns[j]/norm, f'A{j}', f'E{i}'),
                ])

    for i in range(Nage):
        model.add_transition_processes([
                (f'E{i}', rate_E[i], f'A{i}'),
                (f'A{i}', rate_A[i], f'I{i}'),
                (f'I{i}', rate_I[i], f'R{i}'),
            ])

    initial_conditions = {}
    for i in range(Nage):
        initial_conditions.update({ f'S{i}': Ns[i]*s00, f'I{i}': Ns[i]*i00 })

    model.set_initial_conditions(initial_conditions)

    return model


def get_model(R0,i00=_i0):
    model = get_model_raw(i00=0) # setup in disease-free state to compute R0
    model_R0_initial = model.get_next_generation_matrix_leading_eigenvalue()
    #print(f"{model_R0_initial=}")
    norm = model_R0_initial
    model = get_model_raw(R0=R0,infection_rate_norm=norm,i00=i00)
    model_R0_final = model.get_next_generation_matrix_leading_eigenvalue()
    #print(f"{model_R0_final=}")
    return model


def get_response_curve(R0s,i00=_i0,CI=None,CA=None,Ns=None):
    model_initial = get_model_raw(i00=0,CI=CI,CA=CA,Ns=Ns) # setup in disease-free state to compute R0
    model_R0_initial = model_initial.get_next_generation_matrix_leading_eigenvalue()
    norm = model_R0_initial
    Omegas = np.zeros_like(R0s)
    new_R0s = np.zeros_like(R0s)
    for i, R0 in enumerate(tqdm(R0s)):
        model = get_model_raw(R0=R0,infection_rate_norm=norm,i00=i00,CI=CI,CA=CA,Ns=Ns)
        R0_calibrated = model.get_next_generation_matrix_leading_eigenvalue()
        res = model.integrate([0,1e9])
        Omegas[i] = get_relative_R_end(res,Ns.sum())
        new_R0s[i] = R0_calibrated

    return new_R0s, Omegas


def plot_response_curve(CI=None,CA=None,Ns=None,ax=None):
    if ax is None:
        fig, ax = pl.subplots(1,1,figsize=(4,3))

    R0s = np.linspace(1.05,5,30)
    new_R0s, Omega = get_response_curve(R0s,_i0,CI,CA,Ns)
    ax.plot(new_R0s, Omega,'s',ms=5,mfc='None',label='age-structured SEAIR')
    Omega_est = 1-1/R0s**2

    R0th = np.linspace(1,5,1001)
    ax.plot(R0th, 1-1/R0th**2,label=r'1-1/R$_0^2$')
    ax.legend()

    ax.set_xlabel('basic reproduction number R$_0$')
    ax.set_ylabel(r'outbreak size $\Omega$')

    #iax = ax.inset_axes([0.6,0.3,0.35,0.4])
    #iax.plot(R0s, np.abs(1-Omega_est/Omega), 's:',ms='4',mfc='None')
    #iax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    #iax.text(0.03,0.97,'rel.err.',transform=iax.transAxes,va='top')

    fig.tight_layout()
    return ax, {'R0':new_R0s.tolist(), 'Omega': Omega.tolist()}

def make_analysis_for_all_countries():
    results = []
    for country in countries.codes:
        C, N = load_contact_data(country)
        ax, result = plot_response_curve(C, 0.5*C, N)
        ax.set_title(country)
        result['countrycode'] = country
        result['country'] = countries.name(country)
        results.append(result)

    return results

def make_all_country_analysis_and_save_as_json():
    results = make_analysis_for_all_countries()
    with open('./result_data/outbreak_size_for_countries.json', 'w') as f:
        json.dump(results, f)

def load_and_plot_country_analyses():

    fig, ax = pl.subplots(3,3,figsize=(6,5),sharex=True,sharey=True)
    ax = ax.flatten()
    with open('./result_data/outbreak_size_for_countries.json', 'r') as f:
        countries = json.load(f)

    for i, country in enumerate(countries):
        R0 = np.array(country['R0'])
        Omega = np.array(country['Omega'])
        handle, = ax[i].plot(R0, Omega, 's',mfc='w',label='SEAIR')
        R0s = np.linspace(1,R0[-1],101)
        ax[i].plot(R0s,1-1/R0s**2,c='k',label='1-1/R$_0^2$')
        ax[i].text(1.0,0.03,country['country'],transform=ax[i].transAxes,ha='right',va='bottom',fontsize='medium')
        ax[i].set_xlim(1,5)
        ax[i].set_xticks([1,5])
        ax[i].set_yticks([0,1])

        ax[-1].plot(R0, Omega,'.',c=handle.get_color(),alpha=0.2)
        #ax[-1].plot(R0, 1-Omega)
        #ax[-1].plot(R0s,+1/R0s**2) 
    ax[0].legend()
    ax[-1].set_xlim(1,5)
    ax[-1].set_xticks([1,5])
    ax[-1].set_ylim(0,1)
    ax[-1].set_yticks([0,1])
    ax[-1].plot(R0s,1-1/R0s**2,c='k')
    fig.tight_layout()
    ax[-1].set_xlim(1,5)
    ax[-1].text(1.0,0.03,'All countries',transform=ax[-1].transAxes,ha='right',va='bottom',fontsize='medium')
    ax[-2].set_xlabel('R$_0$',labelpad=-7)
    ax[3].set_ylabel('outbreak size Ω',labelpad=-5)
    #ax[-1].set_xscale('log')
    #ax[-1].set_yscale('log')

    return ax

def country_errors():
    fig, ax = pl.subplots()

    with open('./result_data/outbreak_size_for_countries.json', 'r') as f:
        countries = json.load(f)

    for i, country in enumerate(countries):
        R0 = np.array(country['R0'])
        Omega = np.array(country['Omega'])
        R0s = R0
        heuristic = 1-1/R0s**2
        SIR_exact = np.array(list(map(exact_SIR_Omega, R0s)))

        ax.plot(R0s, np.abs(1-heuristic/Omega),'s',c=colors[0],alpha=0.4)
        ax.plot(R0s, np.abs(1-SIR_exact/Omega),'.',c=colors[1],alpha=0.4)
        ax.plot(R0s, np.abs(1-heuristic/Omega),c=colors[0],alpha=0.2)
        ax.plot(R0s, np.abs(1-SIR_exact/Omega),c=colors[1],alpha=0.2)
        #ax.plot(R0s, np.abs(heuristic-Omega),'s',c=colors[0],alpha=0.4)
        #ax.plot(R0s, np.abs(SIR_exact-Omega),'.',c=colors[1],alpha=0.4)
        #ax.plot(R0s, np.abs(heuristic-Omega),c=colors[0],alpha=0.2)
        #ax.plot(R0s, np.abs(SIR_exact-Omega),c=colors[1],alpha=0.2)

        print(country['country'], np.all(np.abs(1-heuristic/Omega) <= np.abs(1-SIR_exact/Omega)))
    ax.set_yscale('log')

    return ax

if __name__ == "__main__":
    import matplotlib.pyplot as pl
    import json
    #R0 = 1
    #model = get_model(R0)
    #t = np.linspace(0,10000,1001)
    #res = model.integrate(t)
    #R = get_relative_R_timeseries(res)
    #pl.plot(t, R)

    #R = model_SIR(t,R0)
    #pl.plot(t, R)


    #ax = plot_response_curve()
    #ax.get_figure().savefig('SEAIR_outbreaksize.pdf')
    #pl.show()

    #pl.show()
    ax = load_and_plot_country_analyses()
    ax[0].get_figure().savefig('./figures/SEAIR_outbreak_sizes_countries.pdf')
    #make_all_country_analysis_and_save_as_json()

    ax = country_errors()
    pl.show()
