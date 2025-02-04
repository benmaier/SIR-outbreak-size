import numpy as np
import epipack as epk

from tqdm import tqdm

from matplotlib.ticker import PercentFormatter

from get_contact_matrix import countries, load_contact_data

from scipy.optimize import newton

from bfmplot.sequences import bochenska2 as colors, markers
from bfmplot.tools import align_legend_right

from theory import sigmoid_final, complicated_new_final, outbreak_size_reduction

#Nage = 10
#Ns = np.loadtxt('data/population.csv',skiprows=1,delimiter=',')[:,1]



#CI = np.ones((Nage, Nage))
#CA = np.ones((Nage, Nage))
#CI = np.eye(Nage)
#CA = np.eye(Nage)
#Ns = np.ones(Nage)/Nage

_i0 = 1e-13

def exact_SIR_Omega(R0):

    is_single = not hasattr(R0,'__len__')
    if is_single:
        R0 = [R0]
    Omegas = np.zeros_like(R0)
    for i, _R0 in enumerate(R0):
        func = lambda r:  1 - r - np.exp(-_R0*r)
        fprm = lambda r: -1 + np.exp(-_R0*r)*_R0
        Omegas[i] = newton(func, 0.5, fprm)
    if is_single:
        return Omegas[0]
    else:
        return Omegas


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

    R0s = np.linspace(1.1,10,60)
    new_R0s, Omega = get_response_curve(R0s,_i0,CI,CA,Ns)
    ax.plot(new_R0s, Omega,'s',ms=5,mfc='None',label='age-structured SEAIR')
    Omega_est = 1-1/R0s**2

    R0th = np.linspace(1,10,1001)
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
        handle, = ax[i].plot(R0, Omega, ls='None', marker=markers[i],mfc='w',label='SEAIR')
        R0s = np.linspace(1,R0[-1],101)
        ax[i].plot(R0s,1-1/R0s**2,c='k',label='1-1/R$_0^2$')
        #ax[i].plot(R0s,complicated_new_final(R0s),c='k',label='1-1/R$_0^2$')
        #ax[i].plot(R0s,1-1/R0s,'--',c='k',label='1-1/R$_0^2$')
        ax[i].text(1.0,0.03,country['country'],transform=ax[i].transAxes,ha='right',va='bottom',fontsize='medium')
        ax[i].set_xlim(1,5)
        ax[i].set_xticks([1,5])
        ax[i].set_yticks([0,1])

        ax[-1].plot(R0, Omega,ls='None',marker=markers[i],c=handle.get_color(),alpha=0.1)
        #ax[-1].plot(R0, 1-Omega)
        #ax[-1].plot(R0s,+1/R0s**2) 
    ax[0].legend()
    #ax[-1].plot(R0s,1-np.exp(-R0s),':',c='k',lw=1,label='1-exp(-R$_0$)')
    ax[-1].set_xlim(1,5)
    ax[-1].set_xticks([1,5])
    ax[-1].set_ylim(0,1)
    ax[-1].set_yticks([0,1])
    ax[-1].plot(R0s,1-1/R0s**2,c='k')
    fig.tight_layout()
    ax[-1].set_xlim(1,5)
    ax[-1].text(1.0,0.03,'All countries',transform=ax[-1].transAxes,ha='right',va='bottom',fontsize='medium')
    ax[-2].set_xlabel('R$_0$',labelpad=-7)
    ax[3].set_ylabel('outbreak size Ω',labelpad=-3)

    ax[-1].legend()
    #ax[-1].set_xscale('log')
    #ax[-1].set_yscale('log')

    return ax

def plot_country_reduction_old():
    fig, ax = pl.subplots(2,4,figsize=(11,3.5),sharex=True,sharey=True)
    ax = ax.flatten()
    with open('./result_data/outbreak_size_for_countries.json', 'r') as f:
        countries = json.load(f)

    for i, country in enumerate(countries):
        R0 = np.array(country['R0'])
        Omega = np.array(country['Omega'])

        ndx = [7, 14, 24, 40, 50]
        markers = ['o','s','d','x','>']

        for jn, n in enumerate(ndx):
            R0_base = R0[n]
            Om_base = Omega[n]
            Re = R0[:n+1]
            thRe = np.linspace(1,R0_base,1001)
            Om = Omega[:n+1]
            red = 1-Om/Om_base
            handle, = ax[i].plot(Re+2**(jn)-1, red, 's', marker=markers[jn],ms=3, alpha=0.5)
            ax[i].plot(thRe+2**(jn)-1, outbreak_size_reduction(R0_base, thRe),color=handle.get_color())

        ax[i].set_title(country['country'],fontsize="small")


        if i>3:
            ax[i].set_xlabel('reduced reprod. no. R$_e$ + offs.')

        if i == 0 or i == 4:
            ax[i].set_ylabel('outbreak size\nreduction')
        #ax[i].set_xscale('log')
        #ax[i].set_yscale('log')
        ax[i].set_ylim(1e-3,1)
        ax[i].set_xlim(1,20)

    fig.tight_layout()
    return ax

def plot_country_reduction():
    fig, ax = pl.subplots(1,5,figsize=(11,1.8),sharex=False,sharey=True)
    ax = ax.flatten()
    with open('./result_data/outbreak_size_for_countries.json', 'r') as f:
        countries = json.load(f)

    for i, country in enumerate(countries):
        ii = 5*i
        R0 = np.array(country['R0'])
        Omega = np.array(country['Omega'])

        ndx = [7, 14, 24, 40, 50]

        for jn, n in enumerate(ndx):
            R0_base = R0[n]
            Om_base = Omega[n]
            Re = R0[:n+1]
            thRe = np.linspace(1,R0_base,1001)
            Om = Omega[:n+1]
            red = 1-Om/Om_base
            handle, = ax[jn].plot(Re, red, ls='None',marker=markers[i],ms=3, alpha=0.15,color=colors[jn])
            if i == len(countries)-1:
                label = r'$\frac{R_0^2-R_e^2}{R_e^2(R_0^2-1)}$'
                ax[jn].plot(thRe, outbreak_size_reduction(R0_base, thRe),color='k',alpha=1,lw=1.5,
                            label=label)
            ax[jn].set_xlim(1, R0_base)

            ax[jn].set_title(f'R$_0$ = {R0_base:3.1f}')
            #ax[jn].set_xscale('log')
            ax[jn].set_xticks(np.arange(1,1+np.floor(R0_base)))

            #ax[ii+jn].set_xticks([1,(R0_base)/2])
            #if i > 3:
            #    ax[ii+jn].set_xticklabels([1,r'$\frac{R_0}{2}$'])
            #else:
            #    ax[ii+jn].set_xticklabels(['',''])
        #ax[ii].set_title(country['country'],fontsize="small")




    ax[0].set_ylabel('outbreak size\nreduction $\\rho_\\Omega$',loc='bottom')
    ax[0].set_ylim(0,1)
    ax[0].legend(loc=(0.37,0.5),fontsize='x-large',)
    #ax[0].set_ylim(1e-3,1)
    #ax[0].set_yscale('log')
        #ax[i].set_xlim(1,20)
    ax[0].yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))

    ax[2].set_xlabel('reduced reproduction number R$_e$')

    fig.tight_layout()
    pl.subplots_adjust(wspace=0.1)

    return ax

def plot_country_reduction_2():
    fig, ax = pl.subplots(2,4*5,figsize=(11,3),sharex=False,sharey=True)
    ax = ax.flatten()
    with open('./result_data/outbreak_size_for_countries.json', 'r') as f:
        countries = json.load(f)

    for i, country in enumerate(countries):
        ii = 5*i
        R0 = np.array(country['R0'])
        Omega = np.array(country['Omega'])

        ndx = [7, 14, 24, 40, 50]
        markers = ['o','s','d','x','>']

        for jn, n in enumerate(ndx):
            R0_base = R0[n]
            Om_base = Omega[n]
            Re = R0[:n+1]
            thRe = np.linspace(1,R0_base,1001)
            Om = Omega[:n+1]
            red = 1-Om/Om_base
            handle, = ax[ii+jn].plot(Re, red, ls='None',marker=markers[jn],ms=3, alpha=0.5,color=colors[jn])
            ax[ii+jn].plot(thRe, outbreak_size_reduction(R0_base, thRe),color='k',alpha=1,lw=1)
            ax[ii+jn].set_xlim(1, R0_base)

            if i == 0:
                a = ax[ii+jn]
                a.text(0.05,0.05,f'R$_0$ = {R0_base:3.1f}',transform=a.transAxes,fontsize='small',rotation=90)

            ax[ii+jn].set_xticks([1,(R0_base)/2])
            if i > 3:
                ax[ii+jn].set_xticklabels([1,r'$\frac{R_0}{2}$'])
            else:
                ax[ii+jn].set_xticklabels(['',''])
        ax[ii].set_title(country['country'],fontsize="small")



        if i == 0 or i == 4:
            ax[ii].set_ylabel('outbreak size\nreduction')
        ax[i].set_yscale('log')
        ax[i].set_ylim(1e-3,1)
        #ax[i].set_xlim(1,20)

    ax[29].set_xlabel('reduced reprodroduction number R$_e$')

    fig.tight_layout()
    pl.subplots_adjust(wspace=0)

    return ax

def country_errors():
    fig, ax = pl.subplots(2,4,figsize=(11,3.5))
    ax = ax.flatten()

    with open('./result_data/outbreak_size_for_countries.json', 'r') as f:
        countries = json.load(f)


    handles_1 = []
    labels_1 = []
    handles_2 = []
    labels_2 = []
    for i, country in enumerate(countries):
        R0 = np.array(country['R0'])
        Omega = np.array(country['Omega'])
        R0s = R0
        heuristic = 1-1/R0s**2
        SIR_exact = exact_SIR_Omega(R0s)
        exp_heur  = 1-np.exp(-R0s)
        sig_final = sigmoid_final(R0s)
        apprx2 = complicated_new_final(R0s)

        heuristics = [exp_heur, SIR_exact, sig_final, heuristic, apprx2]
        markers = ['o','s','d','x','>']
        #labels = [r'1-1/R$_0^2$', 'SIR exact', '1-exp(-R$_0$)', '2(1-1/R$_0$)/R$_0$']
        label = ['1-exp(-R$_0$)', 'SIR exact', '2(1-1/R$_0$)/R$_0$', r'1-1/R$_0^2$', 'Ω2']

        for j, heur in enumerate(heuristics):
            #for k, heur_backgr in enumerate(heuristics):
            #    if k == j:
            #        continue

            #    ax[i].plot(R0s, np.abs(1-heur_backgr/Omega),marker=markers[k],c='grey',alpha=0.1)
            #    ax[i].plot(R0s, np.abs(1-heur_backgr/Omega),c='grey',alpha=0.1)
            handle1, = ax[i].plot(R0s, np.abs(1-heur/Omega),marker=markers[j],c=colors[j],alpha=0.4,ms=3)
            handle2, = ax[i].plot(R0s, np.abs(1-heur/Omega),c=colors[j],alpha=1)
            ax[i].set_yscale('log')

            if i == 0:
                if j < 2:
                    handles_1.append((handle1, handle2))
                    labels_1.append(label[j])
                else:
                    handles_2.append((handle1, handle2))
                    labels_2.append(label[j])
        ax[i].set_title(country['country'],fontsize='small')
        #ax.plot(R0s, np.abs(heuristic-Omega),'s',c=colors[0],alpha=0.4)
        #ax.plot(R0s, np.abs(SIR_exact-Omega),'.',c=colors[1],alpha=0.4)
        #ax.plot(R0s, np.abs(heuristic-Omega),c=colors[0],alpha=0.2)
        #ax.plot(R0s, np.abs(SIR_exact-Omega),c=colors[1],alpha=0.2)

    leg = ax[0].legend(handles_1, labels_1, frameon=True, edgecolor='None',borderpad=0)
    leg.get_frame().set_facecolor((1,1,1,0.05))
    align_legend_right(leg)
    leg = ax[4].legend(handles_2, labels_2, frameon=True, edgecolor='None', borderpad=0)
    leg.get_frame().set_facecolor((1,1,1,0.05))
    align_legend_right(leg)

        #print(country['country'], np.all(np.abs(1-heuristic/Omega) <= np.abs(1-SIR_exact/Omega)))
    for j in range(8):
        ax[j].set_ylim(1e-3,1)
        ax[j].grid()
        ax[j].set_xlim(1,10)
        ax[j].set_xticks([1,2,4,6,8,10])

    for a in ax[4:]:
        a.set_xlabel('basic reproduction number R$_0$')

    ax[0].set_ylabel('relative error')
    ax[4].set_ylabel('relative error')

    fig.tight_layout()

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

    #make_analysis_for_all_countries()

    ax = country_errors()
    ax[0].get_figure().savefig('./figures/country_errors.pdf')

    ax = plot_country_reduction()
    ax[0].get_figure().savefig('./figures/country_reductions.pdf')

    print(f"{exact_SIR_Omega(R0=5.0)=}")
    pl.show()
