"""helpers for the 'traceability' logs"""
from functools import wraps

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from traceability import load_traceability_logs_to_df

### moving avgs

def rollavg_convolve(a,n):
    'scipy.convolve'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float')/n, 'same')[n//2:-n//2+1]

def rollavg_convolve_edges(a,n):
    'scipy.convolve, edge handling'
    assert n%2==1
    return sci.convolve(a,np.ones(n,dtype='float'), 'same')/sci.convolve(np.ones(len(a)),np.ones(n), 'same')

def running_avg(a):
    return np.cumsum(a) / np.arange(1, len(a)+1, 1)

### decreasing pandas pain

def isna(x):
    return x is None or x != x

def nancall(f):
    @wraps(f)
    def _f(arg, *args, **kwargs):
        if isna(arg):
            return None
        return f(arg, *args, **kwargs)
    return _f

### mirotarg stuff

def mtlen(mt):
    return len(mt['k'][0])

def mtresult(mt):
    k, u, s, _ = loadmt(mt, True)
    return s.mean()

def mt_target_vs_result(df_, buffer=20):
    df = df_[df_.ntok_miro > buffer]

    _, axes = plt.subplots(2, 1)

    filt = df.is_miro_v2

    ymin, ymax = df.mtresult.min()-0.1, df.mtresult.max()+0.1

    plt.sca(axes[0])
    plt.scatter(df[filt==False].mirotarg, df[filt==False].mtresult, c='g', label='v1')
    # plt.axhline(0, ls='--', c='k', alpha=0.75)
    plt.ylim(ymin, ymax)
    plt.title('v1')

    plt.sca(axes[1])
    plt.scatter(df[filt].mirotarg, df[filt].mtresult, c='r', label='v2')
    # plt.axhline(0, ls='--', c='k', alpha=0.75)
    plt.title('v2')
    plt.ylim(ymin, ymax)
    plt.show()

def mt_annotate(df):
    df['mtlen'] = df.miro_traces.apply(nancall(mtlen))
    df['mtresult'] = df.miro_traces.apply(nancall(mtresult))
    df['mtdelta'] = df.mtresult - df.mirotarg
    df['is_miro_v2'] = df.model_info.apply(nancall(lambda d: d.get('sampling_info', {}).get('MIRO_V2', True)))
    df['mlr'] = df.model_info.apply(nancall(lambda d: d.get('sampling_info', {}).get('MIRO_LR')))
    df['pre_continue_length'] = df.model_info.apply(nancall(lambda d: d.get('sampling_info', {}).get('pre_continue_length', True)))
    df['ntok_miro'] = None
    df.loc[df.mtlen.notnull(), 'ntok_miro'] = df.loc[df.mtlen.notnull(), "mtlen"]- df.loc[df.mtlen.notnull(), "pre_continue_length"]
    df['mtplotarg'] = [(mt, v2, pcl)
                       for mt, v2, pcl in zip(df.miro_traces.values,
                                              df.is_miro_v2.values,
                                              df.pre_continue_length.values)
                       ]
    return df

def annotated_load_traceability_logs(*args, **kwargs):
    df = load_traceability_logs_to_df(*args, **kwargs)
    df = mt_annotate(df)
    return df

def loadmt(mt, v2):
    k = np.array(mt['k'][0])[1:]
    mu = np.array(mt['mu'][0])[1:]
    s = np.array(mt['surprise'][0])[1:]
    mirotarg = mt['mu'][0][0]
    if not v2:
        mirotarg = mirotarg/2
    return k, mu, s, mirotarg

def mtshow(mt, v2, mtstart):
    k, u, s, mirotarg = loadmt(mt, v2)

    fix, axes = plt.subplots(3, 1)

    plt.sca(axes[0])
    s_avg = running_avg(s)
    plt.plot(s_avg, label='running_avg')
    s_smoothed = rollavg_convolve_edges(s, 11)
    plt.plot(s_smoothed, alpha=1, label='smoothed')
    plt.plot(s, alpha=0.75, marker='o', markersize=2, lw=0)
    plt.axhline(mirotarg, ls='--', lw=1, c='k', label=f'targ={mirotarg:.2f}')
    axes[0].axvline(mtstart, ls='--', c='r', alpha=0.75)
    plt.legend()

    axes[0].set_ylim(np.percentile(s_smoothed, 5), np.percentile(s_smoothed, 95))
    axes[0].set_title('surprise')
    axes[1].plot(u)
    axes[1].axvline(mtstart, ls='--', c='r', alpha=0.75)
    axes[1].set_title('mu')

    axes[2].plot(k, marker='o', markersize=3, lw=1, ls='--')
    axes[2].axvline(mtstart, ls='--', c='r', alpha=0.75)
    axes[2].set_yscale('log')
    axes[2].set_title('k')

    plt.tight_layout()
    plt.show()

def mtphaseshow(mt, v2, mtstart):
    k, u, s, _ = loadmt(mt, v2)

    k, u, s = k[mtstart:], u[mtstart:], s[mtstart:]

    fix, axes = plt.subplots(2, 1)
    plt.sca(axes[0])
    s_avg = running_avg(s)
    plt.scatter(s_avg, u, s=1)
    plt.xlabel('surprise running_avg')
    plt.ylabel('mu')

    for ix, w in zip([1,], [11,]):
        plt.sca(axes[ix])
        s_smoothed = rollavg_convolve_edges(s, w)
        plt.scatter(s_smoothed, u, s=1)

        plt.xlabel(f'surprise smoothed @ {w}')
        plt.ylabel('mu')

    plt.tight_layout()
    plt.show()
