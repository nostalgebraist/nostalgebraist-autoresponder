"""helpers for the 'traceability' logs"""
from functools import wraps

import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import seaborn as sns

from persistence.traceability import load_traceability_logs_to_df

### moving avgs


def rollavg_convolve(a, n):
    "scipy.convolve"
    assert n % 2 == 1
    return sci.convolve(a, np.ones(n, dtype="float") / n, "same")[n // 2 : -n // 2 + 1]


def rollavg_convolve_edges(a, n):
    "scipy.convolve, edge handling"
    assert n % 2 == 1
    return sci.convolve(a, np.ones(n, dtype="float"), "same") / sci.convolve(
        np.ones(len(a)), np.ones(n), "same"
    )


def running_avg(a):
    return np.cumsum(a) / np.arange(1, len(a) + 1, 1)


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
    return len(mt["k"][0])


def mtresult(mt, start_ix=0):
    k, u, s, _ = loadmt(mt, True, True)
    return s[start_ix:].mean()


def mt_target_vs_result(df_, buffer=20):
    df = df_[df_.ntok_after_first_step > buffer]

    _, axes = plt.subplots(2, 1)

    filt = df.is_miro_v2

    ymin, ymax = df.mtresult.min() - 0.1, df.mtresult.max() + 0.1

    plt.sca(axes[0])
    sns.regplot(data=df[filt == False], x='mirotarg', y='mtresult',
                # hue='mlr', palette=['g', 'r'],
                )
    plt.ylim(ymin, ymax)
    plt.title("v1")

    plt.sca(axes[1])
    sns.regplot(data=df[filt], x='mirotarg', y='mtresult',
                # hue='mlr', palette=['g', 'r'],
                )
    plt.title("v2")
    plt.ylim(ymin, ymax)
    plt.show()


def _get_first_step_length(sampling_info: dict):
    for k in ["first_step_length", "pre_continue_length"]:
        if k in sampling_info:
            return sampling_info[k]


def mt_annotate(df):
    # miro + miro/breakruns shared
    df["mtlen"] = df.miro_traces.apply(nancall(mtlen))
    df["mtresult"] = df.miro_traces.apply(nancall(mtresult))
    df["mtdelta"] = df.mtresult - df.mirotarg
    df["is_miro_v2"] = df.model_info.apply(
        nancall(lambda d: d.get("sampling_info", {}).get("MIRO_V2", True))
    )
    df["mlr"] = df.model_info.apply(
        nancall(lambda d: d.get("sampling_info", {}).get("MIRO_LR"))
    )
    df["first_step_length"] = df.model_info.apply(
        nancall(lambda d: _get_first_step_length(d.get("sampling_info", {})))
    )
    df["use_first_step"] = df.model_info.apply(
        nancall(lambda d: d.get("sampling_info", {}).get("USE_FIRST_STEP", True))
    )
    df.loc[df.use_first_step == False, "first_step_length"] = 0

    df["mtresult_after_first_step"] = None
    filt = df.mtlen.notnull() & df.first_step_length > 0
    df.loc[filt, "mtresult_after_first_step"] = df.loc[df.mtlen.notnull(), :].apply(
        lambda row: mtresult(row.miro_traces, int(row.first_step_length)),
        axis=1
    )

    # BREAKRUNS
    df["is_breakruns"] = df.model_info.apply(
        nancall(lambda d: d.get("sampling_info", {}).get("BREAKRUNS", False))
    )
    df["breakruns_tau"] = df.model_info.apply(
        nancall(lambda d: d.get("sampling_info", {}).get("BREAKRUNS_TAU", None))
    )
    df["breakruns_decay"] = df.model_info.apply(
        nancall(lambda d: d.get("sampling_info", {}).get("BREAKRUNS_DECAY", None))
    )
    df.loc[
        (df.breakruns_decay.isnull()) & (df.is_breakruns == True),
        "breakruns_decay"
    ] = 0.

    # generic
    for name in ["T", "p", "first_step_T", "first_step_p"]:
        df[f"sampling_{name}"] = df.model_info.apply(
            nancall(lambda d: d.get("sampling_info", {}).get(name, None))
        )

    # TODO: verify
    df["ntok_after_first_step"] = None
    df.loc[df.mtlen.notnull(), "ntok_after_first_step"] = (
        df.loc[df.mtlen.notnull(), "mtlen"]
        - df.loc[df.mtlen.notnull(), "first_step_length"]
    )

    # mtplotarg
    df["mtplotarg"] = [
        (mt, v2, br, pcl, T, tau, fsT)
        for mt, v2, br, pcl, T, tau, fsT in zip(
            df.miro_traces.values,
            df.is_miro_v2.values,
            df.is_breakruns,
            df.first_step_length,
            df.sampling_T,
            df.breakruns_tau,
            df.sampling_first_step_T
        )
    ]
    return df


def annotated_load_traceability_logs(*args, **kwargs):
    df = load_traceability_logs_to_df(*args, **kwargs)
    df = mt_annotate(df)
    return df


def loadmt(mt, v2, br):
    k = np.array(mt["k"][0])
    mu = np.array(mt["mu"][0])
    s = np.array(mt["surprise"][0])
    mirotarg = mt["mu"][0][0]
    if (not v2) and (not br):
        mirotarg = mirotarg / 2
    return k, mu, s, mirotarg


def mtshow(mt, v2, br, mtstart, T, tau, fsT,
           figsize=(12, 8)):
    mtstart = int(mtstart)
    k, u, s, mirotarg = loadmt(mt, v2, br)

    fix, axes = plt.subplots(3, 1, figsize=figsize)

    plt.sca(axes[0])
    s_avg = running_avg(s)
    plt.plot(s_avg, label="running_avg")
    s_smoothed = rollavg_convolve_edges(s, 11)
    plt.plot(s_smoothed, alpha=1, label="smoothed")
    plt.plot(s, alpha=0.75, marker="o", markersize=2, lw=0)
    plt.axhline(mirotarg, ls="--", lw=1, c="k", label=f"targ={mirotarg:.2f}")
    axes[0].axvline(mtstart, ls="--", c="r", alpha=0.75)
    axes[0].set_ylim(np.percentile(s_smoothed, 2), np.percentile(s_smoothed, 98))
    axes[0].set_title("surprise")
    plt.legend()

    axes[1].plot(u)
    axes[1].axvline(mtstart, ls="--", c="r", alpha=0.75)
    axes[1].set_title("mu")

    if br:
        eff_T = (u * tau) + T
        eff_T[:mtstart] = fsT
        axes[2].plot(eff_T, marker="o", markersize=3, lw=1, ls="--")
        axes[2].axvline(mtstart, ls="--", c="r", alpha=0.75)
        axes[2].axhline(1., ls="--", c="k", alpha=0.75)
        axes[2].set_title("eff_T")
    else:
        axes[2].plot(k + 1, marker="o", markersize=3, lw=1, ls="--")
        axes[2].axvline(mtstart, ls="--", c="r", alpha=0.75)
        axes[2].set_yscale("log")
        axes[2].set_title("k")

    plt.tight_layout()
    plt.show()


def mtphaseshow(mt, v2, br, mtstart):
    k, u, s, _ = loadmt(mt, v2, br)

    k, u, s = k[mtstart:], u[mtstart:], s[mtstart:]

    fix, axes = plt.subplots(2, 1)
    plt.sca(axes[0])
    s_avg = running_avg(s)
    plt.scatter(s_avg, u, s=1)
    plt.xlabel("surprise running_avg")
    plt.ylabel("mu")

    for ix, w in zip(
        [
            1,
        ],
        [
            11,
        ],
    ):
        plt.sca(axes[ix])
        s_smoothed = rollavg_convolve_edges(s, w)
        plt.scatter(s_smoothed, u, s=1)

        plt.xlabel(f"surprise smoothed @ {w}")
        plt.ylabel("mu")

    plt.tight_layout()
    plt.show()
