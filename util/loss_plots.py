# loss tracking nov 2020
import warnings
from copy import deepcopy
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.markers import MarkerStyle

def calc_adam_bias_corrector(tmax=None, beta1=0.9, beta2=0.999, resets=[], times=None):
    if times is None:
        if tmax is None:
             raise ValueError("pass one of times or tmax")
        times = np.arange(1, tmax+1)
    times = pd.Series([t for t in times], index=[t for t in times])

    reset_times = [times.index.min()] + resets + [times.index.max()]

    for t1, t2 in zip(reset_times[:-1], reset_times[1:]):
        delta = times.loc[t1]-1
        vals = times.loc[t1:t2].values
        vals_c = np.arange(1, len(vals)+1)
        times.loc[t1:t2] = vals_c

    results = []

    numer = np.sqrt(1-np.power(beta2, times.values))
    denom = 1-np.power(beta1, times.values)
    results = numer / denom

    results = pd.Series(results, index=times.index)

    return results

def _resets_from_df(df):
    return df[df.was_reset>0].index.tolist()

def extract_loss(lines, adam_beta1=0.9, adam_beta2=0.999, xscale=1, alpha=0.99, noise_alpha=0.99, noise_alpha_G=None, noise_alpha_S=None, infer_resets=False, clip_reset_lr=True, clip_large_grads=True):
    steps, losses, avg_losses, rates = [],[],[],[]
    val_steps, val_losses = [], []
    resets = []
    timestamps = []
    noise_metrics = {"gn_small": [], "gn_big": [], "G_noise": [], "S_noise": []}

    reset_on_next_step = False

    for line in lines:
        seg = line.partition("loss=")[2]
        if len(seg)>0:
            loss_str, _, seg2 = seg.partition(" ")
            avg_str = seg2.partition("avg=")[2].split(" ")[0]
            rate_str = seg2.partition("rate=")[2].split(" ")[0]
            losses.append(float(loss_str))
            avg_losses.append(float(avg_str))
            rates.append(float(rate_str))

            seg3 = line.partition("[")[2]
            stepstr = seg3.split(" ")[0]
            steps.append(int(stepstr))

            if reset_on_next_step:
                resets.append(1)
                reset_on_next_step = False
            elif False: #infer_resets and len(steps)>1 and steps[-1]-steps[-2]>1:
                resets.append(1)
            else:
                resets.append(0)

            seg_timestamp = line.partition(" ")[0]
            try:
             timestamp = pd.Timestamp(seg_timestamp)
             if infer_resets and len(timestamps) > 0:
                 last_timestamp = timestamps[-1]
                 if timestamp - last_timestamp > pd.Timedelta(seconds=60*60):
                     print(f"inferred reset from {timestamp}, {last_timestamp}, {timestamp - last_timestamp}")
                     reset_on_next_step = True
            except:
             timestamp = None
            timestamps.append(timestamp)

        seg_val = line.partition("loss_val=")[2]
        if len(seg_val)>0:
            val_loss_str = seg_val.rstrip("\n")
            val_losses.append(float(val_loss_str))

            seg3 = line.partition("[")[2]
            stepstr = seg3.split(" ")[0]
            val_steps.append(int(stepstr))
        for name in noise_metrics.keys():
            seg_noise = line.partition(f"{name}=")[2]
            if len(seg_noise)>0:
                metric_str, _, _ = seg_noise.partition(" ")
                noise_metrics[name].append(float(metric_str))

        if infer_resets:
            if "Training..." in line:
                reset_on_next_step = True

    df=pd.DataFrame({"losses": losses, "avg_losses": avg_losses, "lr": rates,
                     "was_reset": resets, "timestamp": timestamps}, index=steps)

    for name in noise_metrics.keys():
        if len(noise_metrics[name])>0:
            # assume we measure these every step --> reuse df step index
            df[name] = noise_metrics[name]
            if name == "G_noise":
                df.loc[df[name]<0, name] = None

    if 'gn_big' in df.columns and clip_large_grads:
         filt_clip_gn = (df.gn_big>1)
         df.loc[filt_clip_gn, list(noise_metrics.keys())] = None

    for name in noise_metrics.keys():
        if name in df.columns:
            if name == "G_noise":
                _alpha = noise_alpha_G if noise_alpha_G is not None else noise_alpha
            elif name == "S_noise":
                _alpha = noise_alpha_S if noise_alpha_S is not None else noise_alpha
            else:
                _alpha = noise_alpha
            df[f"avg_{name}"] = df[name].ewm(alpha=(1-_alpha)).mean()

    if "avg_G_noise" in df.columns and "avg_S_noise" in df.columns:
        df["B_simple"] = df["avg_S_noise"] / df["avg_G_noise"]
        df["B_simple_inst"] = df.S_noise / df.G_noise
    if len(val_steps)>0:
        df["val_losses"] = None
        for step, vl in zip(val_steps, val_losses):
            if step not in df.index:
                df.loc[step, :] = None
                if infer_resets:
                    df.loc[step, "was_reset"] = 1
        df.loc[val_steps, "val_losses"] = val_losses
        df = df.sort_index()

    resets_step_indexed = _resets_from_df(df)
    if len(resets_step_indexed[1:])>0:
        print(f"found resets: {resets_step_indexed[1:]}")
    df["adam_bias_corr"] = calc_adam_bias_corrector(
        times=df.index.values,
        resets=resets_step_indexed,
        beta1=adam_beta1,
        beta2=adam_beta2,
    )
    df["adam_bias_corr_ref"] = calc_adam_bias_corrector(
        times=df.index.values,
        resets=[],
        beta1=adam_beta1,
        beta2=adam_beta2,
    )
    df["adam_bias_lr_mult"] = df.adam_bias_corr / df.adam_bias_corr_ref
    df["lr_intended"] = [v for v in df.lr.values]
    df["lr"] = df.lr * df.adam_bias_lr_mult

    if clip_reset_lr:
        filt = (df.lr_intended < df.lr_intended.shift(1)) & (df.lr_intended < df.lr_intended.shift(-1))
        df.loc[filt, "lr"] = None
        df.loc[filt, "lr_intended"] = None

    if alpha is not None:
        df["avg_losses"] = df.losses.ewm(alpha=xscale*(1-alpha)).mean()

    return df


def load_loss(path="loss_logs/autoresponder_v9_experimental_const_lr.txt", xscale=1, alpha=0.99, noise_alpha=0.99, noise_alpha_G=None, noise_alpha_S=None, infer_resets=False, clip_large_grads=True):
    with open(path, "r") as f:
        l=[line for line in f]
    return extract_loss(l, xscale=xscale, alpha=alpha, noise_alpha=noise_alpha, noise_alpha_G=noise_alpha_G, noise_alpha_S=noise_alpha_S, infer_resets=infer_resets, clip_large_grads=clip_large_grads)

def _compute_lr_scale(df, ref_col):
    n_burn_in = len(df)//3
    typical_ref = df[ref_col].iloc[n_burn_in:].median()
    max_ref = df[ref_col].iloc[n_burn_in:].max()
    min_ref = df[ref_col].iloc[n_burn_in:].min()

    print(f"typical_ref {typical_ref:.2f}, min_ref {min_ref:.2f}, max_ref {max_ref:.2f}")
    to_typical = typical_ref - min_ref
    to_top = max_ref - min_ref
    lrmult = to_top/to_typical

    print(f"scaling up {df.lr.max():.4e} by {to_top:.2f}/{to_typical:.2f}={lrmult:.1%}")
    lrscale_max = (lrmult)*df.iloc[n_burn_in:].lr.max()
    print(f"lrscale_max: {lrscale_max:.4e}")

    return lrscale_max

def show_noise_scale(df, delay_show=False, opacity=1, loss_indexed=False, loss_axmin=None, step_axmin=None):
    fig, axes = plt.subplots(2, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="pandas")

        to_plot = df
        if loss_indexed:
            to_plot = df.set_index("avg_losses").sort_index()
            if loss_axmin is not None:
                to_plot = to_plot[to_plot.index >= loss_axmin]
        else:
            if step_axmin is not None:
                to_plot = to_plot[to_plot.index >= step_axmin]

        ms = 1

        # TODO: DRY
        kwargs = dict(marker='.', ax=axes[0], c='k',)
        to_plot.B_simple.plot(lw=1, markersize=0, alpha=opacity/2, **kwargs)
        axes[0].legend()
        to_plot.B_simple.plot(lw=0, markersize=ms, alpha=opacity, label='B_simple ewm', **kwargs)

        # kwargs_big = dict(marker='.', ax=axes[1], c='b',)
        # kwargs_small = dict(marker='.', ax=axes[1], c='r',)
        # to_plot.avg_gn_big.plot(lw=1, markersize=0, alpha=opacity/2, **kwargs_big)
        # to_plot.avg_gn_small.plot(lw=1, markersize=0, alpha=opacity/2, **kwargs_small)
        # axes[1].legend()
        # to_plot.avg_gn_big.plot(lw=0, markersize=ms, alpha=opacity, label='gn_big ewm', **kwargs_big)
        # to_plot.avg_gn_small.plot(lw=0, markersize=ms, alpha=opacity, label='gn_small ewm', **kwargs_small)
        # axes[1].set_yscale('log')

        kwargs_G = dict(marker='.', ax=axes[1], c='g',)
        to_plot.avg_G_noise.plot(lw=1, markersize=0, alpha=opacity/2, **kwargs_G)

        axes[1].legend(loc='upper left')

        axes_1_right = axes[1].twinx()
        kwargs_S = dict(marker='.', ax=axes_1_right, c='m',)

        to_plot.avg_G_noise.plot(lw=0, markersize=ms, alpha=opacity, label='G_noise ewm', **kwargs_G)
        to_plot.avg_S_noise.plot(lw=1, markersize=0, alpha=opacity/2, **kwargs_S)
        axes_1_right.legend(loc='lower left')
        to_plot.avg_S_noise.plot(lw=0, markersize=ms, alpha=opacity, label='S_noise ewm', **kwargs_S)

        axes[1].set_yscale('log')
        axes_1_right.set_yscale('log')


        # kwargs = dict(marker='.', ax=axes[3], c='k',)
        # to_plot.B_simple_inst.plot(lw=1, markersize=0, alpha=opacity/2, **kwargs)
        # axes[3].legend()
        # to_plot.B_simple_inst.plot(lw=0, markersize=ms, alpha=opacity, label='B_simple_inst', **kwargs)
        # axes[3].set_yscale('log')

    if loss_indexed:
        for ax in axes:
            lim = ax.get_xlim()
            ax.set_xlim(lim[1], lim[0])

    if not delay_show:
        plt.show()
    return axes

def noise_scale_load_and_show(path, eplen=None, epcolor='r', alpha=0.99, noise_alpha=0.99, noise_alpha_S=None, noise_alpha_G=None, opacity=1, showlr=False, lrcolor='g', loss_indexed=False, loss_axmin=None, step_axmin=None, infer_resets=False):
    df=load_loss(path, alpha=alpha, noise_alpha=noise_alpha, noise_alpha_G=noise_alpha_G, noise_alpha_S=noise_alpha_S, infer_resets=infer_resets)

    axes = show_noise_scale(df, delay_show=True, opacity=opacity, loss_indexed=loss_indexed, loss_axmin=loss_axmin, step_axmin=step_axmin)

    if eplen is not None and not loss_indexed:
        eplens=list(range(eplen, df.index.max(), eplen))
        for ax in axes:
            [ax.axvline(ep, c=epcolor, ls='--', lw=1) for ep in eplens]

    if showlr:
        for ax, col in zip(axes, ["B_simple", "B_simple_inst"]):
            lrscale_max = _compute_lr_scale(df, col)
            ax_lr = ax.twinx()
            lrmin = df.lr.min()
            # lr_series = df[df.lr>lrmin].lr
            to_plot = df
            # TODO: DRY
            if loss_indexed:
                to_plot = df.set_index("avg_losses").sort_index()
                if loss_axmin is not None:
                    to_plot = to_plot[to_plot.index >= loss_axmin]

            lr_series = to_plot.lr
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", module="pandas")
                lr_series.plot(ax=ax_lr, c=lrcolor, ls='--', alpha=max(0.2, 0.5*opacity))
            ax_lr.set_ylim(0, lrscale_max)

            if loss_indexed:
                lim = ax_lr.get_xlim()
                ax_lr.set_xlim(lim[1], lim[0])

    plt.show()

def load_and_show(path="loss_logs/autoresponder_v9_experimental_const_lr.txt", eplen=1867, sample_period=-1,
                  show_batches=True, delay_show=False, sets_ticks=True, xscale=1, alpha=0.99, ax=None,
                  opacity=1, epcolor='r', showlr=False, lrscale_max=None, lrcolor='g', ax_lr=None,
                  show_val=False, infer_resets=False):
    df=load_loss(path=path, xscale=xscale, alpha=alpha, infer_resets=infer_resets)

    df.index = [int(xscale*ix) for ix in df.index]
    eplen = int(eplen*xscale)
    sample_period = int(sample_period*xscale)

    eplens=list(range(eplen, df.index.max(), eplen))
    samples = []
    if sample_period > 0:
        samples = list(range(sample_period, df.index.max(), sample_period))

    if ax is not None:
        plt.sca(ax)
    if show_batches:
        df.losses.plot(marker='.', lw=0, markersize=2, alpha=0.5*opacity)
        df.avg_losses.plot(marker='.', lw=0, markersize=2, alpha=opacity)
    else:
        df.avg_losses.plot(marker='.', lw=0, markersize=2, alpha=opacity)

    if show_val and "val_losses" in df.columns:
        df.val_losses.dropna().plot(marker='.', lw=1.5, ls='-', markersize=10, alpha=0.67*opacity, c='k')
    [plt.axvline(ep, c=epcolor, ls='--', lw=1) for ep in eplens]
    [plt.axvline(s, c='g', ls='-', lw=2) for s in samples]

    if sets_ticks:
        if max(df.index) < 7*xscale*200:
            plt.gca().xaxis.set_major_locator(MultipleLocator(xscale*50))
        else:
            plt.gca().xaxis.set_major_locator(MultipleLocator(xscale*200))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.grid();

    if ax is None:
        ax = plt.gca()

    if showlr:
        if lrscale_max is None:
            lrscale_max = _compute_lr_scale(df, "avg_losses")

        newax = False
        if ax_lr is None:
            newax = True
            ax_lr = ax.twinx()
        lrmin = df.lr.min()
        # lr_series = df[df.lr>lrmin].lr
        lr_series = df.lr
        lr_series.plot(ax=ax_lr, c=lrcolor, ls='--', alpha=max(0.2, 0.5*opacity))
        if newax or ax_lr.get_ylim()[1] < lrscale_max:
            ax_lr.set_ylim(0, lrscale_max)

    if not delay_show:
        plt.show();

    return df, ax, ax_lr

def comparison_load_and_show(
    path_ref="loss_logs/autoresponder_v9_experimental_const_lr.txt",
    path_new="loss_logs/autoresponder_v9_experimental_1558M.txt",
    eplen_ref=1887, eplen=830,
    sample_period_ref=-1, sample_period=-1,
    xscale_ref=1, alpha=None,
    showlr=True,
    infer_resets=False, infer_resets_ref=False,
    ):
    _, ax1, ax_lr = load_and_show(
        path=path_ref, show_batches=False, eplen=eplen_ref, epcolor='orange', sample_period=sample_period_ref,
        xscale=xscale_ref, delay_show=True, sets_ticks=True, alpha=alpha, showlr=showlr, lrcolor='m', opacity=0.5,
        infer_resets=infer_resets_ref,
        );
    load_and_show(path=path_new, show_batches=False, eplen=eplen, epcolor='r', sample_period=sample_period,
                  delay_show=False, sets_ticks=False, alpha=alpha, showlr=showlr, ax_lr=ax_lr, ax=ax1,
                  infer_resets=infer_resets,
                  # ax=ax1.twiny()
                  );

def val_show(df, ax=None, delay_show=False, step_axmin=None, eplen=None, showlr=False, lrscale_max=None, ax_lr=None, lrcolor='g', opacity=0.67):
    to_plot = df.val_losses.dropna()
    if step_axmin is not None:
        to_plot = to_plot[to_plot.index >= step_axmin]

    to_plot.plot(marker='.', lw=1.5, ls='-', markersize=10, alpha=opacity, c='k', ax=ax)
    if ax is None:
        ax = plt.gca()
    ax.axhline(to_plot.min(), lw=1, ls='--', alpha=opacity*0.667, c='b')

    if eplen is not None:
        eplens=[e for e in range(eplen, to_plot.index.max(), eplen) if e > to_plot.index.min()]
        [ax.axvline(ep, c='r', ls='--', lw=1) for ep in eplens]

    if showlr:
        step_axmin_ = 0 if step_axmin is None else step_axmin
        df_lr = df[df.index >= step_axmin_]
        if lrscale_max is None:
            lrscale_max = _compute_lr_scale(df_lr, "avg_losses")

        newax = False
        if ax_lr is None:
            newax = True
            ax_lr = ax.twinx()
        lrmin = df_lr.lr.min()
        # lr_series = df[df.lr>lrmin].lr
        lr_series = df_lr.lr
        lr_series.plot(ax=ax_lr, c=lrcolor, ls='--', alpha=max(0.2, 0.5*opacity))
        if newax or ax_lr.get_ylim()[1] < lrscale_max:
            ax_lr.set_ylim(0, lrscale_max)

    if not delay_show:
        plt.show()

def val_vs_train_show(df, ax=None, delay_show=False, step_axmin=None):
    _valdf = df[df.val_losses.notnull()]

    if step_axmin is not None:
        _valdf = _valdf[_valdf.index >= step_axmin]

    to_plot = _valdf.set_index("avg_losses").val_losses

    to_plot.plot(marker='.', lw=1.5, ls='-', markersize=10, alpha=0.67, c='k', ax=ax)
    if ax is None:
        ax = plt.gca()
    xmin, xmax, ymin, ymax = ax.axis()
    _min = min(xmin, ymin)
    _max = max(xmax, ymax)
    ax.plot([_min, _max], [_min, _max], c='g', ls='--', alpha=0.67)
    ax.axis((_max, _min, _min, _max))
    if not delay_show:
        plt.show()

def val_vs_train_show2(df, ax=None, delay_show=False, step_axmin=None):
    _valdf = df[df.val_losses.notnull()]

    to_plot = _valdf.avg_losses - _valdf.val_losses
    if step_axmin is not None:
        to_plot = to_plot[to_plot.index >= step_axmin]
    to_plot.plot(marker='.', lw=1.5, ls='-', markersize=10, alpha=0.67, c='k', ax=ax)

    if not delay_show:
        plt.show()

def val_vs_train_show3(df, ax=None, delay_show=False, step_axmin=None, eplen=None, lrindexed=False):
    _valdf = df[df.val_losses.notnull()]

    to_plot = deepcopy(_valdf)
    to_plot["val_losses"] = to_plot["val_losses"] - _valdf["val_losses"]
    to_plot["avg_losses"] = to_plot["avg_losses"] - _valdf["val_losses"]
    if step_axmin is not None:
        to_plot = to_plot[to_plot.index >= step_axmin]

    if lrindexed:
        to_plot = to_plot.set_index("lr")

    to_plot.val_losses.plot(marker='.', lw=1.5, ls='-', markersize=10, alpha=0.67, c='k', ax=ax)
    to_plot.avg_losses.plot(marker='.', lw=1.5, ls='-', markersize=10, alpha=0.67, ax=ax)

    filt = to_plot.val_losses < to_plot.avg_losses
    to_fill = to_plot[filt]
    ax.fill_between(to_fill.index,
                    to_fill.val_losses.astype(float).values,
                    to_fill.avg_losses.astype(float).values,
                    color='g', alpha=0.33
                    )

    filt = to_plot.val_losses > to_plot.avg_losses
    to_fill = to_plot[filt]
    ax.fill_between(to_fill.index,
                    to_fill.val_losses.astype(float).values,
                    to_fill.avg_losses.astype(float).values,
                    color='r', alpha=0.33
                    )

    if eplen is not None and (not lrindexed):
        eplens=[e for e in range(eplen, to_plot.index.max(), eplen) if e > to_plot.index.min()]
        if ax is None:
            ax = plt.gca()
        [ax.axvline(ep, c='r', ls='--', lw=1) for ep in eplens]

    if not delay_show:
        plt.show()

def val_load_and_show(path, alpha=0.99, noise_alpha=0.99, noise_alpha_S=0.99, noise_alpha_G=0.995, infer_resets=False, step_axmin=None, eplen=None, showlr=False, lrindexed=False):
    df=load_loss(path, alpha=alpha, noise_alpha=noise_alpha, noise_alpha_G=noise_alpha_G, noise_alpha_S=noise_alpha_S, infer_resets=infer_resets)

    fig, axes = plt.subplots(2, 1)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", module="pandas")

        val_show(df, ax=axes[0], delay_show=True, step_axmin=step_axmin, eplen=eplen, showlr=showlr)
        val_vs_train_show3(df, ax=axes[1], step_axmin=step_axmin, eplen=eplen, lrindexed=lrindexed)

def compare_val_trajectories(*dfs, eplen=None, delay_show=False,):
    # styles_head = dict(marker='.', lw=1.5, ls='-', markersize=10, alpha=0.67, c='k',)
    # styles_last = dict(marker='^', lw=1.5, ls='-', markersize=10, alpha=0.67, c='g',)
    styles = dict(lw=1.5, ls='-', markersize=8, alpha=0.75, )
    markers = (k for k in MarkerStyle().markers.keys() if k not in {'.', ','})

    lrkeys_to_dfs = {(df.lr.max(), df.lr.mean()): df for df in dfs}

    for lrkey in sorted(lrkeys_to_dfs.keys()):
        df = lrkeys_to_dfs[lrkey]
        label = f"(max, avg) = ({lrkey[0]:.1e}, {lrkey[1]:.1e})"
        (df.val_losses.dropna()-df.val_losses.dropna().iloc[0]).plot(
            label=label, marker=next(markers), **styles,
        )

    plt.legend()
    # for df in dfs[:-1]:
    #     (df.val_losses.dropna()-df.val_losses.dropna().iloc[0]).plot(**styles_head)
    #
    # df = dfs[-1]
    # (df.val_losses.dropna()-df.val_losses.dropna().iloc[0]).plot(**styles_last)

    if eplen is not None:
        eplens=[e for e in range(eplen, int(plt.axis()[1]), eplen)]
        ax = plt.gca()
        [ax.axvline(ep, c='r', ls='--', lw=1) for ep in eplens]

    if not delay_show:
        plt.show()
