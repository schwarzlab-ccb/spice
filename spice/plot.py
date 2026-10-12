from collections import Counter
import warnings

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress, t

from spice import config, data_loaders
from spice.utils import chrom_id_from_id, get_logger, get_diffs_from_events_df, linkage_order
from spice.segmentation import create_events_in_segmentation_full
from spice.event_inference.data_structures import FullPaths
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT, LS_I_DICT, LS_I_DICT_REV
# from spice.tsg_og.event_rate_per_loci import calc_event_rate_per_loci
# from spice.tsg_og.simulation import convolution_simulation


logger = get_logger('plotting')

COLORS_TAB10 = np.array([np.append(x, 1) for x in plt.get_cmap("tab10").colors])
plt.rcParams.update({
    'legend.fontsize': 'x-large',
    'figure.figsize': (15, 5),
    'axes.labelsize': 'x-large',
    'axes.titlesize': 20,
    'figure.titlesize': 20,
    'xtick.labelsize':'x-large',
    'ytick.labelsize':'x-large'
    })

CHROMS = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
CENTROMERES = data_loaders.load_centromeres()
CHROM_LENS = data_loaders.load_chrom_lengths()
# Genome-linear start offset per chromosome
CHR_CUM_STARTS = CHROM_LENS.cumsum().shift(1, fill_value=0).to_dict()
HG19_CHR_CUM_STARTS = CHR_CUM_STARTS   # backwards-compatible alias (now assembly-aware)
CHROM_LENS = data_loaders.load_chrom_lengths()

ls_colors = {
    'whole_chrom': "#254B64", #colors_(0),
    'whole_arm': "#35A7FF", #colors_(1),
    'telomere_bound': "#FFCB77", #colors_(2),
    'centromere_bound': "#918C85", #colors_(2),
    'internal': "#84E296", #colors_(6),
    'internal:small': "#42714b", #colors_(6),
    'internal:mid1': "#589764", #colors_(6),
    'internal:mid2': "#6ebc7d", #colors_(6),
    'internal:large': "#84e296", #colors_(6),
        }

ls_titles = {
    'internal': 'internal',
    'internal:small': 'internal (small)',
    'internal:mid1': 'internal (mid1)',
    'internal:mid2': 'internal (mid2)',
    'internal:large': 'internal (large)',
    'centromere_bound': 'centromere-bound',
    'telomere_bound': 'telomere-bound',
    'whole_arm': 'whole-arm',
    'whole_chrom': 'whole-chromosome',
    'whole_arm+whole_chrom': 'whole-arm + whole-chromosome',
}
ls_titles_cap = {
    'internal': 'Internal',
    'internal:small': 'Internal (small)',
    'internal:mid1': 'Internal (mid1)',
    'internal:mid2': 'Internal (mid2)',
    'internal:large': 'Internal (large)',
    'centromere_bound': 'Centromere-bound',
    'telomere_bound': 'Telomere-bound',
    'whole_arm': 'Whole-arm',
    'whole_chrom': 'Whole-chromosome',
    'whole_arm+whole_chrom': 'Whole-arm + whole-chromosome',
}

wgd_titles = {
    'nowgd': 'non-WGD',
    'pre': 'pre-WGD',
    'post': 'post-WGD',
    'wgd': 'WGD',
}

event_colors = {
    'gain': "#F46036",
    'loss': "#2892D7",
    'all': "#04A777",
    'diff': "#861657",
}



def plot_inferred_events_per_id(
    cur_id,
    chrom_segments=None,
    events_df=None,
    timed_segments=None,
    timing_posterior=None,
    sv_data=None,
    scores=None,
    sort_by_scores=True,
    show_data=True,
    show_solutions=True,
    show_svs=True,
    show_timing=False,
    show_overlap=False,
    title=None,
    max_cols=6,
    single_row=False,
    show_legend=True,
    figsize=None,
    width_factor=5,
    height_factor=5,
    overlap_threshold=0.95,
    maj_min_phased=True,
    lw=10,
    markersize=8,
    events_unit_size=True,
    sv_matching_threshold=config['params']['sv_matching_threshold'],
    include_A_phased_separate=False,
    include_AB_phased_as_majmin=True,
):
    
    events_df.query("id == @cur_id").copy()

    show_data = show_data and (chrom_segments is not None)
    show_svs = show_svs and show_data and (sv_data is not None)
    show_timing = show_timing and show_data and (timed_segments is not None and timing_posterior is not None)
    show_overlap = show_overlap and show_data and (show_svs or show_timing)
    show_solutions = show_solutions and (events_df is not None)

    if not show_data and not show_solutions:
        raise ValueError('Either data or results have to be plotted')

    if include_A_phased_separate and include_AB_phased_as_majmin:
        raise ValueError(
            "include_A_phased_separate and include_AB_phased_as_majmin cannot both be True")
    if show_timing and show_svs:
        raise NotImplementedError('show_timing and show_svs cannot be both true currently')

    # preprocessing
    cur_chrom = cur_id.split(':')[1]
    cur_chrom_segments = chrom_segments.query('id == @cur_id') if chrom_segments is not None else None
    if len(chrom_segments.index.names) == 1:
        cur_chrom_segments = cur_chrom_segments.reset_index(drop=True).set_index(['sample_id', 'chrom', 'allele'])
    cur_segments_timing = None
    segments_overlap = None   
    if show_timing:
        raise NotImplementedError('timing data is currently not supported for plotting')

    if show_svs:
        chrom_id = chrom_id_from_id(cur_id)
        if 'chrom_id' not in sv_data.columns:
            sv_data['chrom_id'] = sv_data.apply(lambda row: chrom_id_from_id(row['sample_id'] + ':' + row['chrom']), axis=1)
        cur_sv_data = sv_data.query('chrom_id == @chrom_id')
        sv_start_matches = (np.abs(cur_sv_data['start'].values - cur_chrom_segments['start'].values[:, None]) < sv_matching_threshold)
        sv_end_matches = (np.abs(cur_sv_data['end'].values - cur_chrom_segments['end'].values[:, None]) < sv_matching_threshold)
        selected_svs = np.logical_and(sv_start_matches.any(axis=0), sv_end_matches.any(axis=0))
        sv_start_matches = np.where(sv_start_matches[:, selected_svs])
        sv_start_matches = sv_start_matches[0][np.argsort(sv_start_matches[1])]
        sv_end_matches = np.where(sv_end_matches[:, selected_svs])
        sv_end_matches = sv_end_matches[0][np.argsort(sv_end_matches[1])]
        sv_types = cur_sv_data.loc[selected_svs, 'svclass'].values
        sv_overlaps = (sv_start_matches, sv_end_matches, sv_types)
    else:
        sv_overlaps = None

    if events_df is not None and show_solutions:
        events_df = events_df.copy()
        if 'chain_nr' not in events_df.columns:
            events_df['chain_nr'] = 0
        if 'n_post' not in events_df.columns or 'n_pre' not in events_df.columns:
            events_df = events_df.join(events_df.groupby(['id', 'chain_nr'])['wgd'].value_counts().unstack('wgd').fillna(0)
                            .rename({'post': 'n_post', 'pre': 'n_pre'}, axis=1).drop('nowgd', axis=1, errors='ignore').astype(int),
                        on=['id', 'chain_nr'])
            if 'n_post' not in events_df:
                events_df['n_post'] = 0
            if 'n_pre' not in events_df:
                events_df['n_pre'] = 0

        diffs = get_diffs_from_events_df(cur_id, events_df, supported_chains_only=False)
        nr_solution_plots = len(diffs)
    else:
        nr_solution_plots = 0

    # Creating the figure and axes
    if not show_solutions:
        max_cols = 2
    ncols = max(2 + int(show_timing) + int(show_overlap), nr_solution_plots if nr_solution_plots < max_cols else max_cols)
    nrows = int(show_data) + int(np.ceil(nr_solution_plots / ncols))
    if single_row:
        old_ncols = ncols
        ncols = ncols + nr_solution_plots
        nrows = 1
    if figsize is None:
        figsize = (width_factor * ncols, height_factor * nrows)
    fig, axs = plt.subplots(
        figsize=figsize, nrows=nrows, ncols=ncols)
    if single_row:
        axs_solutions = axs[old_ncols:]
        axs_data = axs[:old_ncols]
    else:
        axs_solutions = axs[1:,].flatten() if show_data else axs.flatten()
        axs_data = axs[0] if show_solutions else axs

    centro_start =  CENTROMERES.loc[cur_chrom, 'centro_start'] - 2
    centro_end =  CENTROMERES.loc[cur_chrom, 'centro_end'] + 2
    breakpoints_in_centromere = np.logical_and(
        cur_chrom_segments['start'].values >= centro_start, cur_chrom_segments['start'].values <= centro_end)


    # Plotting
    ## Plot first row (copynumber data and external data)
    if show_data:
        plot_cur_dat(
            cur_chrom_segments, cur_segments_timing, allele="cn", ax=axs_data[0], lw=lw,
            sv_overlaps=sv_overlaps, breakpoints_in_centromere=breakpoints_in_centromere,
            markersize=markersize)
        axs_data[0].set_title("Copy-number profile")
        plot_cur_dat(
            cur_chrom_segments, cur_segments_timing=None, allele="cn", overlap_threshold=overlap_threshold,
            timed_segments_overlap=segments_overlap, sv_overlaps=sv_overlaps, unit_size=True, ax=axs_data[1],
            lw=lw, breakpoints_in_centromere=breakpoints_in_centromere, markersize=markersize)
        axs_data[1].set_title("Copy-number segments")

        if show_timing:
            raise NotImplementedError('timing data is currently not supported for plotting')
        if show_overlap:
            raise NotImplementedError('show_overlap is currently not supported for plotting')

    ## Plot second row (solutions)
    if show_solutions:
        plot_all_diffs(
            cur_id,
            events_df,
            scores=scores.drop(-1, axis=0, errors='ignore').values if scores is not None else None,
            axs=axs_solutions,
            show_legend=show_legend,
            sv_overlaps=sv_overlaps,
            cur_chrom_segments=cur_chrom_segments,
            sort_by_scores=sort_by_scores,
            unit_size=events_unit_size,
            lw=lw
        )   

    # final touches to figure
    if single_row:
        if len(axs) == 3:
            axs[-1].set_title('Inferred events')
    else:
        for i in range(ncols - (2 + int(show_overlap))):
            axs_data[-1 - i].axis("off")
    fig.tight_layout()
    if title is None:
        title = cur_id
    if scores is not None and -1 in scores.index:
        title += f" (all separate: {latex_10_notation(scores.loc[-1])})"
    fig.suptitle(title, fontsize=20, va="bottom", y=1.0)

    cur_sample, cur_chrom, cur_allele = cur_id.split(':')
    fig.suptitle(
        f'{cur_sample} - {cur_chrom.replace("chr", "Chromosome ")} - {cur_allele.replace("cn_", "Allele ")} - ({"WGD" if events_df["wgd"].values[0] != "nowgd" else "nonWGD"})',
        y=1.1)

    return fig


def plot_sample_cn_and_events_single_chrom(
    cur_sample, cur_chrom, final_events_df, chrom_segments, axs=None, has_wgd=False, figsize=(5, 15), small_event_th=5e6):
    if axs is None:
        fig, axs = plt.subplots(ncols=1, nrows=3, figsize=figsize, sharex='col', sharey=False, squeeze=True)
    assert len(axs) == 3
    for i, haplotype in enumerate(['a', 'b']):
        cur_id_hap = cur_sample + ':' + cur_chrom + f':cn_{haplotype}'
        cur_chrom_segments_hap = chrom_segments.query('id == @cur_id_hap')
        if len(cur_chrom_segments_hap) == 0:
            cur_chrom_segments_hap = pd.DataFrame(
                [[0, CHROM_LENS.loc[cur_chrom], cur_chrom, 2 if has_wgd else 1]],
                columns=['start', 'end', 'chrom', 'cn'])
        axs[0].axhline(0, c='black', lw=2.5)
        plot_cur_dat(
            cur_chrom_segments_hap, None, allele="cn", ax=axs[0], breakpoints_in_centromere=[], color=f'C{i}', lw=3)
        plot_cur_events(cur_id_hap, final_events_df, ax=axs[1+i], has_wgd=has_wgd, small_event_th=small_event_th,
                            unit_size_events=False, chrom_segments=chrom_segments)

    axs[0].set_title(f'{cur_chrom}\nCN profile')
    axs[1].set_title('Events haplotype A')
    axs[2].set_title('Events haplotype B')

    return axs



def plot_cur_sv(cur_id, dat, sv_data, events_df, ax=None, figsize=(20, 3.5), allele='cn_a',
                string_as_title=True, show_legend=True, plot_dat=True):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    cur_dat, cur_sv_data, cur_events_df = filter_on_id_svs(cur_id, dat, sv_data, events_df)

    ax = plot_cur_dat(cur_dat, ax=ax, allele=allele)
    for i, (_, sv) in enumerate(cur_sv_data.iterrows()):
        ax.axvspan(sv['start'], sv['end'], color=f'C{i}', alpha=0.5, label=f"SV {i} ({sv['svclass']})")
    if show_legend:
        ax.legend(bbox_to_anchor=(1, 1))

    if string_as_title:
        ax.set_title(''.join(cur_dat[[cur_id.split(':')[-1]]].values[:, 0]))

    return ax


def plot_single_events(cur_chain, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 2))
    
    assert cur_chain['chain_nr'].nunique() == 1
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_white_red", [( 0, event_colors['loss'] ), ( 0.5, "white" ), ( 1, event_colors['gain'] )]
    )
    ax.imshow(np.stack([np.fromiter(diff, dtype=int) * (1 if type=="gain" else -1) for type, diff in cur_chain[['type', 'diff']].values]),
                cmap=cmap, aspect='auto', vmin=-1, vmax=1)

    return ax


def plot_events(cur_events):
    n_chains = cur_events['chain_nr'].nunique()
    fig, axs = plt.subplots(nrows=n_chains, figsize=(20, 2*n_chains))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_white_red", [( 0, event_colors['loss'] ), ( 0.5, "white" ), ( 1, event_colors['gain'] )]
    )
    for chain, ax in zip(cur_events['chain_nr'].unique(), axs):
        cur_chain = cur_events.query('chain_nr == @chain and type != "loh"')
        ax.imshow(np.stack([np.fromiter(diff, dtype=int) * (1 if type=="gain" else -1) for type, diff in cur_chain[['type', 'diff']].values]),
                  cmap=cmap, aspect='auto', vmin=-1, vmax=1)
    plt.tight_layout()

    return ax


def filter_on_id_svs(cur_id, dat, sv_data, events_df):
    cur_sample = cur_id.split(':')[0]
    cur_chrom = cur_id.split(':')[1]
    chrom_id = cur_sample + ':' + cur_chrom

    cur_dat = dat.query('sample_id == @cur_sample and chrom == @cur_chrom')
    cur_sv_data = sv_data.query('sample == @cur_sample and chrom == @cur_chrom')
    cur_events_df = events_df.query('type != "loh" and chrom_id == @chrom_id') if events_df is not None else None

    return cur_dat, cur_sv_data, cur_events_df


def plot_bipartite_connections(
        res, starts=None, ends=None, wgd=False,
        show_starts_ends_per_event=False, title=None,
        profile=None, show_loh=False, show_connection_title=False,
        max_cols=7, size_factor=3, max_adjust=0.5, fontsize=11, lim_adjust=0.5,
        radius_adjust=0.05, lw_circle=2, lw_connection=3):
    '''
    Plot "runes"
    '''
    from spice.event_inference.events_from_graph import raw_events_from_FullPaths  # lazy: needs ortools/fstlib/medicc
    if len(res) == 0:
        return None

    if isinstance(res, FullPaths):
        res = raw_events_from_FullPaths(res, wgd=wgd)

    if starts is None:
        if wgd:
            starts = np.unique([event[0] for path in res for pre_post in path for event in pre_post])
        else:
            starts = np.unique([event[0] for path in res for event in path])
    if ends is None:
        if wgd:
            ends = np.unique([event[1] for path in res for pre_post in path for event in pre_post])
        else:
            ends = np.unique([event[1] for path in res for event in path])

    max_pos = max(np.concatenate([starts, ends]))
    radius = (max_pos+lim_adjust)*radius_adjust

    nrows = int(np.ceil(len(res) / max_cols))
    ncols=len(res) if len(res) < max_cols else max_cols
    fig, axs = plt.subplots(figsize=(size_factor*ncols, size_factor*nrows), nrows=nrows, ncols=ncols)
    if not hasattr(axs, '__len__'):
        axs = np.array([axs])

    for i, (ax, cur_res) in enumerate(zip(axs.flat, res)):

        if wgd:
            cur_res_flat = tuple(cur_res[0] + cur_res[1])

        if show_starts_ends_per_event:
            if wgd:
                starts = np.unique([event[0] for event in cur_res_flat])
                ends = np.unique([event[1] for event in cur_res_flat])
            else:
                starts = np.unique([event[0] for event in cur_res])
                ends = np.unique([event[1] for event in cur_res])

        for start, count in zip(*np.unique(starts, return_counts=True)):
            circle = plt.Circle((start, max_pos), radius=radius, edgecolor='black', facecolor='white',
                                linewidth=lw_circle, zorder=2)
            ax.add_patch(circle)
            ax.text(start, max_pos, start, fontsize=fontsize, va='center', ha='center')
        for end, count in zip(*np.unique(ends, return_counts=True)):
            circle = plt.Circle((end, 0), radius=radius, edgecolor='black', facecolor='white',
                                linewidth=lw_circle, zorder=2)
            ax.add_patch(circle)
            ax.text(end, 0, end, fontsize=fontsize, va='center', ha='center')

        if wgd:
            pre_counter = Counter(cur_res[0])
            post_counter = Counter(cur_res[1])
            total_counter = pre_counter + post_counter
            for which, pre_post_res, colors in zip(['pre', 'post'], cur_res, [['C4', 'C1'], ['C0', 'C3']]):
                for line in pre_post_res:
                    total_count = total_counter[line]
                    pre_count = pre_counter[line]
                    cur_lines = np.linspace(-max_adjust, max_adjust, total_count + 2)[1:-1]
                    cur_lines = cur_lines[:pre_count] if which == 'pre' else cur_lines[pre_count:]
                    for adjust in cur_lines:
                        ax.plot([line[0]+adjust, line[1]+adjust], [max_pos, 0],
                                '-', zorder=1, lw=lw_connection, color=colors[0] if line[0]<line[1] else colors[1])
            if show_connection_title:
                ax.set_title((''.join([str(x[0]) for x in cur_res[0]]) + ' / ' + ''.join([str(x[0]) for x in cur_res[1]]) + '\n' +
                              ''.join([str(x[1]) for x in cur_res[0]]) + ' / ' + ''.join([str(x[1]) for x in cur_res[1]])),
                                fontsize=fontsize, va='bottom', ha='center')

        else:
            cur_res_counter = Counter(cur_res)
            for line, count in cur_res_counter.items():
                for adjust in np.linspace(-max_adjust, max_adjust, count + 2)[1:-1]:
                    ax.plot([line[0]+adjust, line[1]+adjust], [max_pos, 0],
                            '-', zorder=1, lw=lw_connection, color='C0' if line[0]<line[1] else 'C3')

            if show_connection_title:
                ax.set_title('\n'.join([''.join(x) for x in np.array(cur_res, dtype=str).T]),
                                fontsize=fontsize, va='bottom', ha='center')

        if show_loh:
            if profile is None:
                raise ValueError('profile must be provided if show_loh is True')
            for loh in np.where(profile == 0)[0]:
                ax.plot([loh+0.5, loh+0.5], [max_pos, 0], '--', zorder=0.5, lw=lw_connection, color='grey')

        ax.set_xlim(-lim_adjust, max_pos + lim_adjust)
        ax.set_ylim(-lim_adjust, max_pos + lim_adjust)

        ax.text(-lim_adjust*2, max_pos + lim_adjust, i, fontsize=fontsize, weight='bold',
                va='top', ha='left')

    for ax in axs.flat:
        ax.set_axis_off()

    if title is not None:
        fig.suptitle(title, fontsize=fontsize*1.5, weight='bold', va='bottom')
    # plt.tight_layout()

    return fig


def plot_cur_dat(
    cur_dat,
    cur_segments_timing=None,
    ax=None,
    unit_size=False,
    timed_segments_overlap=None,
    sv_overlaps=None,
    allele="cn_a",
    lw=8,
    markersize=8,
    breakpoints_in_centromere=None,
    overlap_threshold=0.95,
    color='black',
    alpha=1,
    ls='-'
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 7))

    if len(cur_dat) == 0:
        return ax

    if unit_size:
        xs = np.arange(len(cur_dat) + 1)
        ax.set_xticks(xs[:-1] + 0.5, labels=xs[:-1])
        ax.set_xlabel("Copy-number segments")
    else:
        xs = np.append(cur_dat.eval("start").values, cur_dat.eval("end").values[-1])
        ax.set_xlabel("Position")
    ax.step(
        xs,
        np.append(
            cur_dat[allele].values.astype(int), cur_dat[allele].values.astype(int)[-1]
        ),
        lw=lw,
        where="post",
        color=color,
        alpha=alpha,
        ls=ls
    )

    if breakpoints_in_centromere is not None:
        for x, in_centromere in zip(xs, breakpoints_in_centromere):
            ax.axvline(x, color="black", linestyle="-" if in_centromere else "--", alpha=0.25)

    if unit_size:
        if timed_segments_overlap is not None:
            timed_segments_overlap = timed_segments_overlap.copy()
            # have to do this so the colors match with the overlap plot
            colors = np.where(timed_segments_overlap.max(axis=1) > overlap_threshold)[0]
            timed_segments_overlap = timed_segments_overlap[
                timed_segments_overlap.max(axis=1) > overlap_threshold
            ]
            overlap_argmax = np.argmax(timed_segments_overlap, axis=1)
            i = 0
            for seg, count in zip(*np.unique(overlap_argmax, return_counts=True)):
                for s in range(count):
                    ax.axvspan(
                        seg + s / count,
                        seg + (s + 1) / count,
                        color=f"C{colors[i]}",
                        alpha=0.5,
                    )
                    i += 1
    else:
        if cur_segments_timing is not None:
            for i, seg in enumerate(cur_segments_timing):
                ax.axvspan(
                    int(seg.split("-")[1]),
                    int(seg.split("-")[2]),
                    color=f"C{i}",
                    alpha=0.5,
                )

        cur_chrom = cur_dat.reset_index()['chrom'].values[0]
        ax.axvspan(CENTROMERES.loc[cur_chrom, 'centro_start'], CENTROMERES.loc[cur_chrom, 'centro_end'], alpha=0.25, color='grey')
    if sv_overlaps is not None:
        ylim = ax.get_ylim()[1] + 0.5
        sv_starts = sv_overlaps[0]
        sv_ends = sv_overlaps[1]
        for i, (start, end) in enumerate(zip(sv_starts, sv_ends)):
            ax.plot([xs[start], xs[end+1]], [ylim+0.05*i, ylim+0.05*i], 'o-', color=f'C{i}', lw=lw,
                    markersize=markersize)

    ax.set_yticks(np.arange(0, ax.get_yticks().max() + 1, 1))
    ax.set_yticklabels(np.arange(0, ax.get_yticks().max() + 1, 1).astype(int))
    ax.set_ylabel("Copy-number")
    ax.set_xlim(0, xs[-1])

    return ax


def plot_single_diff(diffs, ax=None, wgd_line=None, breakpoints_in_centromere=None, figsize=(5, 5),
                     sv_overlap=None, unit_size=True, cur_chrom_segments=None, small_event_th=5e6, lw=3,
                     horizontal_lines=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if unit_size:
        x_pos = np.arange(diffs.shape[1] + 1)
    else:
        assert cur_chrom_segments is not None, "cur_chrom_segments must be provided if unit_size is False"
        x_pos = np.append(cur_chrom_segments.reset_index()["start"].values, cur_chrom_segments.reset_index()["end"].values[-1])
    y_pos = np.arange(len(diffs) + 1)

    if wgd_line is not None:
        diffs = diffs[::-1] # makes sure pre is on top
        index_0 = np.argsort([''.join(d.astype(str)) for d in diffs[:wgd_line]])[::-1]
        index_1 = np.argsort([''.join(d.astype(str)) for d in diffs[wgd_line:]])[::-1]
        diffs = np.concatenate([
            np.array(diffs[:wgd_line])[index_0],
            np.array(diffs[wgd_line:])[index_1]
        ], axis=0)
        if sv_overlap is not None:
            sv_overlap = sv_overlap[::-1]
            sv_overlap = np.concatenate([
                np.array(sv_overlap[:wgd_line])[index_0],
                np.array(sv_overlap[wgd_line:])[index_1]
            ], axis=0)
        pass
    else:
        index = np.argsort([''.join(d.astype(str)) for d in diffs])
        diffs = np.array(diffs)[index]
        sv_overlap = np.array(sv_overlap)[index] if sv_overlap is not None else None
    if breakpoints_in_centromere is None:
        breakpoints_in_centromere = np.zeros(len(x_pos)).astype(bool)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "blue_white_red", [( 0, event_colors['loss'] ), ( 0.5, "white" ), ( 1, event_colors['gain'] )]
    )
    ax.pcolormesh(x_pos, y_pos, diffs, cmap=cmap, vmin=-1, vmax=1, shading="flat")

    if not unit_size and small_event_th is not None:
        for i, d in enumerate(diffs):
            cur_start = x_pos[np.where(d!=0)[0][0]]
            cur_end = x_pos[np.where(d!=0)[0][-1]+1]
            if cur_end - cur_start < small_event_th:
                ax.plot((cur_start + (cur_end-cur_start)/2), i+0.5, 'o', markersize=min(25, 10+15/diffs.shape[1]), 
                        color=cmap(np.inf) if np.max(d) == 1 else cmap(-np.inf),
                        markeredgecolor='black', markeredgewidth=1, zorder=9)
    
    for x, in_centromere in zip(x_pos, breakpoints_in_centromere):
        ax.axvline(x, color="black", linestyle="-" if in_centromere else "--", alpha=0.25)

    if not unit_size:
        ax.axvspan(
            CENTROMERES.loc[cur_chrom_segments['chrom'].values[0], 'centro_start'],
            CENTROMERES.loc[cur_chrom_segments['chrom'].values[0], 'centro_end'], alpha=0.25, color='black')

    if sv_overlap is not None:
        for event_i, sv_i in enumerate(sv_overlap):
            if sv_i is None:
                continue
            ax.plot([0, 0], [event_i+0.5, event_i+0.5], 'o', c=f"C{int(sv_i)}", markersize=10)

    if wgd_line is not None:    
        ax.axhline(wgd_line, color="black", linestyle="-", lw=lw, alpha=1)
        ax.set_yticks(
            np.append(np.arange(diffs.shape[0]) + 0.5, wgd_line), labels=np.append(np.arange(diffs.shape[0])[::-1], 'WGD')
        )
    else:
        ax.set_yticks(
            np.arange(diffs.shape[0]) + 0.5, labels=np.arange(diffs.shape[0])[::-1]
        )
    if horizontal_lines:
        for y in range(diffs.shape[0]+1):
            ax.axhline(y, color="black", linestyle="--", lw=1, alpha=0.3)
    if unit_size:
        ax.set_xticks(np.arange(diffs.shape[1]) + 0.5, labels=np.arange(diffs.shape[1]))
    ax.set_xlabel("Copy-number segments")
    ax.set_ylabel("Events")

    return ax


def plot_all_diffs(
    cur_id,
    events_df,
    axs=None,
    size_factor=5,
    max_cols=5,
    show_legend=False,
    scores=None,
    sv_overlaps=None,
    cur_chrom_segments=None,
    lw=3,
    sort_by_scores=True,
    unit_size=True,
    small_event_th=5e6,
    horizontal_lines=False
):
    from spice.event_inference.mcmc_for_large_chroms import _get_events_from_diff  # lazy: needs ortools/fstlib/medicc
    cur_events = events_df.query("id == @cur_id")
    cur_chrom = cur_id.split(":")[1]
    diffs = get_diffs_from_events_df(cur_id, cur_events)
    had_sv_overlaps = sv_overlaps is not None
    sv_type_by_index = None
    if sv_overlaps is not None:
        if len(sv_overlaps) > 2:
            sv_type_by_index = {i: sv_type for i, sv_type in enumerate(sv_overlaps[2])}
        event_tuples = [np.sort(_get_events_from_diff(cur_diff, False), axis=1) for cur_diff in diffs]
        sv_overlaps = [np.where(np.logical_and(
            cur_event_tuples[:, 0][:, None] == sv_overlaps[0],
            cur_event_tuples[:, 1][:, None] == (sv_overlaps[1]+1),
        )) for cur_event_tuples in event_tuples]
        sv_overlaps = [x if len(x[0])>0 else None for x in sv_overlaps]
        sv_overlaps = [[cur_sv_overlaps[1][cur_sv_overlaps[0]==i] for i in range(len(cur_diff))] if cur_sv_overlaps is not None else None for cur_diff, cur_sv_overlaps in zip(diffs, sv_overlaps)]
        sv_overlaps = [[x[0] if x is not None and len(x) > 0 else None for x in cur_sv_overlaps] if cur_sv_overlaps is not None else None for cur_sv_overlaps in sv_overlaps]
        assert len(sv_overlaps) == len(diffs)

    else:
        sv_overlaps = [None] * len(diffs)
    if cur_events["wgd"].values[0] != "nowgd":
        wgd_lines = cur_events.groupby("chain_nr")["n_post"].first().values
    else:
        wgd_lines = [None] * len(diffs)

    if cur_chrom_segments is not None:
        centro_start =  CENTROMERES.loc[cur_chrom, 'centro_start'] - 2
        centro_end =  CENTROMERES.loc[cur_chrom, 'centro_end'] + 2
        breakpoints_in_centromere = np.logical_and(
            cur_chrom_segments['start'].values >= centro_start, cur_chrom_segments['start'].values <= centro_end)
    else:
        breakpoints_in_centromere = None

    if scores is None:
        scores = [None] * len(diffs)
        sort_by_scores = False
    else:
        assert len(scores) == len(diffs)

    if axs is None:
        nrows = int(np.ceil(len(diffs) / max_cols))
        ncols = len(diffs) if len(diffs) < max_cols else max_cols
        fig, axs = plt.subplots(
            figsize=(size_factor * ncols, size_factor * nrows), nrows=nrows, ncols=ncols
        )
    if not hasattr(axs, "__len__"):
        axs = np.array([axs])

    if sort_by_scores:
        diffs = [
            (int(np.arange(len(diffs))[i]), diffs[i], scores[i], sv_overlaps[i])
            for i in np.argsort(scores)[::-1]
        ]
    else:
        diffs = [(i, diff, score, sv_overlap) for i, (diff, score, sv_overlap) in enumerate(zip(diffs, scores, sv_overlaps))]

    for i, (ax, diff, wgd_line) in enumerate(zip(axs.flatten(), diffs, wgd_lines)):
        plot_single_diff(
            diff[1], ax=ax, wgd_line=wgd_line, breakpoints_in_centromere=breakpoints_in_centromere, small_event_th=small_event_th,
            sv_overlap=diff[3], unit_size=unit_size, cur_chrom_segments=cur_chrom_segments.reset_index(), lw=lw,
            horizontal_lines=horizontal_lines)
        if diff[2] is not None:
            ax.set_title(
                f"solution {diff[0]}\n" + latex_10_notation(diff[2]), fontsize=20
            )
        else:
            ax.set_title(f"solution {diff[0]}", fontsize=20)

    for i in range(len(axs) - len(diffs)):
        axs.flatten()[-1 - i].axis("off")

    if show_legend:
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "blue_white_red", [( 0, event_colors['loss'] ), ( 0.5, "white" ), ( 1, event_colors['gain'] )]
        )
        ax = axs.flatten()[-1]
        ax.plot(
            [],
            [],
            color=cmap(np.inf),
            marker="s",
            label="gain",
            lw=0,
            ms=25,
        )
        ax.plot(
            [],
            [],
            color=cmap(-np.inf),
            marker="s",
            label="loss",
            lw=0,
            ms=25,
        )

        # Add SV legend markers only when SVs are plotted in the figure.
        if had_sv_overlaps:
            present_sv_indices = set()
            for cur_chain in sv_overlaps:
                if cur_chain is None:
                    continue
                for cur_sv_i in cur_chain:
                    if cur_sv_i is not None:
                        present_sv_indices.add(int(cur_sv_i))

            for cur_sv_i in sorted(present_sv_indices)[:5]:
                sv_type = sv_type_by_index.get(cur_sv_i, None) if sv_type_by_index is not None else None
                ax.plot(
                    [],
                    [],
                    color=f"C{cur_sv_i}",
                    marker="o",
                    label=f"SV {cur_sv_i} ({sv_type})" if sv_type is not None else f"SV {cur_sv_i}",
                    lw=0,
                    ms=10,
                )

        ax.legend(bbox_to_anchor=(1, 1.05), fontsize=20)

    return axs


def plot_cur_events(cur_id, events_df, ax=None, has_wgd=False, unit_size_events=False,
                    chrom_segments=None, small_event_th=5e6, lw=3):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    cur_events = events_df.query('id == @cur_id').copy()
    if len(cur_events) == 0:
        ax.set_yticks([])
        ax.set_xticks([])
        return ax
    if chrom_segments is not None:
        cur_chrom_segments = chrom_segments.query('id == @cur_id').copy()
        if len(cur_chrom_segments) == 0:
            cur_chrom_segments = None
    if 'chain_nr' not in cur_events.columns:
        cur_events['chain_nr'] = 0
    if cur_chrom_segments is not None:
        cur_chrom = cur_id.split(':')[1]
        centro_start =  CENTROMERES.loc[cur_chrom, 'centro_start'] - 2
        centro_end =  CENTROMERES.loc[cur_chrom, 'centro_end'] + 2
        breakpoints_in_centromere = np.logical_and(
            cur_chrom_segments['start'].values >= centro_start, cur_chrom_segments['start'].values <= centro_end)
    else:
        breakpoints_in_centromere = None
    if has_wgd and 'n_post' not in cur_events.columns or 'n_pre' not in cur_events.columns:
        cur_events = cur_events.join(cur_events.groupby(['id', 'chain_nr'])['wgd'].value_counts().unstack('wgd').fillna(0)
                        .rename({'post': 'n_post', 'pre': 'n_pre'}, axis=1).drop('nowgd', axis=1, errors='ignore').astype(int),
                    on=['id', 'chain_nr'])
        if 'n_post' not in cur_events:
            cur_events['n_post'] = 0
        if 'n_pre' not in cur_events:
            cur_events['n_pre'] = 0

    wgd_line = cur_events["n_post"].values[0] if has_wgd else None
    diffs = get_diffs_from_events_df(cur_id, cur_events.query('id == @cur_id'))[0]

    plot_single_diff(
        diffs, ax=ax, wgd_line=wgd_line, breakpoints_in_centromere=breakpoints_in_centromere, small_event_th=small_event_th,
        unit_size=unit_size_events, cur_chrom_segments=cur_chrom_segments, lw=lw)

    return ax
   

def plot_gains_losses_over_chrom(
          dat_consistent, events_in_segmentation, chrom, genes_of_interest=None, affected_samples=None,
          chr_lengths=CHROM_LENS, chr_cum_starts=CHR_CUM_STARTS, tsg_og=None,
          centromeres=CENTROMERES, figsize=(25, 12), title=None):

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 4, figure=fig)

    ax_cnp = plt.subplot(gs[0, :])
    ax_gains = [plt.subplot(gs[1, i]) for i in range(4)]
    ax_losses = [plt.subplot(gs[2, i]) for i in range(4)]

    cur_dat_consistent = dat_consistent.query('chrom == @chrom').copy()
    sample_0 = cur_dat_consistent.index.get_level_values('sample_id').unique()[0]

    cur_mean_cn = cur_dat_consistent[['cn_a', 'cn_b']].sum(axis=1).groupby(['chrom', 'start', 'end'], observed=True).mean()

    x = (cur_dat_consistent.loc[sample_0].index.get_level_values('start').values + 
        cur_dat_consistent.loc[sample_0].index.get_level_values('end').values)/2
    ax_cnp.step(x=x,
            y=cur_mean_cn.values,
            label='all', color='green', lw=2)
    if affected_samples is not None:
        cur_mean_cn = cur_dat_consistent.loc[np.intersect1d(cur_dat_consistent.index.get_level_values('sample_id').unique(), affected_samples)][['cn_a', 'cn_b']].sum(axis=1).groupby(['chrom', 'start', 'end'], observed=True).mean()
        ax_cnp.step(x=x,
                y=cur_mean_cn.values,
                label='all', color='green', lw=2, ls='--')
    ax_cnp.set_ylabel('copy-numbers', fontsize=25)
    ax_cnp.set_title(f'{chrom}: Average total copy-numbers (noWGD)', fontsize=25, va='bottom')
    ax_cnp.axhline(2, lw=2, color='black')

    x = (events_in_segmentation.loc[chrom].index.get_level_values('start').values + 
         events_in_segmentation.loc[chrom].index.get_level_values('end').values)/2
    for key, ax_index in zip(['whole', 'all', 'internal', 'telomere_bound_l', 'telomere_bound_r', 'centromere_bound_l', 'centromere_bound_r'],
                             [0, 0, 1, 2, 2, 3, 3]):
            if key == 'all':
                cur_gains = events_in_segmentation.loc[chrom, 'gain'].sum(axis=1).values
                cur_losses = events_in_segmentation.loc[chrom, 'loss'].sum(axis=1).values
            elif key == 'whole':
                cur_gains = events_in_segmentation.loc[chrom, 'gain'][['whole_chrom', 'whole_arm']].sum(axis=1).values
                cur_losses = events_in_segmentation.loc[chrom, 'loss'][['whole_chrom', 'whole_arm']].sum(axis=1).values
            else:
                cur_gains = events_in_segmentation.loc[chrom, ('gain', key)].values
                cur_losses = events_in_segmentation.loc[chrom, ('loss', key)].values 
            ax_gains[ax_index].step(x=x,
                    y=cur_gains,
                    label='gains', lw=2, color=event_colors['gain'], ls='--' if (key == "whole") else '-')
            ax_losses[ax_index].step(x=x,
                    y=-cur_losses,
                    color=event_colors['loss'], label='losses', lw=2, ls='--' if (key == "whole") else '-')
            ax_gains[ax_index].set_title(key.replace('_r', ''), fontsize=25, va='bottom')

    for ax in [ax_cnp]+ax_gains+ax_losses:
        if tsg_og is not None:
            ax2 = ax.twinx()
            ax2.axhline(0, lw=2, color='grey')
            bin_count = chr_lengths.loc[chrom] // 1_000_000
            hist1, bins = np.histogram(tsg_og.query('type=="tsgs" and chr == @chrom')["lin_pos"]-chr_cum_starts[chrom], weights=tsg_og.query('type=="tsgs" and chr == @chrom')["value"], bins=bin_count)
            ax2.bar(bins[:-1], hist1 * -1, width=np.diff(bins), align='edge', color=event_colors['loss'], alpha=0.25)
            hist2, bins = np.histogram(tsg_og.query('type=="ogs" and chr == @chrom')["lin_pos"]-chr_cum_starts[chrom], weights=tsg_og.query('type=="ogs" and chr == @chrom')["value"], bins=bin_count)
            ax2.bar(bins[:-1], hist2, width=np.diff(bins), align='edge', color=event_colors['gain'], alpha=0.25)

        if centromeres is not None:
            ax.axvspan(centromeres.loc[chrom, 'centro_start'], centromeres.loc[chrom, 'centro_end'], alpha=0.25, color='grey')

    if genes_of_interest is not None:
        for (name, pos) in genes_of_interest['pos'].items():
            for ax in [ax_cnp]+ax_gains+ax_losses:
                ax.axvline(pos, lw=2, color='grey', label=name, linestyle='--', zorder=0)
            ax_cnp.text(pos, 2, name, fontsize=25, va='bottom', color='grey')

    for ax in ax_gains[1:]+ax_losses[1:]:
        ax.axhline(0, c='k', lw=2)


    ax_gains[0].set_ylabel(f'Gains', fontsize=25, va='bottom')
    ax_losses[0].set_ylabel(f'Losses', fontsize=25, va='bottom')

    if title is not None:
        fig.suptitle(title, fontsize=25)

    plt.tight_layout()
    return fig


def plot_events_across_chromosome(cur_chrom, cur_segmented_events=None, final_events_df=None, which='both', ax=None, figsize=(20, 10), title=None,
                                  legend=True, legend_outside=True, xlabel=True, ylabel=True,
                                  include_centromere_bound=False, split_left_right=False,
                                  segmentation=100e3):

    if cur_segmented_events is None and final_events_df is None:
        raise ValueError("Either cur_segmented_events or final_events_df must be provided")

    if cur_segmented_events is None:
        cur_segmented_events = create_events_in_segmentation_full(final_events_df, segmentation=segmentation)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    assert which in ['gain', 'gains', 'loss', 'losses', 'both']
    if which == 'both':
        which = 'gains and losses'

    colors = ls_colors
    for lr in ['l', 'r']:
        colors[f'telomere_bound_{lr}'] = colors['telomere_bound']
        colors[f'centromere_bound_{lr}'] = colors['centromere_bound']

    tab20_cmap = plt.get_cmap('tab20')
    if include_centromere_bound:
        if split_left_right:
            columns = ['whole_chrom', 'whole_arm', 'telomere_bound_l', 'telomere_bound_r',
                    'centromere_bound_l', 'centromere_bound_r', 'internal']
        else:
            columns = ['whole_chrom', 'whole_arm', 'telomere_bound',
                    'centromere_bound', 'internal']
    else:
        if split_left_right:
            columns = ['whole_chrom', 'whole_arm', 'telomere_bound_l', 'telomere_bound_r', 'internal']
        else:
            columns = ['whole_chrom', 'whole_arm', 'telomere_bound', 'internal']
    # if not split_left_right:
    #     for gain_loss in ['gain', 'loss']:
    #         cur_segmented_events[gain_loss]['telomere_bound'] = cur_segmented_events[gain_loss][['telomere_bound_l', 'telomere_bound_r']].sum(axis=1)
    #         cur_segmented_events[gain_loss]['centromere_bound'] = cur_segmented_events[gain_loss][['centromere_bound_l', 'centromere_bound_r']].sum(axis=1)
    colors = [colors[x] for x in columns]

    cur_gains = cur_segmented_events['gain'].loc[cur_chrom][columns] if 'gain' in which else None
    cur_losses = cur_segmented_events['loss'].loc[cur_chrom][columns] if 'loss' in which else None
    x_pos = cur_segmented_events['gain'].loc[cur_chrom].index.get_level_values('start').values
    
    if cur_gains is not None:
        ax.stackplot(x_pos, cur_gains.T.values,
                     baseline='zero', labels=columns, colors=colors)
    if cur_losses is not None:
        ax.stackplot(x_pos, cur_losses.T.values * (-1 if cur_gains is not None else 1),
                    baseline='zero', labels=columns if cur_gains is None else [], colors=colors)
    ax.axhline(0, color='black', lw=3)
    ax.set_xlim(0, cur_segmented_events.loc[cur_chrom].index[-1][1])
    ax.axvspan(CENTROMERES.loc[cur_chrom, 'centro_start'], CENTROMERES.loc[cur_chrom, 'centro_end'],
                color='grey', alpha=0.4)
    if xlabel:
        ax.set_xlabel('genome position', fontsize=20)
    if ylabel:
        if which=='gains and losses':
            ylim = ax.get_ylim()
            ax.text(-0.075, (-2*ylim[0]+ylim[1] )/(2*(ylim[1] - ylim[0])), 'gains', transform=ax.transAxes, fontsize=20, ha='center', va='center', rotation='vertical')
            ax.text(-0.075, (-ylim[0] )/(2*(ylim[1] - ylim[0])), 'losses', transform=ax.transAxes, fontsize=20, ha='center', va='center', rotation='vertical')
        else:
            ax.set_ylabel(f'{"gains" if "gain" in which else "loss"}', fontsize=20)
    
    if legend:
        ax.legend(bbox_to_anchor=(1.2, 1) if legend_outside else None)

    if title is None:
        ax.set_title(f'{cur_chrom} - {which}', fontsize=20)
    else:
        ax.set_title(title, fontsize=20)

    return ax


def set_label_sizes(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.label.set_size(20)
    ax.yaxis.label.set_size(20)


def plot_tsg_og_results(
        cur_chrom, 
        data_per_length_scale,
        orientation='h',
        fig=None,
        figsize=None,
        xlim=None,
        cluster_i=None,
        legend=False,
        relative_window_size=None,
        simulated_conv=None,
        simulated_resim=None,
        simulated_conv_alt=None,
        final_selection_points=None,
        restrict_selection_points_to_nonzero=True,
        plot_signal_bounds=True,
        plot_signal_bounds_around_conv=False,
        loci_widths=None,
        plot_genes=False,
        genes_lw=3,
        lw=2,
        genes_markersize=8,
        cur_biscut_loci=None,
        cur_gistic_loci=None,
        cur_cosmic_loci=None,
        cur_davoli_loci=None,
        cur_genes=None,
        show_plateaus=False,
        gene_spacing=None
        ):

    centromeres_observed = data_loaders.load_centromeres(observed=True, extended=False)
    colors = {
        ('gain', 'pos'): event_colors['gain'],
        ('gain', 'neg'): 'orange',
        ('loss', 'pos'): event_colors['loss'],
        ('loss', 'neg'): 'purple',
    }

    if simulated_conv is None:
        simulated_conv = 8*[None]
    if simulated_resim is None:
        simulated_resim = 8*[None]
    if simulated_conv_alt is None:
        simulated_conv_alt = 8*[None]

    if loci_widths is not None:
        assert final_selection_points is not None, "final_selection_points must be provided if loci_widths is provided"
        assert len(loci_widths) == len(final_selection_points[0]), "loci_widths must be the same length as final_selection_points[0]"

    assert all([x['chrom']==cur_chrom for x in data_per_length_scale.values()]), f'Wrong data_per_length_scale for current chrom {cur_chrom}'

    assert orientation in ['h', 'v']
    if figsize is None:
        figsize = (25, 17) if orientation == 'h' else (25, 10)
    
    if xlim is None:
        xlim = [0, CHROM_LENS.loc[cur_chrom]]
    else:
        xlim = [max(0, xlim[0]), min(CHROM_LENS.loc[cur_chrom], xlim[1])]
    if cluster_i is not None:
        assert final_selection_points is not None
        cur_pos = final_selection_points[0][cluster_i][0].pos
        up_down = 'up' if any([final_selection_points[j][cluster_i][0].fitness > 0 for j in range(0, 8, 2)]) else 'down'
    if gene_spacing is None:
        gene_spacing = 0.04 if orientation == 'h' else 0.02

    if fig is None:
        fig, axs = plt.subplots(figsize=figsize, nrows=4 if orientation == 'h' else 1,
                                ncols=1 if orientation == 'h' else 4, sharex=True if orientation == 'h' else False)
    else:
        axs = fig.get_axes()[:4]

    for i, (data, generated_signal_conv, generated_signal_resim, generated_signal_conv_alt) in \
        enumerate(zip(data_per_length_scale.values(), simulated_conv, simulated_resim, simulated_conv_alt)):
        ax = axs[i//2]
        direction = 1 if i % 2 == 0 else -1
        cur_segment_size = DEFAULT_SEGMENT_SIZE_DICT[data['length_scale']]
        if relative_window_size:
            xlim = (max(0, int(cur_pos - relative_window_size*data['loci_width']*cur_segment_size)),
                    int(min(cur_pos + relative_window_size*data['loci_width']*cur_segment_size, CHROM_LENS.loc[cur_chrom])))
            xlim_bin = (int(xlim[0] / cur_segment_size), int(xlim[1] / cur_segment_size))
        else:
            xlim_bin = (np.array(xlim) / cur_segment_size).astype(int)
        cur_x = cur_segment_size * np.arange(xlim_bin[0], xlim_bin[1])

        ax.plot(cur_x, direction * data['signals'][xlim_bin[0]:xlim_bin[1]], label='Observed', c='k', zorder=2,
                lw=lw)
        if plot_signal_bounds:
            if plot_signal_bounds_around_conv:
                ax.fill_between(cur_x,
                                direction * (generated_signal_conv - (data['signals'] - data['signal_bounds'][0]))[xlim_bin[0]:xlim_bin[1]],
                                direction * (generated_signal_conv + (data['signal_bounds'][1] - data['signals']))[xlim_bin[0]:xlim_bin[1]],
                                alpha=0.5, color='C2', zorder=0)
            else:
                ax.fill_between(cur_x, direction * data['signal_bounds'][0][xlim_bin[0]:xlim_bin[1]], direction * data['signal_bounds'][1][xlim_bin[0]:xlim_bin[1]],
                                alpha=0.5, color='grey', zorder=0)
        if show_plateaus:
            ax.plot(cur_x, direction * (data['plateau_signals'][xlim_bin[0]:xlim_bin[1]] + data['signals'][xlim_bin[0]:xlim_bin[1]]),
                    label='Observed plateaus', c='k', zorder=2, ls='--',
                            lw=lw)

        if generated_signal_resim is not None:
            ax.fill_between(cur_x, direction * np.quantile(generated_signal_resim, 0.025, axis=0)[xlim_bin[0]:xlim_bin[1]],
                direction * np.quantile(generated_signal_resim, 0.975, axis=0)[xlim_bin[0]:xlim_bin[1]],
                alpha=0.5, color='C1', zorder=1)
            ax.plot(cur_x, direction * generated_signal_resim.mean(axis=0)[xlim_bin[0]:xlim_bin[1]], label='Inferred\nresim', alpha=1, c='C1', ls='--',
                    zorder=3)
        if generated_signal_conv is not None:
            ax.plot(cur_x, direction * generated_signal_conv[xlim_bin[0]:xlim_bin[1]], label='Inferred\nconv',
                    alpha=1, c='C2', ls='--', lw=lw, zorder=4)
        if generated_signal_conv_alt is not None:
            ax.plot(cur_x, direction * generated_signal_conv_alt[xlim_bin[0]:xlim_bin[1]], label='Inferred\nconv alt',
                    alpha=1, c='C5', ls=':', lw=lw, zorder=5)

        ax.axhline(0, c='k', lw=lw)
        ax.axvspan(*centromeres_observed.loc[cur_chrom, data['length_scale']].values, color='grey', alpha=0.5, label='Centromere', zorder=9)
        ax.set_xlim(*xlim)
        if xlim[0] == 0 and xlim[1] == CHROM_LENS.loc[cur_chrom]:
            ax.set_xticks(np.arange(xlim[0], xlim[1], 1e7))
        if direction == 1:
            if cluster_i is not None: 
                ax.set_title(f'{data["length_scale"]} fit: gain {final_selection_points[i][cluster_i][0].fitness:.2f} / loss {final_selection_points[i+1][cluster_i][0].fitness:.2f}')
            else:
                ax.set_title(data['length_scale'])

        if final_selection_points is not None:
            is_nonzero_everywhere = np.any(np.stack([[locus[0].fitness != 0 for locus in ls_selection_points] for ls_selection_points in final_selection_points]), axis=0)
            for cluster_j in range(len(final_selection_points[0])):
                if not is_nonzero_everywhere[cluster_j]:
                    continue
                pos_j = final_selection_points[0][cluster_j][0].pos
                is_current = cluster_i is not None and cluster_i == cluster_j
                is_up = any([final_selection_points[j][cluster_j][0].fitness > 0 for j in range(0, 8, 2)])
                cur_is_positive = final_selection_points[i][cluster_j][0].fitness > 0

                if restrict_selection_points_to_nonzero:
                    cur_is_nonzero = final_selection_points[i][cluster_j][0].fitness != 0
                else:
                    cur_is_nonzero = ((final_selection_points[i//2 * 2][cluster_j][0].fitness != 0) or
                                      (final_selection_points[i//2 * 2 + 1][cluster_j][0].fitness != 0))

                if restrict_selection_points_to_nonzero and not cur_is_nonzero and not is_current:
                    continue
                # prevent double plotting
                if (not restrict_selection_points_to_nonzero) and i % 2 != 0:
                    continue

                ymin = 0 if restrict_selection_points_to_nonzero and i % 2 == 0 else -1000
                ymax = 0 if restrict_selection_points_to_nonzero and i % 2 != 0 else 1000

                ax.plot(2*[pos_j], [ymin, ymax], c='gray' if (is_current and not cur_is_nonzero) else event_colors['gain'] if is_up else event_colors['loss'],
                        ls=':' if not cur_is_nonzero else '-' if (cur_is_positive or not restrict_selection_points_to_nonzero) else '--', zorder=0, alpha=1 if is_current else 0.5,
                        lw=3 if is_current else lw)
                if loci_widths is not None:
                    cur_std = np.std(loci_widths[cluster_j])
                    # ax.axvspan(pos_j - cur_std, pos_j + cur_std, color=event_colors['gain'] if is_up else event_colors['loss'], alpha=0.125, zorder=9)
                    cur_width_patch = patches.Rectangle((pos_j - 1.5*cur_std, ymin), 3*cur_std, ymax-ymin, 
                         facecolor=event_colors['gain'] if is_up else event_colors['loss'], alpha=0.125, zorder=9, fill=True)
                    ax.add_patch(cur_width_patch)

            # Just for the legend
            ax.axvline(-1e7, c=event_colors['gain'], zorder=0, ls='--', alpha=1, lw=1, label='OG')
            ax.axvline(-1e7, c=event_colors['loss'], zorder=0, ls='--', alpha=1, lw=1, label='TSG')

        ylim = 1.1 * max(
            generated_signal_conv[xlim_bin[0]:xlim_bin[1]].max() if generated_signal_conv is not None else 0,
            data['signals'][xlim_bin[0]:xlim_bin[1]].max())
        if show_plateaus:
            ylim += data['plateau_signals'][xlim_bin[0]:xlim_bin[1]].max()
        if i%2 == 0:
            ax.set_ylim(ax.get_ylim()[0], ylim)
        else:
            ax.set_ylim(-ylim, ax.get_ylim()[1])

        if orientation == 'h' or i == 0:
            ax.set_ylabel('Number of events\n← gains / losses →')
        if legend and i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=20)

    if plot_genes:
        for ax in axs[:4]:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            bottom = ylim[0]
            height = ylim[1] - ylim[0]
            if cur_biscut_loci is not None:
                ax.text(xlim[0], (1-5*gene_spacing) * height + bottom, 'Biscut', fontsize=12, color='black', va='center', ha='left')
                ax.text(xlim[0], 1*gene_spacing * height + bottom, 'Biscut', fontsize=12, color='black', va='center', ha='left')
                for j, row in cur_biscut_loci.query('chrom == @cur_chrom and pos > @xlim[0] and pos < @xlim[1]').reset_index(drop=True).iterrows():
                    cur_y = ((1-5*gene_spacing) if row['type'] == 'gain' else 1*gene_spacing) * height + bottom
                    ax.plot([row['start'], row['end']], [cur_y, cur_y], lw=genes_lw, c=colors[(row['type'], row['negpos'])], zorder=9)
                    ax.plot([row['pos']], [cur_y], 'o', markersize=genes_markersize, c=colors[(row['type'], row['negpos'])], zorder=9)
            if cur_gistic_loci is not None:
                ax.text(xlim[0], (1-4*gene_spacing) * height + bottom, 'Gistic', fontsize=12, color='black', va='center', ha='left')
                ax.text(xlim[0], 2*gene_spacing * height + bottom, 'Gistic', fontsize=12, color='black', va='center', ha='left')
                for j, row in cur_gistic_loci.query('chrom == @cur_chrom and pos > @xlim[0] and pos < @xlim[1]').reset_index(drop=True).iterrows():
                    cur_y = ((1-4*gene_spacing) if row['type'] == 'gain' else 2*gene_spacing) * height + bottom
                    ax.plot([row['start'], row['end']], [cur_y, cur_y], lw=genes_lw, c=colors[(row['type'], 'pos')], zorder=9)
                    ax.plot([row['pos']], [cur_y], 'o', markersize=genes_markersize, c=colors[(row['type'], 'pos')], zorder=9)
            if cur_cosmic_loci is not None:
                ax.text(xlim[0], (1-3*gene_spacing) * height + bottom, 'COSMIC', fontsize=12, color='black', va='center', ha='left')
                for j, row in cur_cosmic_loci.query('chrom == @cur_chrom and pos > @xlim[0] and pos < @xlim[1]').reset_index(drop=True).iterrows():
                    cur_y = (1-3*gene_spacing) * height + bottom
                    ax.plot([row['start'], row['end']], [cur_y, cur_y], lw=genes_lw, c='green' if row['Tier']==1 else 'yellow', alpha=1, zorder=9)
                    ax.plot([row['pos']], [cur_y], 'x', markersize=genes_markersize, c='green' if row['Tier']==1 else 'yellow', alpha=1, zorder=9)
            if cur_davoli_loci is not None:
                ax.text(xlim[0], (1-2*gene_spacing) * height + bottom, 'Davoli', fontsize=12, color='black', va='center', ha='left')
                for j, row in cur_davoli_loci.query('chrom == @cur_chrom and pos > @xlim[0] and pos < @xlim[1]').reset_index(drop=True).iterrows():
                    cur_y = (1-2*gene_spacing) * height + bottom
                    ax.plot([row['start'], row['end']], [cur_y, cur_y], lw=genes_lw, c=event_colors['gain'] if row['which']=='og' else event_colors['loss'] if row['which']=='tsg' else 'green', alpha=1, zorder=9)
                    ax.plot([row['pos']], [cur_y], 'x', markersize=genes_markersize, c=event_colors['gain'] if row['which']=='og' else event_colors['loss'] if row['which']=='tsg' else 'green', alpha=1, zorder=9)
            if cur_genes is not None:
                ax.text(xlim[0], (1-1*gene_spacing) * height + bottom, 'All', fontsize=12, color='black', va='center', ha='left')
                for j, row in cur_genes.query('chrom == @cur_chrom and pos > @xlim[0] and pos < @xlim[1]').reset_index(drop=True).iterrows():
                    cur_y = (1-1*gene_spacing) * height + bottom
                    ax.plot([row['start'], row['end']], [cur_y, cur_y], lw=1, c='grey', alpha=1, zorder=9)
                    ax.plot([row['pos']], [cur_y], 'o', markersize=3, c='grey', zorder=9)
            ax.set_ylim(ylim)

    if orientation == 'h':
        axs[-1].set_xlabel('Position [Mbp]')
    else:
        for ax in axs:
            ax.set_xlabel('Position [Mbp]')
    if cluster_i is not None:
        fig.suptitle(f'{cur_chrom} - locus {cluster_i} ({up_down}) at {cur_pos:.2e}', fontsize=20)
    else:
        fig.suptitle(f'{cur_chrom}' + (f': {len(final_selection_points[0])} inferred loci' if final_selection_points is not None else ''), fontsize=20)
    plt.tight_layout()

    return fig, axs


def plot_inferred_events_per_sample(
    cur_sample,
    chrom_segments,
    final_events_df,
    figsize=(25, 10),
    cn_lw=3,
    unit_size=True,
):

    has_wgd = final_events_df.query('sample == @cur_sample')['has_wgd'].any()
    events_df = final_events_df.query('sample == @cur_sample').copy()
    unique_ids = events_df['id'].unique()
    if 'chain_nr' not in events_df.columns:
        events_df['chain_nr'] = 0
    if 'n_post' not in events_df.columns or 'n_pre' not in events_df.columns:
        events_df = events_df.join(events_df.groupby(['id', 'chain_nr'])['wgd'].value_counts().unstack('wgd').fillna(0)
                        .rename({'post': 'n_post', 'pre': 'n_pre'}, axis=1).drop('nowgd', axis=1, errors='ignore').astype(int),
                    on=['id', 'chain_nr'])
        if 'n_post' not in events_df:
            events_df['n_post'] = 0
        if 'n_pre' not in events_df:
            events_df['n_pre'] = 0

    fig, axss = plt.subplots(figsize=figsize, nrows=3 + int(unit_size), ncols=len(CHROMS), sharex='col',
                            gridspec_kw={'width_ratios': CHROM_LENS.values, 'wspace': 0, 'hspace': 0.1})

    dat_axs_i = {'cn_a': 0, 'cn_b': 2 if unit_size else 0}
    solutions_axs_i = {'cn_a': 1, 'cn_b': 3 if unit_size else 2}

    for cur_chrom, axs in zip(CHROMS, axss.T):
        for cur_allele, allele_color, allele_ls in zip(
            ['cn_a', 'cn_b'], ['C0', 'C1'], ['-', '--']):
            cur_id = f'{cur_sample}:{cur_chrom}:{cur_allele}'
            if (cur_sample, cur_chrom, cur_allele) not in chrom_segments.index:
                axs[dat_axs_i[cur_allele]].plot([0, CHROM_LENS.loc[cur_chrom]], 2*[2 if has_wgd else 1],
                            color=allele_color, alpha=0.5, lw=cn_lw, ls=allele_ls)
                continue
            cur_chrom_segments = chrom_segments.loc[(cur_sample, cur_chrom, cur_allele)]
            plot_cur_dat(cur_chrom_segments, None, allele="cn", ax=axs[dat_axs_i[cur_allele]], lw=cn_lw, unit_size=unit_size,
                            color=allele_color, alpha=0.5, ls=allele_ls)

            if cur_id not in unique_ids:
                # allele_ax.axis('off')
                axs[solutions_axs_i[cur_allele]].axvspan(
                    *CENTROMERES.loc[cur_chrom], color='black', alpha=0.25)
                continue
            plot_all_diffs(
                cur_id,
                events_df,
                axs=axs[solutions_axs_i[cur_allele]],
                show_legend=False,
                cur_chrom_segments=cur_chrom_segments,
                unit_size=unit_size,
                lw=5,
                horizontal_lines=True
            )  

    max_cn = chrom_segments.loc[cur_sample]['cn'].max()
    for ax in axss.flat:
        ax.set_title('')
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
    for ax in axss[0].flat:
        ax.set_ylim(-0.1, max_cn + 0.1)
        axss[0, 0].set_yticks(np.arange(0, max_cn + 1, 1))
    for cur_chrom, axs in zip(CHROMS, axss.T):
        axs[0].set_title(cur_chrom, fontsize=8)
    axss[0, 0].set_ylabel('Copy Number' + (' allele a' if unit_size else ''), fontsize=15)
    if unit_size:
        for ax in axss[2].flat:
            ax.set_ylim(-0.1, max_cn + 0.1)
        axss[2, 0].set_yticks(np.arange(0, max_cn + 1, 1))
        axss[2, 0].set_ylabel('Copy Number allele b', fontsize=15)
    axss[solutions_axs_i['cn_a'], 0].set_ylabel('Events allele a', fontsize=15)
    axss[solutions_axs_i['cn_b'], 0].set_ylabel('Events allele b', fontsize=15)

    fig.suptitle(f'{cur_sample} ({"WGD" if has_wgd else "no WGD"})', fontsize=15)
    fig.text(0.5, 0.02, 'Genomic Segments' if unit_size else 'Genomic Position', ha='center', fontsize=15)
    # plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)

    return fig, axss



def latex_10_notation(n, k=2):

    if n == -1:
        return r'$-1$'
    if n == 0:
        return r'$0$'

    if n<0:
        sign = -1
        n = -n
    else:
        sign = 1

    power = int(np.floor(np.log10(n)))
    number = np.round(n/(10**(power)), k)

    if number/10 >= 1:
        number /= 10
        power += 1

    if k == 0:
        number = int(number)

    return ('-' if sign == -1 else '') + r'${{{}}}\cdot 10^{{{}}}$'.format(number, power)
