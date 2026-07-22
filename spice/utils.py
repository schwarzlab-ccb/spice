
import logging
import os
import pickle
import re
import itertools
import warnings
from functools import wraps
import functools
import threading
import signal

import pandas as pd
import numpy as np

from spice.logging import get_logger, log_debug
# No other SPICE imports here!


def chrom_id_from_id(cur_id):
    return re.sub(r':cn_[ab](:\d+$)?', '', cur_id)


def add_filename_suffix(path, suffix):
    """Insert `suffix` before the file extension.

    Uses os.path.splitext instead of a naive '.tsv' substring replace, so it
    works correctly regardless of the input file's extension (or lack of one).
    E.g. add_filename_suffix('a/b.tsv', '_split') -> 'a/b_split.tsv'
         add_filename_suffix('a/b.csv', '_split') -> 'a/b_split.csv'
         add_filename_suffix('a/b', '_split') -> 'a/b_split.tsv'
    """
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext or '.tsv'}"


def get_sister_allele(cur_id):
    match = re.search(r':cn_([ab])(:\d+)?$', cur_id)
    if match is None:
        raise ValueError(
            f"Cannot determine sister allele for id '{cur_id}': expected an id ending in "
            "':cn_a' or ':cn_b' (optionally followed by ':<int>').")
    sister = 'b' if match.group(1) == 'a' else 'a'
    start, end = match.span(1)
    return cur_id[:start] + sister + cur_id[end:]

class CALC_NEW_SKIP():
    def __repr__(self):
        return 'CALC_NEW skip signal'
    def __eq__(self, other):
        return other is None


class CALC_NEW:
    def __init__(self, filename=None, force_new=False, verbose=False):
        self.filename = filename
        self.force_new = force_new
        self.logger = get_logger('CALC_NEW')
        if not verbose:
            self.logger.setLevel(logging.WARNING)


    def __call__(self, func):
        def wrapper(*args, **kwargs):

            self.result = None
            self.save_data = True

            if 'calc_new_filename' in kwargs:
                filename = kwargs['calc_new_filename']
                del kwargs['calc_new_filename']
            else:
                filename = self.filename
            if 'calc_new_force_new' in kwargs:
                force_new = kwargs['calc_new_force_new']
                del kwargs['calc_new_force_new']
            else:
                force_new = self.force_new
            if 'calc_new_verbose' in kwargs:
                if kwargs['calc_new_verbose']:
                    self.logger.setLevel(logging.DEBUG)
                else:
                    self.logger.setLevel(logging.WARNING)
                del kwargs['calc_new_verbose']

            if filename is None:
                log_debug(self.logger, f"File path is None, ignoring calc_new")
                self.result = func(*args, **kwargs)
                return self.result
            else:
                if os.path.exists(filename):
                    if not force_new:
                        log_debug(self.logger, f"Data exists at '{filename}'")
                        self.save_data = False
                        log_debug(self.logger, f"Loading data")
                        with open(filename, 'rb') as f:
                            self.result = pickle.load(f)
                        return self.result
                    else:
                        log_debug(self.logger, f"Data exists at but forcing recalculation for '{filename}'")
                        self.result = None
                else:
                    log_debug(self.logger, f"Data does not exist at {filename}")

                if self.result is None:
                    self.result = func(*args, **kwargs)
                    if isinstance(self.result, CALC_NEW_SKIP):
                        log_debug(self.logger, f"Function returned skip signal, not saving data")
                        self.save_data = False
                        self.result = None
                    else:
                        log_debug(self.logger, f"Save data to {filename}")
                        if os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
                            os.makedirs(os.path.dirname(filename), exist_ok=True)
                        with open(filename, 'wb') as f:
                            pickle.dump(self.result, f)

                return self.result
        return wrapper


def assert_close(a, b):
    assert np.isclose(a, b), print(f"{a} - {b} = {a - b}")


def open_pickle(filename, n_elements=None, fail_if_nonexisting=True,
                fail_for_other_errors=True, data_type=None):

    if not fail_if_nonexisting and not os.path.exists(filename):
        local_logger = get_logger('utils')
        local_logger.warning('File does not exist, returning None')
        return None
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            if fail_if_nonexisting:
                raise Exception(f"File {filename} was not found:\n{str(e)}")
        elif fail_for_other_errors:
            raise Exception(f"Opening pickle {filename} failed with error code:\n{str(e)}")
        data = None
    if n_elements is not None and not hasattr(data, '__len__') and data is None:
        data = tuple([None] * n_elements)

    if data_type is not None:
        assert isinstance(data, data_type), f"Data type is {type(data)}, expected {data_type}"

    return data


def save_pickle(obj, filename, create_dir_if_not_exists=True):

    if create_dir_if_not_exists and os.path.dirname(filename) and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def linkage_order(data):
    import scipy.cluster.hierarchy as shc
    linkage = shc.linkage(data, method='ward')
    dend = shc.dendrogram(linkage, no_plot=True)

    return dend['leaves']


# TODO: Create to events_from_graph?
def create_full_df_from_diff_df(df, cur_id, cur_chrom_segments, chrom_lengths=None,
                                calc_telomere_bound=True):
    if chrom_lengths is None:
        from spice.data_loaders import load_chrom_lengths
        chrom_lengths = load_chrom_lengths()
    cur_chrom_segments = cur_chrom_segments.query('id == @cur_id')

    df = df.copy()
    df['id'] = cur_id
    df[['sample', 'chrom', 'allele']] = cur_id.split(':')
    df['chrom_length'] = chrom_lengths.loc[df['chrom'].values[0]]
    df['chrom_id'] = cur_id[:cur_id.rfind(':')]
    df['start'] = cur_chrom_segments['start'].values[df['diff'].map(lambda x: x.find('1')).values]
    df['end'] = cur_chrom_segments['end'].values[df['diff'].map(lambda x: x.rfind('1')).values]
    # width cannot simply be end - start because of LOH-shortened events
    df['total_loh_gaps'] = [cur_chrom_segments.eval('end - start').values[
        diff.find('1'):diff.rfind('1')][
            np.fromiter(diff[diff.find('1'):diff.rfind('1')], int) == 0].sum() for diff in df['diff'].values]
    df['width'] = df.eval('(end - start) - total_loh_gaps')
    
    if calc_telomere_bound:
        df[['telomere_bound', 'whole_arm', 'whole_chrom']] = np.stack(calc_telomere_bound_whole_arm_whole_chrom(df), axis=1)
       
    if 'is_gain' in df.columns:
        df['type'] = df['is_gain'].map({True: 'gain', False: 'loss'})
    if 'type' in df.columns:
        df['type_diff'] = df[['type', 'diff']].apply(lambda x: f'{x["type"]}:{x["diff"]}', axis=1)

    return df


def calc_telomere_bound_whole_arm_whole_chrom(data, return_left_and_right=False):
    telomere_bound_l, telomere_bound_r = calc_telomere_bound_left_and_right(data)
    centromere_bound_l, centromere_bound_r = calc_centromere_bound(data)

    # Fix problems with chrX assignment
    if isinstance(data, pd.DataFrame):
        left_bool = data.eval('chrom == "chrX" and start < 28e5').values
        right_bool = data.eval('chrom == "chrX" and end > 1548e5').values
        telomere_bound_l = np.logical_or(telomere_bound_l, left_bool)
        telomere_bound_r = np.logical_or(telomere_bound_r, right_bool)
    # else:
    #     logger.warning("Cannot fix chrX telomere bounds for non-DataFrame data")
    telomere_bound = np.logical_or(telomere_bound_l, telomere_bound_r)
    whole_chrom = np.logical_and(telomere_bound_l, telomere_bound_r)
    whole_arm = np.logical_and(
        (~whole_chrom),
        np.logical_or(
            np.logical_and(telomere_bound_l, centromere_bound_r),
            np.logical_and(telomere_bound_r, centromere_bound_l)
        )
    )
    if return_left_and_right:
        return centromere_bound_l, centromere_bound_r, telomere_bound_l, telomere_bound_r, telomere_bound, whole_arm, whole_chrom
    else:
        return telomere_bound, whole_arm, whole_chrom


def calc_telomere_bound_left_and_right(data):
    if isinstance(data, pd.DataFrame):
        telomere_bound_l = data['diff'].astype(str).str.startswith('1').values
        telomere_bound_r = data['diff'].astype(str).str.endswith('1').values

    elif isinstance(data, tuple):
        assert len(data) == 6, f'Data tuple must have 6 elements. Current has {len(data)} elements.'
        event_sorted, cn_profile, starts, ends, centro_start, centro_end = data
        telomere_bound_l = (event_sorted == 0).any(axis=1)
        telomere_bound_r = (event_sorted == len(cn_profile)).any(axis=1)
    else:
        raise ValueError(f"Data type {type(data)} not supported")

    return telomere_bound_l, telomere_bound_r


def calc_centromere_bound(data):
    if isinstance(data, pd.DataFrame):
        from spice.data_loaders import load_centromeres # import has to be here to prevent circular imports
        centromeres = load_centromeres().astype(int)
        data = (data
                .drop(columns=['centro_start', 'centro_end'], errors='ignore')
                .join(centromeres, on='chrom'))

        centromere_bound_r = data.eval('end >= centro_start-2 and end <= centro_end+2').values
        centromere_bound_l = data.eval('start >= centro_start-2 and start <= centro_end+2').values

    elif isinstance(data, tuple):
        assert len(data) == 6, f'Data tuple must have 6 elements. Current has {len(data)} elements.'
        event_sorted, cn_profile, starts, ends, centro_start, centro_end = data
        # Note: safer to use np.logical_and and np.logical_or instead of & and |
        centromere_bound_r = np.logical_and(ends >= centro_start-2, ends <= centro_end+2)
        centromere_bound_l = np.logical_and(starts >= centro_start-2, starts <= centro_end+2)
    else:
        raise ValueError(f"Data type {type(data)} not supported")

    return centromere_bound_l, centromere_bound_r


@CALC_NEW()
def create_chrom_type_pos_indices(events_df):
    assert (events_df.index == np.arange(len(events_df))).all(), "index has to be 0, 1, 2, 3, ..."
    chrom_type_pos_indices = dict()
    for chrom, type, pos in itertools.product(events_df['chrom'].unique(), events_df['type'].unique(), events_df['pos_detail'].unique()):
        chrom_type_pos_indices[(chrom, type, pos)] = events_df.query('chrom == @chrom and type == @type and pos_detail == @pos').index.values

    assert len(np.unique(np.concatenate([x for x in chrom_type_pos_indices.values()]))) == len(events_df), f"{len(np.unique(np.concatenate([x for x in chrom_type_pos_indices.values()]))), len(events_df)}"
    assert all(np.sort(np.unique(np.concatenate([x for x in chrom_type_pos_indices.values()]))) == np.sort(events_df.index))

    return chrom_type_pos_indices


def create_full_paths_events_df(cur_full_paths, chrom_segments):
    unique_events_df = pd.DataFrame(cur_full_paths.events.values())
    unique_events_df = create_full_df_from_diff_df(unique_events_df, cur_full_paths.id, chrom_segments)
    cur_events_df = pd.concat([unique_events_df.iloc[np.array(list(sol.elements()))] for sol in cur_full_paths.solutions], axis=0)
    cur_events_df['chain_nr'] = np.repeat(np.arange(cur_full_paths.n_solutions), cur_full_paths.n_events)
    cur_events_df['chain'] = cur_events_df['id'] + ':' + cur_events_df['chain_nr'].astype(str)
    cur_events_df['events_per_chrom'] = cur_full_paths.n_events
    cur_events_df['n_paths'] = int(cur_full_paths.n_solutions)
    return cur_events_df


 # TODO: move
def get_diffs_from_events_df(cur_id, events_df, supported_chains_only=False):
    cur_events = events_df.query("id == @cur_id")

    # needs to be separate query in case cur_events do not have supported_chain colum
    if supported_chains_only:
        cur_events = cur_events.query("supported_chain")
    if cur_events.empty:
        print("no events found")
        return None
    if 'chain_nr' not in cur_events.columns:
        cur_events = cur_events.copy()
        cur_events['chain_nr'] = 0

    all_diffs = []
    for chain_nr, events in cur_events.groupby("chain_nr"):
        diffs = np.stack(
            [
                np.fromiter(diff, int) * (1 if type == "gain" else -1)
                for diff, type in events[["diff", "type"]].values
            ]
        )
        # sort pre before post
        diffs = diffs[np.argsort(events["wgd"].values)[::-1]]
        # revert to get pre-WGD LOHs in the right order
        n_pre = (events["wgd"].values == 'pre').sum()
        diffs[n_pre:] = diffs[n_pre:][::-1]
        all_diffs.append(diffs)
    return all_diffs


def suppress_warnings(warning_type=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                if warning_type:
                    warnings.simplefilter("ignore", warning_type)
                else:
                    warnings.simplefilter("ignore")
                return func(*args, **kwargs)
        return wrapper
    return decorator


class FunctionTimeoutError(TimeoutError):
    pass


def timeout(seconds, mode="auto", error=FunctionTimeoutError):
    if seconds is None or seconds == "None":
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator
    if seconds <= 0:
        raise ValueError("seconds must be > 0")

    def _run_with_signal(func, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("signal timeouts only work in the main thread")

        def _handle(signum, frame):
            raise error(f"{func.__name__} timed out after {seconds} seconds")

        old = signal.signal(signal.SIGALRM, _handle)
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return func(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    def _run_in_process(func, *args, **kwargs):
        try:
            from joblib.externals.loky import get_reusable_executor
        except Exception:
            import multiprocessing as mp
            import queue

            q = mp.Queue(1)

            def runner(q_, a, k):
                try:
                    q_.put(("ok", func(*a, **k)))
                except BaseException as e:
                    q_.put(("err", e))

            p = mp.Process(target=runner, args=(q, args, kwargs), daemon=True)
            p.start()
            try:
                status, payload = q.get(timeout=seconds)
            except queue.Empty:
                p.terminate()
                p.join(timeout=1)
                raise error(f"{func.__name__} timed out after {seconds} seconds")

            p.join(timeout=1)
            if status == "ok":
                return payload
            raise payload

        ex = get_reusable_executor(max_workers=1)
        fut = ex.submit(func, *args, **kwargs)
        try:
            return fut.result(timeout=seconds)
        except TimeoutError:
            try:
                fut.cancel()
            finally:
                try:
                    ex.shutdown(wait=False, kill_workers=True)
                except TypeError:
                    ex.shutdown(wait=False)
            raise error(f"{func.__name__} timed out after {seconds} seconds")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            m = mode.lower()
            if m == "process":
                return _run_in_process(func, *args, **kwargs)
            if m == "signal":
                return _run_with_signal(func, *args, **kwargs)

            if threading.current_thread() is threading.main_thread():
                return _run_with_signal(func, *args, **kwargs)
            return _run_in_process(func, *args, **kwargs)

        return wrapper

    return decorator
