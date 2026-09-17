import os

import pandas as pd
from joblib import Parallel, delayed

from spice.logging import log_debug
from spice.random_state import derive_seed, seed_task


def save_fail_reports(failed_reports, results_dir=None, cur_step=None, logger=None):
    from spice import directories, config
    if results_dir is None:
        results_dir = directories['results_dir']
    fail_path = os.path.join(results_dir, config['name'], 'events', f'failed_reports{("_" + cur_step) if cur_step is not None else ""}.tsv')
    if len(failed_reports) == 0:
        empty_fail_report = pd.DataFrame(columns=['id', 'step', 'error', 'status'])
        if cur_step is None: # top-level
            empty_fail_report.to_csv(fail_path, sep='\t', index=False)
        return empty_fail_report
    df_fail = pd.DataFrame(failed_reports)
    for col in ['id', 'step', 'error', 'status']:
        if col not in df_fail.columns:
            df_fail[col] = None
    df_fail[['id', 'step', 'error', 'status']].to_csv(fail_path, sep='\t', index=False)
    if logger is not None:
        logger.info(f"A total of {len(df_fail)} tasks failed during execution. "
                    f"Saved failure report with to {fail_path}")
    return df_fail


def step_aware_cleanup(results_dir, requested_steps=None):
    """Delete artifacts from the first requested step onward, including preprocessing.

    Steps order: ['preprocessing', 'split', 'all_solutions', 'disambiguate', 'large_chroms', 'combine']
    - Preprocessing artifacts under `directories.data_dir`:
      - `{name}_processed.tsv`
      - `{name}_processed_split.tsv`
    - Other artifacts under `results_dir` per WGD branch.
    """
    import shutil
    from spice import config
    from spice.utils import get_logger
    logger = get_logger('utils')
    name = config.get('name')

    step_order = ['preprocessing', 'split', 'all_solutions', 'disambiguate', 'large_chroms', 'combine']
    if requested_steps and all(s not in step_order for s in (requested_steps or [])):
        log_debug(logger, 'No step-aware cleanup requested (no --keep-old).')
        return

    step_targets = {
        'preprocessing': [
            ('__data__', f'{name}_processed.tsv'),
            ('__data__', f'{name}_processed_split.tsv'),
        ],
        'split': [
            ('events/nowgd', 'chrom_data_full'),
            ('events/nowgd', 'chrom_data_large'),
            ('events/wgd', 'chrom_data_full'),
            ('events/wgd', 'chrom_data_large'),
            ('', 'WGD_status_major_cn.png'),
            ('', 'WGD_status_ploidy_loh.png'),
        ],
        'all_solutions': [
            ('events/nowgd', 'full_paths_multiple_solutions'),
            ('events/nowgd', 'full_paths_single_solution'),
            ('events/wgd', 'full_paths_multiple_solutions'),
            ('events/wgd', 'full_paths_single_solution'),
        ],
        'disambiguate': [
            ('events/nowgd', 'knn_solved_chroms'),
            ('events/wgd', 'knn_solved_chroms'),
        ],
        'large_chroms': [
            ('events/nowgd', 'mcmc_solved_chroms_large'),
            ('events/wgd', 'mcmc_solved_chroms_large'),
        ],
        'combine': [
            ('', 'final_events.tsv'),
            ('events', 'events_summary.tsv'),
            ('events', 'failed_reports.tsv'),
        ],
    }

    def delete_path(rel_root: str, rel_name: str):
        if rel_root == '__data__':
            data_dir = config['directories']['data_dir']
            target = os.path.join(data_dir, rel_name) if data_dir else None
        else:
            target = os.path.join(results_dir, rel_root, rel_name) if rel_root else os.path.join(results_dir, rel_name)
        if not target:
            return
        if os.path.isdir(target):
            log_debug(logger, f'Removing directory {target}')
            shutil.rmtree(target, ignore_errors=True)
        elif os.path.isfile(target):
            log_debug(logger, f'Removing file {target}')
            try:
                os.remove(target)
            except FileNotFoundError:
                pass

    start_idx = 0 if requested_steps is None else min(step_order.index(s) for s in requested_steps if s in step_order)

    targets = []
    for s in step_order[start_idx:]:
        targets.extend(step_targets.get(s, []))

    if targets:
        logger.info('Step-aware cleanup enabled (no --keep-old).')
        for rel_root, rel_name in targets:
            delete_path(rel_root, rel_name)
    else:
        logger.info('No mapped artifacts to clean for requested steps.')



def _run_batch(cur_ids, cores, desc, func, logger):
    """Run a batch of tasks either serially or in parallel."""
    n_jobs = cores if (cores is not None and cores > 1) else 1
    logger.info(f"{desc}: running on {n_jobs} core(s) for {len(cur_ids)} items")

    def _seeded(cid):
        """Run one task on its own RNG stream, keyed on (step, id).

        Keyed on identity rather than call order, so a sample's result does not depend on --cores,
        on which chunk it was scattered into, or on re-running that sample alone. This is also what
        makes the threading backend below safe: np_rng()/py_rng() are per-thread, so seeding here
        gives each task a private stream instead of letting interleaved threads share one.
        """
        seed_task(derive_seed(desc, cid))
        return func(cid)

    def _format_exception(e: Exception) -> str:
        etype = type(e).__name__
        msg = str(e)
        tb = e.__traceback__
        if tb is None:
            return f"{etype}: {msg}"
        while tb.tb_next is not None:
            tb = tb.tb_next
        frame = tb.tb_frame
        return f"{etype}: {msg} (at {os.path.basename(frame.f_code.co_filename)}:{tb.tb_lineno})"

    if n_jobs == 1:
        def _safe_func(cid):
            try:
                return _seeded(cid)
            except Exception as e:
                logger.error(f"{desc}: failed for id '{cid}'", exc_info=False)
                return {"id": cid, "status": "failed", "error": _format_exception(e), "step": desc}
        results = []
        for i, cid in enumerate(cur_ids):
            logger.info(f'{desc}: {i+1} / {len(cur_ids)} ({100*i/len(cur_ids):.1f}%) - {cid}')
            results.append(_safe_func(cid))
        return results
    else:
        from tqdm import tqdm
        from threading import Lock
        lock = Lock()
        pbar = tqdm(total=len(cur_ids), desc=desc, position=0)
        progress = {'count': 0}
        log_every = max(10, len(cur_ids) // 10)
        def _safe_func_parallel(cid):
            try:
                result = _seeded(cid)
            except Exception as e:
                logger.error(f"{desc}: failed for id '{cid}'", exc_info=False)
                result = {"id": cid, "status": "failed", "error": _format_exception(e), "step": desc}
            with lock:
                progress['count'] += 1
                pbar.update(1)
                if progress['count'] % log_every == 0 or progress['count'] == len(cur_ids):
                    logger.info(f'{desc}: {progress["count"]} / {len(cur_ids)} ({100*progress["count"]/len(cur_ids):.1f}%)')
            return result
        results = Parallel(n_jobs=n_jobs, backend='threading')(delayed(_safe_func_parallel)(cid) for cid in cur_ids)
        pbar.close()
        return results
