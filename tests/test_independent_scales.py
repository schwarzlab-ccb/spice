"""Scientific regression tests for gain/loss-paired independent detection."""
from copy import deepcopy
from unittest.mock import Mock
import numpy as np
import pandas as pd
import pytest
from spice import main_loci_functions as main
from spice.scale_modes import SCALES, activate_scale, active_tracks, check_cache_mode
from spice.independent_detection import refit_independent_scales
from spice.length_scales import DEFAULT_SEGMENT_SIZE_DICT as SEG
from spice.random_state import seed_task
from spice.tsg_og import detection, loci, permutation, simulation
from spice.utils import open_pickle, save_pickle

@pytest.fixture
def model(monkeypatch):
    """Analytic convolution; native fitting/filtering decisions remain under test."""
    chrom = 'chr21'
    size = int(detection.CHROM_LENS.loc[chrom])
    data = {}
    for i, (scale, direction) in enumerate((s,d) for s in SCALES for d in ('gain','loss')):
        grid = np.arange(size // SEG[scale]) * SEG[scale]
        center = 22e6 + (i//2)*5e6
        signal = 10 + (8 if direction == 'gain' else -3)*np.exp(-((grid-center)/1e6)**2)
        data[(scale,direction)] = dict(chrom=chrom, signals=signal, length_scale=scale,
            type=direction, length_scale_i=i, cur_widths=np.array([2e6]),
            loci_width=max(4,int(2e6/SEG[scale])), kernel=np.ones(3),
            non_centromere_index=np.arange(len(signal)), cur_loss_norm=10.,
            height_multiplier=np.ones(len(signal)), centromere_values={},
            signal_bounds=(signal-.1,signal+.1), signal_upsampling=SEG[scale]/SEG['small'])
    def convolve(cur_chrom, selection_points, cur_widths, cur_length_scale, cur_signal,
                 segment_size=None, **kwargs):
        grid = np.arange(len(cur_signal))*(segment_size or SEG[cur_length_scale])
        result = np.full(len(grid),10.)
        for locus in selection_points:
            result += locus.fitness*np.exp(-((grid-locus.pos)/1e6)**2)
        return result
    for module in (simulation,detection,loci):
        monkeypatch.setattr(module,'convolution_simulation',convolve)
    return chrom,data

def points(scale, positions=(30e6,), fitness=2.):
    i=SCALES.index(scale)*2
    return [[simulation.SelectionPoints(loci=[(p,fitness if j==i else -fitness/2 if j==i+1 else 0.)])
             for p in positions] for j in range(8)]

@pytest.mark.parametrize('scale',SCALES)
def test_optimizer_and_loss_ignore_other_scales(model,scale):
    chrom,data=model
    selected=activate_scale(data,scale)
    perturbed=deepcopy(selected)
    for value in perturbed.values():
        if not value['fit_active']:
            value['signals'][:]=np.nan
            value['signal_bounds']=(value['signals'],value['signals'])
            value['cur_loss_norm']=np.nan
    initial=points(scale)
    def fit(current):
        seed_task(731)
        fitted,loss,_=detection._optimize_selection_points(
            80,list(zip(*initial)),current,chrom,up_down_order=[True],
            allow_pos_change=True,max_pos_change=2e6)
        return np.array([[(x[0].pos,x[0].fitness) for x in row] for row in fitted]),loss
    fitted,loss=fit(selected)
    other,other_loss=fit(perturbed)
    np.testing.assert_array_equal(fitted,other)
    assert loss==other_loss and np.isfinite(loss)
    inactive=np.setdiff1d(np.arange(8),active_tracks(selected))
    assert not fitted[:,inactive,1].any()
    baseline=simulation.convolution_simulation_per_ls(chrom,selected,initial)
    base_loss=detection.calc_mse_loss(selected,baseline)
    for i in active_tracks(selected):
        changed=deepcopy(selected)
        list(changed.values())[i]['signals']+=100
        assert detection.calc_mse_loss(changed,baseline)>base_loss

def frame(scale,fitness):
    result=pd.DataFrame(dict(chrom=['chr1']*len(fitness),pos=np.arange(len(fitness))+20e6,
                             type='OG',length_scale=scale,detection_scale_mode='independent'))
    for s in SCALES:
        result[f'fitness_{s}_gain']=fitness if s==scale else 0.
        result[f'fitness_{s}_loss']=0.
    return result

def test_scoring_uses_own_scale_and_one_bh_family():
    observed=pd.concat([frame('small',[5.,9.]),frame('large',[5.])],ignore_index=True)
    null=permutation.null_from_loci([frame('small',[1.,2.,3.]),frame('large',[10.,20.,30.])])
    np.testing.assert_array_equal(permutation.fitness_statistic(observed),[5,9,5])
    scored=loci.assign_p_values(observed,null,strategy='pooled')
    np.testing.assert_allclose(scored.p_value_raw,[.25,.25,1.])
    np.testing.assert_allclose(scored.p_value,[.375,.375,1.])
    assert scored.loc[:1,'p_value_large'].isna().all()
    assert np.isnan(scored.loc[2,'p_value_small'])
    changed=null.copy()
    changed.loc[changed.length_scale=='large','stat_large']=0
    np.testing.assert_array_equal(permutation.permutation_p(observed,changed,'pooled')[:2],[.25,.25])

@pytest.mark.parametrize('strategy',['pooled','perchrom','zpool','zpool_chrom'])
def test_absent_scale_null_is_conservative(strategy):
    observed=frame('mid2',[5.])
    null=permutation.null_from_loci([frame('small',[1.,2.,3.])])
    assert permutation.permutation_p(observed,null,strategy)[0]==1

def test_reject_joint_null_and_mixed_pool():
    independent=frame('small',[2.])
    joint=independent.drop(columns=['length_scale','detection_scale_mode'])
    with pytest.raises(ValueError,match='detection_scale_mode'):
        permutation.permutation_p(independent,permutation.null_from_loci([joint]))
    with pytest.raises(ValueError,match='detection_scale_mode'):
        permutation.null_from_loci([independent,joint])

def test_refit_only_changed_scale(model):
    chrom,data=model
    combined=[a+b for a,b in zip(points('small',(22e6,)),points('large',(37e6,)))]
    original=deepcopy(combined)
    optimizer=Mock(return_value=(points('small',(22e6,),fitness=5),None))
    result=refit_independent_scales(optimizer,chrom,combined,data,['small','large'],{'small','mid1'},71)
    optimizer.assert_called_once()
    args=optimizer.call_args.kwargs
    np.testing.assert_array_equal(active_tracks(args['data_per_length_scale']),[0,1])
    assert args['N_iterations_optimization']==71
    assert result[0][0][0].fitness==5
    assert result[6][1][0].fitness==original[6][1][0].fitness
    optimizer.reset_mock()
    refit_independent_scales(optimizer,chrom,result,data,['small','large'],set(),71)
    optimizer.assert_not_called()

def test_cache_rejects_unmarked_legacy_and_opposite_modes(tmp_path):
    cache=tmp_path/'detection'/'chr8'
    save_pickle([],str(cache/'detection.pickle'))
    with pytest.raises(ValueError,match='fresh output'):
        check_cache_mode(tmp_path,'chr8','independent',write=True)
    assert not (cache/'scale_mode.json').exists()
    check_cache_mode(tmp_path,'chr8','joint',write=True)
    check_cache_mode(tmp_path,'chr1','independent',write=True)
    with pytest.raises(ValueError,match='fresh output'):
        check_cache_mode(tmp_path,'chr1','joint')

@pytest.mark.parametrize('steps',['fast','full'])
@pytest.mark.parametrize('empty',[False,True])
def test_native_cascade_separate_scale_caches_and_resume(model,tmp_path,monkeypatch,steps,empty):
    chrom,data=model
    if empty:
        for value in data.values():
            value['signals'][:] = 10
            value['signal_bounds'] = (value['signals']-1, value['signals']+1)
    check_cache_mode(tmp_path,chrom,'independent',write=True)
    save_pickle(data,str(tmp_path/'data_per_length_scale'/f'{chrom}.pickle'))
    boot=[np.stack([v['signals'],v['signals']]) for v in data.values()]
    save_pickle(boot,str(tmp_path/'signal_bootstrap'/f'{chrom}_N_2.pickle'))
    monkeypatch.setattr(main,'bootstrap_sampling_of_signal',lambda **kw:boot)
    monkeypatch.setattr(main,'collect_data_per_length_scale',lambda *a,**kw:deepcopy(data))
    native_parallel=detection.Parallel
    monkeypatch.setattr(detection,'Parallel',lambda **kw:native_parallel(n_jobs=1))
    monkeypatch.setattr(detection,'calc_total_events_per_loci',
        lambda cur_chrom,final_events_df,cur_selection_points:
            {key:np.full(len(cur_selection_points[0])+1,100.) for key in data})
    args=dict(final_events_df=pd.DataFrame(),cur_chrom=chrom,which=steps,name='test',
        loci_results_dir=str(tmp_path),detection_scale_mode='independent',N_loci=2,
        N_bootstrap=2,N_bootstrap_for_widths=2,N_kernel=10,
        detection_N_iterations_base=10,detection_max_N_iterations=20,
        detection_final_N_iterations=40,ranking_N_iterations=5,
        flipping_N_iterations=5,flipping_N_iterations_single=3,
        limiting_N_iterations_optim=3,within_ci_N_iterations=3,
        optimizing_N_iterations_optimization=5,infer_widths_N_iterations=3,
        merge_N_iterations_optim=3,filter_N_iterations_optim=3,
        final_limiting_N_iterations_optim=3,th_locus_prominence=0,th_locus_mean_fitness=0)
    result=main.run_loci_detection_per_chrom(**args)
    labels=open_pickle(str(tmp_path/'detection'/chrom/'final_locus_scales.pickle'))
    assert len(labels)==len(result['final_selection_points'][0])
    assert set(result['scales'])==set(SCALES)
    positions=[]
    for scale in SCALES:
        path=tmp_path/'detection'/chrom/scale
        assert (path/'final_selection_points.pickle').exists()
        detected=open_pickle(str(path/'detection.pickle'))[0]
        positions.append([point[0].pos for point in detected[0]])
        for j,track in enumerate(detected):
            if j//2!=SCALES.index(scale):
                assert all(point[0].fitness==0 for point in track)
    if empty:
        assert not labels and not any(positions)
    else:
        assert len({tuple(p) for p in positions})>1
    resumed=main.run_loci_detection_per_chrom(**args)
    for before,after in zip(result['final_selection_points'],resumed['final_selection_points']):
        assert [(p[0].pos,p[0].fitness) for p in before]==[(p[0].pos,p[0].fitness) for p in after]

def test_prominence_and_fitness_filter_use_own_pair(model,monkeypatch):
    chrom,data=model
    selected=activate_scale(data,'large')
    initial=points('large',(37e6,),fitness=8.)
    width=[[36.95e6,37e6,37.05e6]]
    monkeypatch.setattr(detection,'calc_total_events_per_loci',
        lambda *a,**kw:{key:np.array([100.,0.]) for key in data})
    keep,_,_=detection._identify_loci_to_filter(
        chrom,selected,initial,width,pd.DataFrame(),th_locus_prominence=0,
        th_locus_mean_fitness=4,th_added_events=0)
    assert keep.tolist()==[True]  # directional fitness 8, not 8/4 across all scales
    for value in selected.values():
        if not value['fit_active']:
            value['signals'][:] = np.nan
    keep_again,_,_=detection._identify_loci_to_filter(
        chrom,selected,initial,width,pd.DataFrame(),th_locus_prominence=0,
        th_locus_mean_fitness=4,th_added_events=0)
    np.testing.assert_array_equal(keep,keep_again)


def test_no_signal_returns_no_dummy_locus(model):
    chrom,data=model
    selected=activate_scale(data,'mid2')
    for value in selected.values():
        value['signals'][:]=10
        value['signal_bounds']=(np.full(len(value['signals']),9.),np.full(len(value['signals']),11.))
    result,_,_=detection.detect_tsgs_ogs_for_all_length_scales(
        chrom,data_per_length_scale=selected,length_scales_for_residuals=[4,5],N_loci=2,
        N_iterations_base=1,max_N_iterations=2,final_N_iterations=2)
    assert all(not track for track in result)

@pytest.mark.parametrize('cutoff,expected_calls,expected_count',[(.05,1,2),(1.1,0,3),(0,0,0)])
def test_combination_preserves_scale_ranks_and_refits_only_removed_scale(
        model,tmp_path,monkeypatch,cutoff,expected_calls,expected_count):
    chrom,data=model
    check_cache_mode(tmp_path,chrom,'independent',write=True)
    groups=[points('small',(30e6,),fitness=2.),points('small',(31e6,),fitness=.1),
            points('large',(25e6,),fitness=2.)]
    combined=[sum((group[i] for group in groups),[]) for i in range(8)]
    base=tmp_path/'detection'/chrom
    save_pickle(combined,str(base/'final_selection_points.pickle'))
    save_pickle([[p-1000,p,p+1000] for p in [30e6,31e6,25e6]],str(base/'final_loci_widths.pickle'))
    save_pickle(['small','small','large'],str(base/'final_locus_scales.pickle'))
    save_pickle(data,str(tmp_path/'data_per_length_scale'/f'{chrom}.pickle'))
    monkeypatch.setattr(main,'calculate_events_per_loci_df',lambda df,**kw:df)
    def refit(**kw):
        fitted=deepcopy(kw['final_selection_points'])
        for i in active_tracks(kw['data_per_length_scale']):
            for j, point in enumerate(fitted[i]):
                fitted[i][j] = simulation.SelectionPoints(loci=[(point[0].pos, point[0].fitness * 2)])
        return fitted,None
    optimizer=Mock(side_effect=refit)
    monkeypatch.setattr(main,'final_optimization_step',optimizer)
    null=permutation.null_from_loci([frame('small',[1.]*200),frame('large',[1.]*200)])
    final,filtered,widths,raw=main.combine_loci(str(tmp_path),calculate_p_value=True,
        p_value_threshold=cutoff,permutation_null=null,p_values_strategy='pooled',
        detection_scale_mode='independent',final_reoptimization_N_iterations=17)
    assert optimizer.call_count==expected_calls
    assert len(final)==len(filtered[chrom][0])==len(widths[chrom])==expected_count
    assert raw.length_scale.tolist()==['large','small','small']
    if expected_calls:
        assert final.loc[final.length_scale=='small','fitness_small_gain'].iloc[0]==4.
        assert final.loc[final.length_scale=='large','fitness_large_gain'].iloc[0]==2.
        np.testing.assert_array_equal(active_tracks(optimizer.call_args.kwargs['data_per_length_scale']),[0,1])
        assert optimizer.call_args.kwargs['N_iterations_optimization']==17
        assert final.loc[final.length_scale=='small','q_value'].iloc[0]==raw.loc[raw.rank_on_chrom==0,'q_value'].iloc[0]


def test_combination_checks_eventless_legacy_cache(tmp_path):
    save_pickle({},str(tmp_path/'data_per_length_scale'/'chr8.pickle'))
    with pytest.raises(ValueError,match='detection_scale_mode'):
        main.combine_loci(str(tmp_path),processed_events=pd.DataFrame({'chrom':['chr1']}),
                          detection_scale_mode='independent')

def test_refit_random_stream_does_not_depend_on_other_scales(model):
    from spice.random_state import np_rng
    chrom,data=model
    def fit(**kw):
        scale=list(kw['data_per_length_scale'])[active_tracks(kw['data_per_length_scale'])[0]][0]
        return points(scale,fitness=float(np_rng().uniform())),None
    def run(removed):
        combined=[a+b for a,b in zip(points('small'),points('large'))]
        result=refit_independent_scales(fit,chrom,combined,data,['small','large'],removed,10)
        return result[6][1][0].fitness
    assert run({'small','large'})==run({'large'})

def test_empty_event_tracks_have_usable_zero_baselines(model,monkeypatch):
    chrom,data=model
    bounds=[(np.zeros(len(v['signals'])),np.zeros(len(v['signals']))) for v in data.values()]
    monkeypatch.setattr(detection,'get_signal_bootstrap_bounds',lambda *a,**kw:bounds)
    events=pd.DataFrame(columns=['chrom','type','pos','width','plateau'])
    prepared=detection.collect_data_per_length_scale(events,chrom,independent_scales=True)
    assert len(prepared)==8
    for value in prepared.values():
        assert not value['signals'].any()
        assert value['cur_loss_norm']==1
        assert not len(value['cur_widths'])
        assert value['signal_upsampling']>0


def test_no_fitness_drift_in_empty_paired_track(model):
    chrom,data=model
    selected=activate_scale(data,'small')
    selected[('small','loss')]['cur_widths']=[]
    initial=points('small')
    initial[1]=[simulation.SelectionPoints(loci=[(30e6,0)])]
    seed_task(13)
    fitted,_,_=detection._optimize_selection_points(80,list(zip(*initial)),selected,chrom,
        up_down_order=[True],allow_pos_change=False)
    assert fitted[0][1][0].fitness==0
