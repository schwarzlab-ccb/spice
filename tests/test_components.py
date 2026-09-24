import numpy as np
import pandas as pd
import pytest
from spice.components import cluster_loci, FITNESS


def frame(pos, q=.01, scale='small', direction='OG'):
    out=pd.DataFrame(dict(chrom='chr1',length_scale=scale,type=direction,pos=pos,
        start=np.asarray(pos)-5,end=np.asarray(pos)+5,p_value=q,q_value=q,
        detection_scale_mode='independent'))
    for col in FITNESS: out[col]=float(col==f'fitness_{scale}_{"gain" if direction=="OG" else "loss"}')
    return out


def test_clustering_uses_track_medians_and_all_raw_peaks():
    a=pd.concat([frame([100,200],.9),frame([1000],scale='large')],ignore_index=True)
    b=pd.concat([frame([110,500]),frame([1200],scale='large')],ignore_index=True)
    medians={('chr1','small','OG'):40,('chr1','large','OG'):1000}
    grouped,members,audit=cluster_loci({10:a,11:b},medians)
    assert sorted(grouped.n_seeds)==[1,1,2,2]
    assert len(members)==6 and audit['input_peaks']==6
    assert not members.duplicated(['component_id','seed']).any()
    assert (grouped.max_centroid_distance <= grouped.cluster_radius).all()
    small=grouped[(grouped.length_scale=='small') & (grouped.n_seeds==2)].iloc[0]
    assert small.mean_q==pytest.approx(.455) and small.width==20
    changed={10:a.assign(q_value=.2,p_value=.2),11:b.assign(q_value=.8,p_value=.8)}
    other,other_members,_=cluster_loci(changed,medians)
    pd.testing.assert_frame_equal(grouped.drop(columns=['mean_q','mean_p']),other.drop(columns=['mean_q','mean_p']))
    pd.testing.assert_frame_equal(members.drop(columns=['p_value','q_value']),other_members.drop(columns=['p_value','q_value']))


def test_directions_scales_and_input_order():
    a=pd.concat([frame([100]),frame([100],direction='TSG'),frame([100],scale='large')],ignore_index=True)
    medians={('chr1',s,d):100 for s in ['small','large'] for d in ['OG','TSG']}
    grouped,_,_=cluster_loci({1:a,2:a},medians)
    assert len(grouped)==3 and (grouped.n_seeds==2).all()
    reversed_groups,_,_=cluster_loci({2:a.iloc[::-1],1:a.iloc[::-1]},medians)
    pd.testing.assert_frame_equal(grouped,reversed_groups)


def test_empty_groups_and_missing_median():
    empty=frame([])
    grouped,members,_=cluster_loci({1:empty,2:empty},{})
    assert grouped.empty and members.empty and 'mean_q' in grouped
    with pytest.raises(ValueError,match='median'):
        cluster_loci({1:frame([100]),2:empty},{})

from test_independent_scales import model


def test_native_component_refit_preserves_geometry_and_scale(model):
    from spice.components import fit_fixed_loci
    chrom,data=model
    table=frame([22e6]).assign(chrom=chrom,mean_q=.01,component_id='component_1')
    fitted,points,audit=fit_fixed_loci(table,data,chrom,seed=10,iterations=80,blocks=2)
    pd.testing.assert_frame_equal(table.drop(columns=FITNESS),fitted.drop(columns=FITNESS))
    assert len(audit)==1 and audit[0]['length_scale']=='small'
    assert audit[0]['final_loss'] <= audit[0]['initial_loss']
    assert all(points[i][0][0].fitness==0 for i in range(2,8))
    assert points[0][0][0].fitness >= 0 and points[1][0][0].fitness <= 0
    assert points[0][0][0].fitness==fitted.fitness_small_gain.iloc[0]
    again,_,_=fit_fixed_loci(table,data,chrom,seed=10,iterations=80,blocks=2)
    pd.testing.assert_frame_equal(fitted,again)


def test_empty_component_fit_is_valid(model):
    from spice.components import fit_fixed_loci
    chrom,data=model
    fitted,points,audit=fit_fixed_loci(frame([]),data,chrom,seed=10,iterations=1,blocks=1)
    assert fitted.empty and all(not track for track in points) and not audit
