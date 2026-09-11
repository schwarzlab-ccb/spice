"""Preserve exhaustive WGD solutions and their order while collecting paths."""
import json
from pathlib import Path

import numpy as np
import pytest

from spice.event_inference.events_from_graph import get_events_for_cur_start_ends_wgd

CASES = json.loads((Path(__file__).parent / 'objects/wgd_path_collection.json').read_text())


@pytest.mark.parametrize('case', CASES)
def test_paths_match_original_enumeration(case):
    result = get_events_for_cur_start_ends_wgd(
        np.array(case['starts']), np.array(case['ends']), case['n_events'], np.array(case['cn']))
    normalized = json.loads(json.dumps(result, default=int))
    assert normalized == case['paths']
