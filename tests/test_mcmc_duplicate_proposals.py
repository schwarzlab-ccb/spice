"""Identical post-WGD events must not produce an empty compensating event."""
import copy

import numpy as np
import pytest

from spice.event_inference import mcmc_for_large_chroms as mcmc


class FixedChoice:
    def __init__(self, breakpoint, side):
        self.breakpoint = breakpoint
        self.side = side
        self.transitions = 0

    def choice(self, options):
        if isinstance(options, range):
            self.transitions += 1
            return 0 if self.transitions == 1 else 5
        selected = (self.breakpoint, self.side)
        assert selected in options
        return selected

    def sample(self, options, count):
        return options[:count]


@pytest.mark.parametrize('event', [(0, 2), (2, 0)])
@pytest.mark.parametrize('side', ['start', 'end'])
def test_identical_post_events_are_rejected(monkeypatch, event, side):
    events = [[], [event, event]]
    index = 0 if side == 'start' else 1
    rng = FixedChoice(event[index], side)
    monkeypatch.setattr(mcmc, 'py_rng', lambda: rng)
    monkeypatch.setattr(mcmc, '_added_pre_loss_passes_loh_filter', lambda *a, **k: True)
    monkeypatch.setattr(mcmc, 'proposal_wgd_simple_swap', lambda *a, **k: copy.deepcopy(events))
    result = mcmc._create_mcmc_proposal_wgd(events, 1, np.array([4, 4]))
    assert all(start != end for epoch in result for start, end in epoch)
    assert rng.transitions == 2
    for before, after in zip(events, result):
        np.testing.assert_array_equal(before, after)


def profile(events, size=2):
    result = np.full(size, 2)
    for weight, epoch in zip((2, 1), events):
        for start, end in epoch:
            result[min(start, end):max(start, end)] += weight if start < end else -weight
    return result


@pytest.mark.parametrize('events,side', [([[], [(0, 1), (0, 2)]], 'start'),
                                        ([[], [(0, 2), (1, 2)]], 'end')])
def test_distinct_post_events_keep_the_same_cn(monkeypatch, events, side):
    original = copy.deepcopy(events)
    index = 0 if side == 'start' else 1
    monkeypatch.setattr(mcmc, 'py_rng', lambda: FixedChoice(events[1][0][index], side))
    proposed = mcmc.proposal_wgd_add_bp(events, profile(events))
    assert proposed is not None
    assert all(start != end for epoch in proposed for start, end in epoch)
    np.testing.assert_array_equal(profile(proposed), profile(events))
    assert events == original


@pytest.mark.parametrize('transition,function,returns_option', [
    (1, 'proposal_wgd_remove_bp', False),
    (2, 'proposal_wgd_extend_shorten_pre_gain', True),
    (3, 'proposal_wgd_extend_shorten_pre_loss', True),
    (4, 'proposal_wgd_switch_loh_loss', True),
])
def test_other_empty_moves_are_rejected(monkeypatch, transition, function, returns_option):
    choices = iter([transition, 5])
    rng = FixedChoice(0, 'start')
    rng.choice = lambda options: next(choices)
    monkeypatch.setattr(mcmc, 'py_rng', lambda: rng)
    valid = [[(0, 2)], [(2, 1)]]
    invalid = [[(0, 2)], [(2, 2), (2, 1)]]
    monkeypatch.setattr(mcmc, function, lambda *a, **k: (invalid, 'F') if returns_option else invalid)
    monkeypatch.setattr(mcmc, 'proposal_wgd_simple_swap', lambda *a, **k: copy.deepcopy(valid))
    assert mcmc._create_mcmc_proposal_wgd(valid, 1, np.array([4, 3])) == valid
