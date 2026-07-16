LENGTH_SCALE_NAMES = ['small', 'mid1', 'mid2', 'large']

DEFAULT_LENGTH_SCALE_BOUNDARIES = {
    'small': (100e3, 1e6),
    'mid1': (1e6, 2.5e6),
    'mid2': (2.5e6, 10e6),
    'large': (10e6, 40e6)
}

DEFAULT_SEGMENT_SIZE_DICT = {'small': 4000, 'mid1': 20000, 'mid2': 40000, 'large': 200000}

LS_I_DICT = {('small', 'gain'): 0,
              ('small', 'loss'): 1,
              ('mid1', 'gain'): 2,
              ('mid1', 'loss'): 3,
              ('mid2', 'gain'): 4,
              ('mid2', 'loss'): 5,
              ('large', 'gain'): 6,
              ('large', 'loss'): 7,
              ('combined', 'gain'): 0,
              ('combined', 'loss'): 1
              }

LS_I_DICT_REV = {
     0: ('small', 'gain'),
     1: ('small', 'loss'),
     2: ('mid1', 'gain'),
     3: ('mid1', 'loss'),
     4: ('mid2', 'gain'),
     5: ('mid2', 'loss'),
     6: ('large', 'gain'),
     7: ('large', 'loss'),
    }
