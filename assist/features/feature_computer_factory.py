'''@file feature_computer_factory.py
contains the FeatureComputer factory'''


def factory(feature):
    '''
    create a FeatureComputer

    Args:
        feature: the feature computer type
    '''

    if feature == 'fbank':
        from .fbank import Fbank
        return Fbank
    elif feature == 'mfcc':
        from .mfcc import Mfcc
        return Mfcc
    elif feature == 'mfcc_pitch':
        from .mfcc_pitch import Mfcc_pitch
        return Mfcc_pitch
    else:
        raise Exception('Undefined feature type: %s' % feature)
