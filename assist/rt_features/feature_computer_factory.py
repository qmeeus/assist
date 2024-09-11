'''@file feature_computer_factory.py
contains the FeatureComputer factory'''

from . import mel

def factory(feature):
    '''
    create a FeatureComputer

    Args:
        feature: the feature computer type
    '''

    if feature == 'mel':
        return mel.Mel
    else:
        raise Exception('Undefined feature type: %s' % feature)
