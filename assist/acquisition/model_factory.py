'''@file model_factory.py
contains the model factory'''

def factory(name):
    '''model factory method

    args:
        name: type of model as a string

    Returns:
        a model class'''

    if name == 'rccn':
        import tfmodel.rccn
        return tfmodel.rccn.RCCN
    elif name == 'rccn_spk':
        import tfmodel.rccn_spk
        return tfmodel.rccn_spk.RCCN_SPK
    elif name == 'pccn':
        import tfmodel.pccn
        return tfmodel.pccn.PCCN
    elif name == 'encoder_decoder':
        import tfmodel.encoder_decoder
        return tfmodel.encoder_decoder.EncoderDecoder
    elif name == 'nmf':
        import nmf
        return nmf.NMF
    elif name == "svm":
        from assist.acquisition.svm import Classifier
        return Classifier
    else:
        raise Exception('unknown acquisition type %s' % name)
