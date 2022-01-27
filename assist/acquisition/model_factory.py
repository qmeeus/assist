'''@file model_factory.py
contains the model factory'''


def model_factory(name):
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
    elif name == "mlp":
        from assist.acquisition.mlp import Classifier
        return Classifier
    elif name in ("lstm", "gru"):
        from assist.acquisition.torch_models.rnn import Classifier
        return Classifier
    elif name in ("att", "att_rnn", "cls_att"):
        from assist.acquisition.torch_models.attention import Classifier
        return Classifier
    elif name == "enc_dec":
        from assist.acquisition.torch_models.encoder_decoder import Classifier
        return Classifier
    else:
        raise Exception('unknown acquisition type %s' % name)
