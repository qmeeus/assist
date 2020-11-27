

def coder_factory(name):
    '''create a Coder object

    args:
        name: the name of the coder

    returns:
        a Coder class
    '''

    if name == 'typeshare_coder':
        from .typeshare_coder import TypeShareCoder
        return TypeShareCoder
    elif name == 'typesplit_coder':
        from .typesplit_coder import TypeSplitCoder
        return TypeSplitCoder
    else:
        raise ValueError(f'Unknown coder:  {name}')
