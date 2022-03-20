from models.pure_transformer import PureT
from models.pure_transformer import PureT_Base
from models.pure_transformer import PureT_Base_22K

__factory = {
    'PureT': PureT,
    'PureT_Base': PureT_Base,
    'PureT_Base_22K': PureT_Base_22K
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown caption model:", name)
    return __factory[name](*args, **kwargs)