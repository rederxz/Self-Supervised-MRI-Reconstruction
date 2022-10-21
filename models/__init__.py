from models import du_recurrent_model
from du_recurrent_model import KRNet, IRNet


def create_model(opts):
    if opts.model_type == 'model_recurrent_dual':
        model = du_recurrent_model.RecurrentModel(opts)

    else:
        raise NotImplementedError

    return model
