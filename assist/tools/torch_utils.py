import re
from .logger import logger


def freeze_modules(model, modules):
    """Freeze model parameters according to modules list.

    Args:
        model (torch.nn.Module): main model to update
        modules (list): specified module list for freezing
    Return:
        model (torch.nn.Module): updated model
    """

    def check(module_name):
        if len(modules) == 1 and modules[0] == "true":
            return True
        return any(re.match(m, module_name) for m in modules)

    for mod, param in model.named_parameters():
        if check(mod):
            logger.info(f"Freezing {mod}, it will not be updated.")
            param.requires_grad = False

    return model
