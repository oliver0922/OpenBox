"""Load a boxgen YAML configuration (see ``configs/waymo.yaml``) with attribute access."""
import yaml


class AttrDict(dict):
    """dict whose keys are also attributes; nested dicts are wrapped, sequences are left alone."""

    def __getattr__(self, name):
        """name: str key -> its value; raises AttributeError for missing keys."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__


def _wrap(value):
    """Recursively wrap dicts as AttrDict. value: any -> AttrDict for a dict, the value unchanged otherwise."""
    if isinstance(value, dict):
        return AttrDict({k: _wrap(v) for k, v in value.items()})
    return value


def load_config(path):
    """Read ``path`` and return the configuration as nested :class:`AttrDict`.

    ``classes.statistical_box_size`` values are converted to ``(l, w, h)`` tuples, the
    form the box-fitting code has always used.
    """
    with open(path) as f:
        cfg = _wrap(yaml.safe_load(f))
    cfg.classes.statistical_box_size = {
        name: tuple(size) for name, size in cfg.classes.statistical_box_size.items()
    }
    return cfg
