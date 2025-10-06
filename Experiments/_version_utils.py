"""This file is from the Python package neurodiffeq.
"""
import warnings
import functools


def deprecated_alias(**aliases):
    """A decorator to deprecate old argument names in favor of new ones.
    See more here https://stackoverflow.com/a/49802489.
    :param aliases: A sequence of keyword argument of the form: old_name="name_name"
    :param aliases: Dict[str,str]
    :return: A decorated function that can receive either `old_name` or `new_name` as input
    :rtype: function
    """

    def deco(f):
        @functools.wraps(f)  # preserves signature and docstring
        def wrapper(*args, **kwargs):
            _rename_kwargs(f.__name__, kwargs, aliases)
            return f(*args, **kwargs)

        return wrapper

    return deco


def _rename_kwargs(func_name, kwargs, aliases):
    for alias, new in aliases.items():
        if alias in kwargs:
            if new in kwargs:
                raise KeyError(f'{func_name} received both `{alias}` (deprecated) and `{new}` (recommended)')
            warnings.warn(f'The argument `{alias}` is deprecated for {func_name}; use `{new}` instead.', FutureWarning)
            kwargs[new] = kwargs.pop(alias)
