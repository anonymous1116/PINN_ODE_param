"""This file is from the Python package neurodiffeq.
This module contains atomic generator classes and useful tools to construct complex generators out of atomic ones
"""
import torch
import numpy as np
from typing import List


def _chebyshev_first(a, b, n):
    nodes = torch.cos(((torch.arange(n) + 0.5) / n) * np.pi)
    nodes = ((a + b) + (b - a) * nodes) / 2
    nodes.requires_grad_(True)
    return nodes


def _chebyshev_second(a, b, n):
    nodes = torch.cos(torch.arange(n) / float(n - 1) * np.pi)
    nodes = ((a + b) + (b - a) * nodes) / 2
    nodes.requires_grad_(True)
    return nodes


def _compute_log_negative(t_min, t_max, whence):
    if t_min <= 0 or t_max <= 0:
        suggested_t_min = 10 ** t_min
        suggested_t_max = 10 ** t_max
        raise ValueError(
            f"In this version of neurodiffeq, "
            f"the interval [{t_min}, {t_max}] cannot be used for log-sampling in {whence} "
            f"If you meant to sample from the interval [10 ^ {t_min}, 10 ^ {t_max}], "
            f"please pass in {suggested_t_min} and {suggested_t_max}"
        )

    return np.log10(t_min), np.log10(t_max)


class BaseGenerator:
    """Base class for all generators; Children classes must implement a `.get_examples` method and a `.size` field.
    """

    def __init__(self):
        self.size = None

    def get_examples(self) -> List[torch.Tensor]:
        pass  # pragma: no cover

    @staticmethod
    def check_generator(obj):
        if not isinstance(obj, BaseGenerator):
            raise ValueError(f"{obj} is not a generator")

    def __add__(self, other):
        self.check_generator(other)
        return ConcatGenerator(self, other)

    def __mul__(self, other):
        self.check_generator(other)
        return EnsembleGenerator(self, other)

    def __xor__(self, other):
        self.check_generator(other)
        return MeshGenerator(self, other)

    def _internal_vars(self) -> dict:
        return dict(size=self.size)

    @staticmethod
    def _obj_repr(obj) -> str:
        if isinstance(obj, tuple):
            return '(' + ', '.join(BaseGenerator._obj_repr(item) for item in obj) + ')'
        if isinstance(obj, list):
            return '[' + ', '.join(BaseGenerator._obj_repr(item) for item in obj) + ']'
        if isinstance(obj, set):
            return '{' + ', '.join(BaseGenerator._obj_repr(item) for item in obj) + '}'
        if isinstance(obj, dict):
            return '{' + ', '.join(
                BaseGenerator._obj_repr(k) + ': ' + BaseGenerator._obj_repr(obj[k])
                for k in obj
            ) + '}'

        if isinstance(obj, torch.Tensor):
            return f'tensor(shape={tuple(obj.shape)})'
        if isinstance(obj, np.ndarray):
            return f'ndarray(shape={tuple(obj.shape)})'
        return repr(obj)

    def __repr__(self):
        d = self._internal_vars()
        keys = ', '.join(f'{k}={self._obj_repr(d[k])}' for k in d)
        return f'{self.__class__.__name__}({keys})'


class Generator1D(BaseGenerator):
    """An example generator for generating 1-D training points.
    :param size: The number of points to generate each time `get_examples` is called.
    :type size: int
    :param t_min: The lower bound of the 1-D points generated, defaults to 0.0.
    :type t_min: float, optional
    :param t_max: The upper boound of the 1-D points generated, defaults to 1.0.
    :type t_max: float, optional
    :param method:
        The distribution of the 1-D points generated.
        - If set to 'uniform',
          the points will be drew from a uniform distribution Unif(t_min, t_max).
        - If set to 'equally-spaced',
          the points will be fixed to a set of linearly-spaced points that go from t_min to t_max.
        - If set to 'equally-spaced-noisy', a normal noise will be added to the previously mentioned set of points.
        - If set to 'log-spaced', the points will be fixed to a set of log-spaced points that go from t_min to t_max.
        - If set to 'log-spaced-noisy', a normal noise will be added to the previously mentioned set of points,
        - If set to 'chebyshev1' or 'chebyshev', the points are chebyshev nodes of the first kind over (t_min, t_max).
        - If set to 'chebyshev2', the points will be chebyshev nodes of the second kind over [t_min, t_max].
        defaults to 'uniform'.
    :type method: str, optional
    :raises ValueError: When provided with an unknown method.
    """

    def __init__(self, size, t_min=0.0, t_max=1.0, method='uniform', noise_std=None):
        r"""Initializer method
        .. note::
            A instance method `get_examples` is dynamically created to generate 1-D training points.
            It will be called by the function `solve` and `solve_system`.
        """
        super(Generator1D, self).__init__()
        self.size = size
        self.t_min, self.t_max = t_min, t_max
        self.method = method
        if noise_std:
            self.noise_std = noise_std
        else:
            self.noise_std = ((t_max - t_min) / size) / 4.0
        if method == 'uniform':
            self.examples = torch.zeros(self.size, requires_grad=True)
            self.getter = lambda: self.examples + torch.rand(self.size) * (self.t_max - self.t_min) + self.t_min
        elif method == 'equally-spaced':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.getter = lambda: self.examples
        elif method == 'equally-spaced-noisy':
            self.examples = torch.linspace(self.t_min, self.t_max, self.size, requires_grad=True)
            self.getter = lambda: torch.normal(mean=self.examples, std=self.noise_std)
        elif method == 'log-spaced':
            start, end = _compute_log_negative(t_min, t_max, self.__class__)
            self.examples = torch.logspace(start, end, self.size, requires_grad=True)
            self.getter = lambda: self.examples
        elif method == 'log-spaced-noisy':
            start, end = _compute_log_negative(t_min, t_max, self.__class__)
            self.examples = torch.logspace(start, end, self.size, requires_grad=True)
            self.getter = lambda: torch.normal(mean=self.examples, std=self.noise_std)
        elif method in ['chebyshev', 'chebyshev1']:
            self.examples = _chebyshev_first(t_min, t_max, size)
            self.getter = lambda: self.examples
        elif method == 'chebyshev2':
            self.examples = _chebyshev_second(t_min, t_max, size)
            self.getter = lambda: self.examples
        else:
            raise ValueError(f'Unknown method: {method}')

    def get_examples(self):
        return self.getter()

    def _internal_vars(self):
        d = super(Generator1D, self)._internal_vars()
        d.update(dict(
            t_min=self.t_min,
            t_max=self.t_max,
            method=self.method,
            noise_std=self.noise_std,
        ))
        return d


class ConcatGenerator(BaseGenerator):
    r"""An concatenated generator for sampling points,
    whose ``get_examples()`` method returns the concatenated vector of the samples returned by its sub-generators.
    :param generators: a sequence of sub-generators, must have a ``.size`` field and a ``.get_examples()`` method
    :type generators: Tuple[BaseGenerator]
    .. note::
        Not to be confused with ``EnsembleGenerator`` which returns all the samples of its sub-generators.
    """

    def __init__(self, *generators):
        super(ConcatGenerator, self).__init__()
        self.generators = generators
        self.size = sum(gen.size for gen in generators)

    def get_examples(self):
        all_examples = [gen.get_examples() for gen in self.generators]
        if isinstance(all_examples[0], torch.Tensor):
            return torch.cat(all_examples)
        # zip(*sequence) is just `unzip`ping a sequence into sub-sequences, refer to this post for more
        # https://stackoverflow.com/questions/19339/transpose-unzip-function-inverse-of-zip
        segmented = zip(*all_examples)
        return [torch.cat(seg) for seg in segmented]

    def _internal_vars(self) -> dict:
        d = super(ConcatGenerator, self)._internal_vars()
        d.update(dict(
            generators=self.generators,
        ))
        return d


class EnsembleGenerator(BaseGenerator):
    r"""A generator for sampling points whose `get_examples` method returns all the samples of its sub-generators.
    All sub-generator must return tensors of the same shape.
    The number of tensors returned by each sub-generator can be different.
    :param generators: a sequence of sub-generators, must have a .size field and a .get_examples() method
    :type generators: Tuple[BaseGenerator]
    .. note::
        Not to be confused with ``ConcatGenerator`` which returns
        the concatenated vector of samples returned by its sub-generators.
    """

    def __init__(self, *generators):
        super(EnsembleGenerator, self).__init__()
        self.size = generators[0].size
        for i, gen in enumerate(generators):
            if gen.size != self.size:
                raise ValueError(f"gens[{i}].size ({gen.size}) != gens[0].size ({self.size})")
        self.generators = generators

    def get_examples(self):
        ret = tuple()
        for g in self.generators:
            ex = g.get_examples()
            if isinstance(ex, list):
                ex = tuple(ex)
            elif isinstance(ex, torch.Tensor):
                ex = (ex,)
            ret += ex

        if len(ret) == 1:
            return ret[0]
        else:
            return ret

    def _internal_vars(self) -> dict:
        d = super(EnsembleGenerator, self)._internal_vars()
        d.update(dict(
            generators=self.generators,
        ))
        return d


class MeshGenerator(BaseGenerator):
    r"""A generator for sampling points whose `get_examples` method returns a mesh of the samples of its sub-generators.
    All sub-generators must return tensors of the same shape, or a tuple of tensors of the same shape.
    The number of tensors returned by each sub-generator can be different, but the intent behind
    this class is to create an N dimensional generator from several 1 dimensional generators, so each input generator
    should represent one of the dimensions of your problem. An exception is made for
    using a ``MeshGenerator`` as one of the inputs of another ``MeshGenerator``. In that case the original
    meshed generators are extracted from the input ``MeshGenerator``, and the final mesh is used using those
    (e.g ``MeshGenerator(MeshGenerator(g1, g2), g3)`` is equivalent to ``MeshGenerator(g1, g2, g3)``, where
    g1, g2 and g3 are ``Generator1D``).
    This is done to make the use of the ^ infix consistent with the use of
    the ``MeshGenerator`` class itself (e.g ``MeshGenerator(g1, g2, g3)`` is equivalent to g1 ^ g2 ^ g3), where
    g1, g2 and g3 are ``Generator1D``).
    :param generators: a sequence of sub-generators, must have a .size field and a .get_examples() method
    :type generators: Tuple[BaseGenerator]
    """

    def __init__(self, *generators):
        super(MeshGenerator, self).__init__()
        self.generators = []
        for g in generators:
            if isinstance(g, MeshGenerator):
                for s in g.generators:
                    self.generators.append(s)
            else:
                self.generators.append(g)
        self.size = np.prod(tuple(g.size for g in self.generators))

    def get_examples(self):
        ret = tuple()
        for g in self.generators:
            ex = g.get_examples()
            if isinstance(ex, list):
                ex = tuple(ex)
            elif isinstance(ex, torch.Tensor):
                ex = (ex,)
            ret += ex

        if len(ret) == 1:
            return ret[0]
        else:
            ret = torch.meshgrid(ret, indexing='ij')
            ret_f = tuple()
            for r in ret:
                ret_f += (r.flatten(),)
            return ret_f

    def _internal_vars(self) -> dict:
        d = super(MeshGenerator, self)._internal_vars()
        d.update(dict(
            generators=self.generators,
        ))
        return d


class SamplerGenerator(BaseGenerator):
    def __init__(self, generator):
        super(SamplerGenerator, self).__init__()
        self.generator = generator
        self.size = generator.size

    def get_examples(self) -> List[torch.Tensor]:
        samples = self.generator.get_examples()
        if isinstance(samples, torch.Tensor):
            samples = [samples]
        samples = [u.reshape(-1, 1) for u in samples]
        return samples

    def _internal_vars(self) -> dict:
        d = super(SamplerGenerator, self)._internal_vars()
        d.update(dict(
            generator=self.generator,
        ))
        return d
