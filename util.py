import logging
import re
from functools import wraps
from re import Pattern
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch as T


def map_module(root: T.nn.Module,
               func: Callable[[T.nn.Module, str], T.nn.Module],
               patt: Optional[str] = None) -> T.nn.Module:
    """Function ``map_module`` applies a function to each leaf of module tree
    which matches to a specified pattern.

    Parameters
    ----------
    root : torch.nn.Module
        Module to modify.
    func : callable
        Function to be applied to every module (or matched to pattern) in
        module tree.
    patt : str, optional
        Pattern to filter modules by path in module tree.

    Returns
    -------
    torch.nn.Module
        Module modified in-place.
    """
    @wraps(func)
    def func_safe(*args, **kwargs):
        node = func(*args, **kwargs)
        if not isinstance(node, T.nn.Module):
            raise ValueError('Mapped result must be toch.nn.Module type '
                             f'but given {type(node)}.')
        return node

    return _map_module(root, func_safe, re.compile(patt or r'.*'), '')


def _map_module(root: T.nn.Module,
                func: Callable[[T.nn.Module, str], T.nn.Module], patt: Pattern,
                path: str) -> T.nn.Module:
    for name, child in root.named_children():
        node = _map_module(child, func, patt, f'{path}/{name}')
        if node != child:
            setattr(root, name, node)
    if patt.match(path or '/'):
        root = func(root, path or '/')
    return root


def numel(module: T.nn.Module):
    value = sum(x.numel() for x in module.parameters()) + \
            sum(x.numel() for x in module.buffers())

    def account_prunned(module: T.nn.Module, path: str):
        nonlocal value
        for name, attr in vars(module).items():
            if not name.endswith('_mask') or not isinstance(attr, T.Tensor):
                continue

            weight_name = name[:-5]
            if not hasattr(module, weight_name):
                continue

            weight = getattr(module, weight_name)
            value -= weight.numel() - attr.sum()
            value += attr.numel()
        return module

    def account_quantized(module: T.nn.Module, path: str):
        nonlocal value
        if isinstance(module, T.nn.quantized.Linear):
            value += module.weight().numel()
            if module.bias() is not None:
                value += module.bias().numel()
        return module

    def account_rest(module: T.nn.Module, path: str):
        account_prunned(module, path)
        account_quantized(module, path)
        return module

    map_module(module, account_rest)
    return value


def sizeof(module: T.nn.Module):
    value = sum(x.numel() * x.element_size() for x in module.parameters()) + \
            sum(x.numel() * x.element_size() for x in module.buffers())

    def account_prunned(module: T.nn.Module, path: str):
        nonlocal value
        for name, attr in vars(module).items():
            if not name.endswith('_mask') or not isinstance(attr, T.Tensor):
                continue

            weight_name = name[:-5]
            if not hasattr(module, weight_name):
                continue

            weight = getattr(module, weight_name)
            value -= (weight.numel() - attr.sum()) * weight.element_size()
            value += attr.numel() * attr.element_size()
        return module

    def account_quantized(module: T.nn.Module, path: str):
        nonlocal value
        if isinstance(module, T.nn.quantized.Linear):
            value += module.weight().numel() * module.weight().element_size()
            if (bias := module.bias()) is not None:
                value += bias.numel() * bias.element_size()
        return module

    def account_rest(module: T.nn.Module, path: str):
        account_prunned(module, path)
        account_quantized(module, path)
        return module

    map_module(module, account_rest)
    return value


def flatten_module(module: T.nn.Module, regexp=None) -> Dict[str, T.nn.Module]:
    modules = {}
    map_module(module, lambda x, y: modules.update(**{y: x}) or x, regexp)
    return modules


def print_flatten(module: T.nn.Module):
    paths = []
    path_len = 0
    names = []
    name_len = 0
    indx_len = 0

    def func(module, path):
        nonlocal path_len, name_len, indx_len
        paths.append(path)
        path_len = max(path_len, len(path))
        name = module.__class__.__name__
        names.append(name)
        name_len = max(name_len, len(name))
        indx_len += 1
        return module

    map_module(module, func)

    indx_len = int(np.ceil(np.log10(indx_len)))
    fmt = f'{{indx:>{indx_len}s}} {{path:{path_len}s}} {{name:{name_len}s}}'
    print(fmt.format(indx='#', path='Path', name='Layer'))
    print('-' * (indx_len + path_len + name_len + 2))
    for i, (path, name) in enumerate(zip(paths, names)):
        print(fmt.format(indx=str(i), path=path, name=name))
