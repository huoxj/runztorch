from typing import Dict, List, Any, Iterator, Union

from nn import Parameter
from tensor import Tensor

class Module:
    _parameters: Dict[str, Parameter]
    _buffers: Dict[str, Tensor]
    _modules: Dict[str, "Module"]

    def __init__(self) -> None:
        self._parameters = {}
        self._buffers = {}
        self._modules = {}

    def forward(self, *input: Any):
        raise NotImplementedError(
            f"Module [{type(self).__name__}] is missing the required \"forward\" function."
        )

    def register_parameter(self, name: str, param: Parameter):
        self._parameters[name] = param
    
    def register_buffer(self, name: str, buffer: Tensor):
        self._buffers[name] = buffer

    def register_module(self, name: str, module: "Module"):
        self._modules[name] = module

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for param in self._parameters.values():
            yield param
        if recurse:
            for module in self._modules.values():
                yield from module.parameters(recurse=True)
    
    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        for buffer in self._buffers.values():
            yield buffer
        if recurse:
            for module in self._modules.values():
                yield from module.buffers(recurse=True)
    
    def modules(self) -> Iterator["Module"]:
        yield self
        for module in self._modules.values():
            yield module
            yield from module.modules()
        
    def __call__(self, *input: Any):
        return self.forward(*input)
    
    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if name in self._parameters:
            return self._parameters[name]
        if name in self._buffers:
            return self._buffers[name]
        if name in self._modules:
            return self._modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
        if isinstance(value, Parameter):
            self.register_parameter(name, value)
        elif isinstance(value, Tensor):
            self.register_buffer(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        else:
            super().__setattr__(name, value)
    
    def __delattr__(self, name: str) -> None:
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)
