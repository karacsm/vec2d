import numpy as np
import numpy.lib.mixins
from numbers import Number

class Vec2D(numpy.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, x: Number, y: Number):
        self._x = x
        self._y = y
        
    def __str__(self):
        return f"(x={self._x}, y={self._y})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(x={self._x}, y={self._y})"
    
    def __array__(self, dtype=None):
        return np.array([self._x, self._y], dtype = dtype)
    
    def __iter__(self):
        yield from (self._x, self._y)
        
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        processed_inputs = (input.__array__() if isinstance(input, self.__class__) else input for input in inputs)
        if method == "__call__":
            output = ufunc(*processed_inputs, **kwargs)
        elif method == "reduce":
            output = ufunc.reduce(*processed_inputs, **kwargs)
        elif method == "outer":
            output = ufunc.outer(*processed_inputs, **kwargs)
        elif method == "accumulate":
            output = ufunc.accumulate(*processed_inputs, **kwargs)
        else:
            return NotImplemented
        if output.shape == (2,):
            return self.__class__(*output)
        else:
            return output
        
    def copy(self):
        return self.__class__(self._x, self._y)
        
    @property
    def x(self):
        return self._x
    @x.setter
    def x(self, value):
        self._x = value
    
    @property
    def y(self):
        return self._y
    @y.setter
    def y(self, value):
        self._y = value

def normalize_vector(v: [np.ndarray[Number] | Vec2D]) -> [np.ndarray[Number] | Vec2D]:
    m = np.max(np.abs(v))
    if m == 0:
        return np.zeros(v.shape)
    v = v / m
    r = np.linalg.norm(v)
    v = v / r
    return v
