# vec2d
2D vector implementation using numpy.

They work like numpy arrays, but they have only 2 components x and y.
Vec2D objects are interoperable with most of numpy's functions and operations; 
The Vec2D class implements the `__array__` and `__array_ufunc__` [numpy interoperability magic methods](https://numpy.org/devdocs/user/basics.interoperability.html) and inherits the basic operator overloads.
