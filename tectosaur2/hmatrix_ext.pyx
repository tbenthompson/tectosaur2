cdef extern from 'hmatrix.cpp':
    double square(double x)

def f(double x):
    return 3*square(x)
