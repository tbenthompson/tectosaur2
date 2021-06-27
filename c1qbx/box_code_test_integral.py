from sage.all import *
import pickle

sx, sy = var('sx, sy')

ox, oy = var('ox, oy')

dx = ox - sx
dy = oy - sy
r2 = (dx ** 2) + (dy ** 2)

r = sqrt(r2)
G = (1 / (2 * pi)) * log(r)
f = (1-sx**2)*(1-sy**2)
integrand = G * f
print('integrand', integrand)

# The opposite assumption that `sy - oy < 0` results in the same solution
# so this is sufficient.
assume(sy-oy>0)
Ia = integral(integrand, sx, -1, 1)
print('sx integral', Ia)

assume(ox < 1)
assume(oy < 1)
assume(ox > -1)
assume(oy > -1)
I1 = integral(Ia, sy, -1, 1)
print('full integral', I1)

sympy_I1 = I1._sympy_()
with open('test_integral.pkl', 'wb') as f:
    pickle.dump(sympy_I1, f)

# Proof that the sy-oy assumption is okay.
forget()

assume(sy-oy<0)
Ib = integral(integrand, sx, -1, 1)

assume(ox < 1)
assume(oy < 1)
assume(ox > -1)
assume(oy > -1)
I2= integral(Ib, sy, -1, 1)
print(I1 - I2)
