Boundary integral method tutorials
============================

Boundary integral methods are powerful tools for solving PDEs. They are particular useful in settings where the object of interest is itself a boundary, for example an earthquake fault surface. 

Over the last few decades, the key parts of a general-purpose boundary integral equation (BIE) solver have come together:

* Quadrature by expansion {cite:p}`Klckner2013` provides a robust approach for numerically calculating singular integrals.
* The fast multipole method {cite:p}`Greengard1987, Ying2004` provides a efficient technique for calculating superficially dense matrix-vector products.
* Direct hierarchical inverse methods {cite:p}`Bebendorf2004, HodlrLib, Greengard2009, Coulier2017` allow for $O(n)$ inversion of the superficially dense BIE matrices. 
* Embedded boundary methods {cite:p}`Ethridge2001, Biros2004` allow for handling body forces and nonlinearities. Smooth function extension methods {cite:p}`Fryklund2018` make this approach accurate.

However, these methods have not become mainstream yet. The only implementation I am aware of that combines most of these methods is {cite:t}`Askham2017`.

This "book" will walk through implementing each of these methods in practical applications. Most of the applications will be to earthquake science problems, however there will be other examples where it makes more sense.

At the moment, there are two sequences:
1. The TDE sequence: Triangular dislocation elements (TDEs) are analytic solutions for the displacement and stress from a triangular fault/crack with a constant displacement discontinuity. While they have major limitations, TDEs are nonetheless frequently used for handling earthquake science modeling problems. I hope this track will help enable TDE users to take their applications to a new level by introducing the capability to handle topography, earth curvature and millions of TDEs.
2. The QBX sequence: Quadrature by expansion is an elegant approach for numerically integrating the singular integrals in boundary integral equations. Combined with a $C^1$ mesh discretization, fast methods and embedded boundary methods, we can build a fast, robust and general PDE solver. Based on these ideas, I will sketch out what I think the next generation of boundary integral equation methods will look like.
