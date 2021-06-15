Boundary integral method tutorials
============================

## Summary

Boundary integral methods are powerful tools for solving PDEs. They are particular useful in settings where the object of interest is itself a boundary, for example an earthquake fault surface. 

Over the last few decades, the key parts of a general-purpose boundary integral equation (BIE) solver have come together:

* Quadrature by expansion {cite:p}`Klckner2013` provides a robust approach for numerically calculating singular integrals.
* The fast multipole method {cite:p}`Greengard1987, Ying2004` provides a efficient technique for calculating superficially dense matrix-vector products.
* Direct hierarchical inverse methods {cite:p}`Bebendorf2004, HodlrLib, Greengard2009, Coulier2017` allow for $O(n)$ inversion of the superficially dense BIE matrices. 
* Embedded boundary methods {cite:p}`Ethridge2001, Biros2004` allow for handling body forces and nonlinearities. Smooth function extension methods {cite:p}`Fryklund2018` make this approach accurate.

However, these methods have not become mainstream yet. The only paper I am aware of that combines most of these methods is {cite:t}`Askham2017` but the source is not available.

This "book" will walk through implementing each of these methods in practical applications. Most of the applications will be to earthquake science problems, however there will be other examples where it makes more sense.

At the moment, there are two sequences:
1. The TDE sequence: Triangular dislocation elements (TDEs) are analytic solutions for the displacement and stress from a triangular fault/crack with a constant displacement discontinuity. While they have major limitations, TDEs are nonetheless frequently used for handling earthquake science modeling problems. I hope this track will help enable TDE users to take their applications to a new level by introducing the capability to handle topography, earth curvature and millions of TDEs.
    * In sections 1 and 2, I start by demonstrating how to use TDEs to model arbitrary fault surfaces and Earth surface topography. 
    * Then, in sections 3, 4 and 5, I explain how to combine modern fast methods like H-matrices and adaptive cross approximation with TDEs. 
2. The QBX sequence: Quadrature by expansion is an elegant approach for numerically integrating the singular integrals in boundary integral equations. Combined with a $C^1$ mesh discretization, fast methods and embedded boundary methods, we can build a fast, robust and general PDE solver. Based on these ideas, I will sketch out what I think the next generation of boundary integral equation methods will look like.
    * In sections 1 and 2, I will explain how to use QBX to evaluate nearfield integrals and then demonstrate QBX on a variety of interesting geometries.
    <!-- * In section 3, I will extend QBX from nearfield integrals to directly evaluating singular boundary integral equations. That will allow us to solve a BEM problem. -->
    <!-- * In section 4, I will introduce basis splines as a method for general purpose $C^1$ mesh discretization. -->
    <!-- * In section 5, I will introduce body force integrals and solve a simple time-dependent viscoelastic earthquake problem. -->

## Prerequisites

The tutorials here are directed at a fairly experienced audience. Some prior knowledge and skills that would be helpful:
* Python skills: particularly standard scientific Python tools like `numpy` and `matplotlib`. A lot of the code here is vectorized because it allows me to communicate the ideas more concisely while also having reasonably fast code. If you feel that your Python skills are weak, the tutorials here could be a decent way to learn. You might consider downloading the notebooks and running them yourself to explore.
* Applied math skills: multi-variable calculus, basic partial differential equations, numerical linear algebra. Do you know what it means to "Use an LU decomposition to solve the linear system resulting from a finite difference discretization of the Poisson equation?" I'll try to provide references when I'm discussing more technical topics.
