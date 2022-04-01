# Design criteria for a practical boundary integral solver

Boundary integral equation methods are a common tool for solving computational modeling problems in earthquake science. Despite that, at the moment, the tooling is a major limitation for the field. Improving both the algorithms and the software is important!

Boundary integral equation (BIE) methods are very promising for a set of problems where the object under study is itself a boundary! At the moment, the standard methodology is to use constant basis function boundary elements for [which](okada) [analytical](tdes) [solutions](qdyn) have been derived. Generally, these elements are used to represent the fault in either a fullspace or halfspace. There are some efforts to move beyond these simple methods but those efforts have not yet panned out. The main barrier is lack of software and knowledge of appropriate algorithms. I think QBX-based integration combined with FMM or H-matrix acceleration would be very powerful.

See [this section on why Okada and triangular dislocations don't work](./against_okada)

## When should we be using an integral equation method?

As discussed in the [introduction](./intro), "integral equation" methods have the potential to be as general as a differential method like a finite element or finite difference method. This comes from the simple observation that almost any physics-derived PDE has a linear homogeneous "core" to it with some kind of nonlinear or inhomogeneous overlay. The linear core can normally be transformed to an integral form involving integrals over kernels. Then, the nonlinear component can be handled with a volumetric integral over the whole domain. For an example of this, consider a problem

POISSON PROBLEM.

This right-hand side $f$ can be essentially any (well-behaved) function, even one that is a function of the unknown $u$.
The numerical behavior

While an integral equation method may have similar generality to This does not mean that integral equation methods are the right choice in all situations. Far from it. But, there are still many advantages. [^adv]

1. A focus on the boundaries, which can be very helpful for minimizing mesh generation effort. This is probably the largest benefit.
2. Advantageous numerical properties. For example, when posed as an integral equation, many problems have tiny condition numbers that remove the need for preconditioners in iterative linear solvers. Another example: numerical error for derivatives of the solution is often much lower than in a differential method.
3. Good computational behavior. For example, a lot of integral equation algorithms use more floating point operations and fewer memory accesses than differential methods like a finite element method. This is a better fit for modern hardware like GPUs.
4. Often, adaptive meshing is much simpler in an integral equation context.


## Target capabilities

So, what would that hypothetical BIE software look like? Let's start by just listing the types of applications that would be nice to tackle, starting with "absolutely critical" down to "nice to have".

* linear elasticity
* complex realistic geometries
  * faults
  * topography
  * material interfaces
  * fault roughness
* large models (large enough that $O(n^2)$ methods won't work)
* high performance
* earthquake rupture modeling
* linear and nonlinear viscoelastic behavior
* elastic-plastic behavior
* seismic waves

This list is probably biased to my personal preferences and the things I've worked on so far, but also covers a huge range of problems. Solving even two or three of these bullet points would be a huge contribution!

## Software design and coding practices

https://blog.nelhage.com/2016/03/design-for-testability/

Software design is a critically important component of getting things done quickly and correctly. I see a lot of software that was either not designed at all or was designed by someone who had read a book or two on software design but had no real-world practice.[^1] Advice in books and online about software design and coding best practices is notoriously bad. Not all of it, but enough of it that if you read it without a critical eye, you might end up massively overengineering your code or writing functions that are four lines each.

[^1]: I actually often find the software that wasn't design at all to often be better than the super overengineered montrosities.

With that little screed out of the way, what am I looking for in terms of software design and ?

- **interfaces that are deep.** In the case of a integral equation solver, I'd like to be able to specify an integral that I want to compute with a single function call.
- **clear organization.** Put conceptually related concepts in the same place.
- **medium-sized, meaningful functions.** A proliferation of tiny function calls makes code super confusing. On the other hand, 2,000 line functions can be very hard to follow and understand.
- **a reasonable range of tests that test things I care about.** Does the approximate algorithm for computing an integral return values with the expected error with respect to the non-approximate algorithm?

## Test cases

Testing an implementation of a numerical method is a notoriously hard thing to do. If you read the internet about how to write automated tests, you'll get tons of posts that explain a particularly approach to unit testing in the context of some kind of business logic where the programmer knows exactly what the code is supposed to do. When we're testing numerical methods, we *don't know what the code is supposed to do!!* I've written two pieces before on this topic: [here](https://tbenthompson.com/post/automated_testing_for_science/) and [here](https://tbenthompson.com/post/testing_scientific_software/).

When looking at a boundary integral and volumetric integral method, there are a few components we can test:

* At the highest level, the full integrated elastic solver can be thoroughly tested. There are a wide variety of idealized elasticity solutions.
  * The analytical solutions for the displacement from a constant amount of slip on a line segment, triangle or rectangular are available. These are called "dislocation solutions" and are common building blocks for boudnary element methods. For line segments, see sections 2.1 and 2.2 of "Earthquake and Volcano Deformation" by Paul Segall. In 3D with constant slip on a triangle, see [the software here](https://github.com/tbenthompson/cutde) and the paper referenced there.
  * The solution for slip on a line segment in a 2D model is particularly useful. I replicated [it using QBX-based BIE methods here](https://tbenthompson.com/book/c1qbx/part3_topo_solver.html). One specific solution that is very useful is exactly
  * Another useful solution is that of a compressed sphere. Solutions for this problem and other similar "elementary elasticity" problems can be found in "Theory of Elasticity" by Timoshenko and Goodier. That book is a treasure trove of useful analytical test problems.
* Several extensions of the elastic solver can be tested in special cases, for example, if the deeper regions are visco-elastic instead of elastic.
* Approximation algorithms like the fast multipole method and hierarchical matrix methods can be tested on arbitrary problems by comparing against the non-approximate solution.
* If we can handle arbitrary integral terms (for example, a body force in an elasticity problem), then we can test our software using *the method of manufactured solutions*.
* **collaborative benchmarking**: I don't know of any analytical solutions for earthquake cycle simulation, but there are several useful testing avenues:There is a project, "The SCEC Sequences of Earthquakes and Aseismic Slip Project" that is a community to benchmark earthquake simulation software solutions against each other. While that doesn't prove that any particular piece of software is correct, it does help to identify errors.
* I've been working on an implementation of body force handling which would enable *method of manufactured solutions* testing.
