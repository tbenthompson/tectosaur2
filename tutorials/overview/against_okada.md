
# Why Okada dislocations often don't work

<sub>Everything here is well known among some groups of researchers but it's still very common to find people working on boundary-integral-based earthquake modeling that are running into these problems without being aware of it.</sub>

Okada dislocations and triangular dislocations are popular fault modeling tools. But, a lot of the time, these methods don't do what we want them to do! Because they suffer from stress singularities and missing stress components, they fail to accurately model faults beyond the simplest of settings.

The basic idea is of a flat rectangular or triangular surface with a constant slip applied. From that input geometry and slip, we can calculate the resulting displacement and stress anywhere in the volume under consideration.

TODO: look at brendan's paper
TODO: look at my previous writing on this.

## Stress singularities!

By necessity, a fault that is modeled with constant slip dislocation elements will have step functions in the slip. (SHOW FIGURE HERE with slip and stress)

## Missing stress components

Another side effect of having stepwise-constant slip is that the stress components that would result from gradients in the true slip distribution are exactly zero .

## There are a lot of papers that probably suffer from these issues


## When can Okada be used safely?

There are two situations where it's okay to use Okada or triangular dislocations:
1. Okada dislocations work okay-ish for modeling friction on a perfectly planar and rectangular fault where dislocations are all the same size. (footnote: Andrew Bradley worked on some techniques to allow different size Okada dislocations.)
2. If you only plan to evaluate the resulting displacements and stresses away from the fault. To be concrete, the approximation should be correct any distance beyond about 1 element length away from the fault.

## What should you do instead?
There are
