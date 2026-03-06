# nbody simulation of dark energy and dark matter
## 2d PCA projection of simulation
![simulation](verse.gif?raw=true)

## algorithm
It is assumed Force = 1/r^2.
A force adjacency matrix, Y, is computed between all bodies.
X, the modified forces, are then calculated as Y = dropout(X^2)*Y.
It is noted that some modified forces in X are negative.
Using Newtonian physics, the forces in X are applied to their corresponding bodies.
This algorithm is then repeated.

## symmetry of 2d PCA projection (1 is perfectly symmetric)
![symmetry metric](scale.png?raw=true)

## k complexity of 2d PCA projection (in bytes)
![k complexity](k.png?raw=true)
