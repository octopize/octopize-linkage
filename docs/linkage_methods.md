# Linkage methods

This page describes the different approaches available to perform linkage between two data sources.

## Concept

Linkage of datasets requires two main components: a notion of distance between two records and an algorithm to associate records from A to records from B.

## Distances

### Gower distance

The gower distance can be used on data containing both numeric and non-numeric variables. This distance is a natural option to consider when computing distances between records in datasets as most real-life datasets contain mixed data types. Gower can be interpreted as a combination of Euclidean and Hamming distance.

For a dataset with a set of categorical variable $C$ and a set of numerical variables $N$, the Gower distance between two records \( x \) and \( y \) is given by:

$$
D_{gower}(x, y) = \sum_{i \in C}{(\{x_i\}-\{y_i\})} + \sum_{i \in N}{1-\frac{|x_i - y_i|}{R_i}} 
$$

$R_i, i \in N$ refers to the value range of variable $i$ in the whole dataset

This library uses the [*gower* library available on pypi](https://pypi.org/project/gower/).


### Euclidean distance

Although, Euclidean distance cannot be used directly on non-numeric data, a dataset can be projected into a multidimensional numeric space in which all records have numeric only coordinates. Factor Analysis can be used for this purpose and this is the solution used in this library by means of Principal Component Analysis (PCA), Factor Analysis of Mixed Data (FAMD) and Multiple Correspondence Analysis (MCA). 

Following a projection P (representing records in |P| dimensions), The Euclidean distance between two records \( x \) and \( y \) is calculated between their coordinates as:

$$
D_{eucl}(x, y) = \sqrt{\sum_{i \in |P|} (P_i(x) - P_i(y))^2}
$$

$P_i(x)$ refers to the projection of $x$ on the $i$-th dimension (i.e. its $i$-th coordinate)  

Before projection, a model needs to be fitted. This can be done either on data from source A or from source B or from both sources (this only really has an influence if anonymization has been performed). The 3 options are available but no significance difference has been observed. For most experiments, a model fitted on both data is used. 

This library uses [*saiph*, available on pypi](https://pypi.org/project/saiph/) for projection using factor analysis. 

## Linkage algorithms

### Linear Sum Assignment (LSA) linkage

LSA is a classic combinatorial optimization problem which aims at assigning a set of objects to another set of objects in a way that the overall assignment cost is minimized.

A solution to a LSA problem is required to be bijective, i.e. any object in $A$ must be assigned to exactly one object in $B$.

In the context of data linkage between two data sources A and B, the objective is to find a mapping $M$ such that the sum of distances over all associated pairs of individuals $x$ and $y$ is minimized:

$$
min \sum_{x, y \in |A|}{D(x,y)*M_{x,y}}
$$

LSA is not bound to any distance and so all distances described in the earlier section can be used.

There is active research on solving LSA problems. In this library, we rely on the [solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) provided by scipy. 

For further details about LSA solvers (including the one used by scipy), we suggest the following paper:

[Dell'Amico, M. and Toth, P. *Algorithms and codes for dense assignment problems: the state of the art*. 2000. Discrete Applied Mathematics.](https://www.sciencedirect.com/science/article/pii/S0166218X99001729)

It is important to note that the complexity of the best LSA solvers is $\mathcal{O}(n^3)$. This may impact linkage of datasets containing many records.

Although not covered in this library, this could be handled by dividing the data in smaller subsets on which LSA can be run in reasonable time. This would however results in an approximation of what a solver would obtain on the whole dataset.

### Greedy linkage

The library also includes some simple greedy algorithms. Those algorithms are simple and easily understandable. However, their performance is limited in comparison with LSA and their use is not recommended on real use cases.

#### `Min order`

The first greedy algorithms, currently named `min_order`, orders all records in $A$ by $min(D(x))$, its closest distance to another record in $B$. Ordered records are then allocated in this order to the closest non-assigned records in $B$.

#### `Min re-order`

The second greedy algorithms, currently named `min_reorder` only differs from `min_order` by the fact that records in $A$ are re-ordered by $min(D(x))$ *after each allocation*. This algorithm provides better results that `min_reorder` but requires significantly more resources.

## Baseline algorithms

To understand the performance of the different linkage options, two baseline linkage methods are available.

### Row order linkage

Row order linkage between two sources A and B will allocate the $i$-th individual from A to the $i$-th individual from B.  

$$
M = \begin{bmatrix}
\begin{matrix}
    1 &  &  & 0 \\
    0 & 1 &  & 0 \\
    \vdots &  & \ddots & \vdots \\
    0& & & 1
  \end{matrix}
  \end{bmatrix} 
$$

In experimental contexts where the real order is known and the data not anonymized, *row_order* linkage yield perfect matching. On similar contexts where the data has been anonymized (but not shuffled), *row_order* will not produce the best linkage but a good linkage. We can consider *row_order* as a good objective. Linkage algorithms approaching the quality of *row_order* can be considered as good.

### Random linkage

To compare linkage solutions, a random linkage approach is made available. Its results will represent a lower bound. 