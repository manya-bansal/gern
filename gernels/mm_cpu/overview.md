This is an example for implementing, fast, single-threaded matrix multiplication on the CPU using Gern, assuming that we are given an efficient inner kernel capable of calculating matrix multiplication efficiently for small (maybe even fixed sized) submatrices.

Assume we are multiplying a N x K matrix named A with a K x M matrix named B, resulting in a N x M matrix named C. The simplest implementation is to tile over the dimenstions N, M, and reduce over K. A simple observation is that we can tile over each dimension multiple times, reducing the tiling size at each turn, giving better cache behavior. 

These is algorithm is very easy to implement in Gern. We define a data type for our matrices, which consist of a pointer into memory, the dimensions of the matrix, and the leading dimension. Queries return back 'views' that point to the correct place in the same place in memory. With this, and the tile and reduce properties of Gern, implementation is trivial.



However, it is possible to do better: https://www.cs.utexas.edu/~pingali/CS378/2008sp/papers/gotoPaper.pdf
As seen in this paper, we need to "pack" certain submatrices of the matrices A and B at certain points, (i.e copying it over to some auxillary memory such that the submatrix becomes continuos). This allows us to fit that submatrices entirely into the cache, giving better memory utilization.

This more complicated behavior can be implemented in Gern. In addition to our 'view' representation desribed in previous paragraphs, we can introduce a 'packed' representation, which stores the query in a static array, and takes in the dimensions of the matrix as template arguments. Then, to our 'view' representation, we can add query functions that takes in the dimensions of the submatrix as query arguments, and return a 'packed' submatrix. Using 'packed' and 'view' matrix representations whenever possible in our Gern program, and making use of either staging, or compiling a Gern program and feeding it into another Gern program, we can easily implement this algorithm. 



Another thing to consider is that, so far, we assumed our inner kernel could perform matrix muliplication for any given two submatrices. This, however, is not always the case. For example, in the blis library, the kernel expects two matrices A and B to multiply, with A being in column major representation, and B being in row major representation, and both of them being continuos in memory. This restriction requires a more complicated "packing" strategy, requiring us to pack the kernel in some sort of 'zig zag fashion.' Detailed explanation can be found in https://dl.acm.org/doi/pdf/10.1145/2764454.

We can introduce these alternative data representations to Gern as well, and once again introduce the needed query functions. However, at this point the boilerplate code needed increases a lot. For N different representation, we need N^2 query functions. Moreover, in the layered structure of a gern program, for each layer we need a data structure that returns a query for the data structure in the lower level, and so on. To avoid writing unnecessary and bloated boilerplate code, CRTP can be used to seperate the query logic and data representation logic.