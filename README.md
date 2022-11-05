# graSS_GraphStructureSummarization
Python implementation of GraSS: Graph Structure Summarization graph summarization algorithm introduced by Kristen LeFevre and Evimaria Terzi.

## Problem description
### Problem 1: k-Gs
Given an input graph $G(V,E)$ and integer $k$, find a summary graph $\pmb{S}$ for $G$ with at most $k$ supernodes $\pmb(V)$ $(|\boldsymbol{V}| \leq k)$, such that the $\mathit{Re}(G|\pmb{S})$ is minimized.

### Problem 2: Gs
Given an input graph $G(V,E)$ with adjacency matrix $A$, find a summary $\pmb{S}$ of this graph with expected adjacency matrix $\bar{A}$ such that the total number of bits $$Tb(\bar{A}) = B(\bar{A})+B(A|\bar{A})$$
is minimized.

### Problem 3: k-CGs
Given an input graph $G(V,E)$ and integer $k$, find a summary graph $\pmb{S}$ for $G$ with supernodes $\pmb{V}$ such that $\mathit{Re(G|\pmb{S})}$ is minimized and for every $V' \in \pmb(V)$ $|V'| \geq k$.

## Folder content
- `paper`: this folder contain the scientific publication written by Kristen LeFevre and Evimaria Terzi which describes GraSS graph summarization algorithm

- `graph_data_structure`: containes the basic graph data structure implementation. We rely on the Python implelemtation proposed in this [website](https://towardsdatascience.com/a-complete-guide-to-graphs-in-python-845a0a3381a1). The code in the folder reads a graph from file (input.txt) and prints it on the output stream and in a file named output.txt.
    - *input.txt* file structure: the first line contains two integers *N* and *M* which are the number of nodes and the number of edges respectively. The *M* following lines are populated by couples of integers *u* *v* which represent an edge from node *u* to node *v*
    
- `k-Gs`: implementation of all the algorithms proposed by LeFevre and Terzi to solve the k-Gs problem
    - `greedy`: baseline *Greedy* algorithm
    - `SamplePairs`: *SamplePairs* algorithm
    - `LinearCheck`: *LinearCheck* algorithm
