# graSS_GraphStructureSummarization
Python implementation of GraSS: Graph Structure Summarization graph summarization algorithm introduced by Kristen LeFevre and Evimaria Terzi.

## Folder content
- `paper`: this folder contain the scientific publication written by Kristen LeFevre and Evimaria Terzi which describes GraSS graph summarization algorithm

- `graph_data_structure`: containes the basic graph data structure implementation. We rely on the Python implelemtation proposed in this [website](https://towardsdatascience.com/a-complete-guide-to-graphs-in-python-845a0a3381a1). The code in the folder reads a graph from file (input.txt) and prints it on the output stream and in a file named output.txt.
    - *input.txt* file structure: the first line contains two integers *N* and *M* which are the number of nodes and the number of edges respectively. The *M* following lines are populated by couples of integers *u* *v* which represent an edge from node *u* to node *v*