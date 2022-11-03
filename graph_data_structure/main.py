import graph

myGraph = graph.Graph()

myGraph.addEdge(0, 1)
myGraph.addEdge(1, 0)
myGraph.addEdge(1, 2)
myGraph.addEdge(2, 1)
myGraph.addEdge(0, 2)
myGraph.addEdge(2, 0)

print(myGraph.numVertices)