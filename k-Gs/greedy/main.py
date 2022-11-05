import sys
import graph
import summary
from grass import kgs_greedy

######################################################
"""
Read the graph from input.txt
"""
######################################################
inputFile = open('input.txt','r')

nm = [int(x) for x in inputFile.readline().split()]
N = nm[0]
M = nm[1]

myGraph = graph.Graph()

for inputEdge in inputFile:
    e = [int(x) for x in inputEdge.split()]
    from_v = e[0]
    to_v = e[1]
    myGraph.addEdge(from_v, to_v)
    myGraph.addEdge(to_v, from_v)

######################################################

######################################################
"""
Create initial summary such that each component of the original
graph is a supernode
"""
######################################################
mySummary = summary.Summary()

#Add components
for c in myGraph.getVertices():
    mySummary.addComponent(c)

#Add initial edges between supernodes
for c in myGraph.getVertices():
    graph_component = myGraph.getVertex(c)
    supernode_from = mySummary.getSupernode(c)
    for neighbour in graph_component.getConnections():
        supernode_to = mySummary.getSupernode(neighbour.getId())
        supernode_from.addNeighbor(supernode_to.getId(), 1)
######################################################

kgs_greedy(myGraph, mySummary, 2)

######################################################
"""
Write the summary on output.txt
"""
######################################################
outputFile = open('output.txt','w')
for n in mySummary.getVertices():
    outputFile.write(str(mySummary.getSupernode(n)))
    outputFile.write("\n")
######################################################