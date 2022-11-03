import sys
import graph

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

######################################################

######################################################
"""
Write the graph on output.txt
"""
######################################################
outputFile = open('output.txt','w')
for n in myGraph.getVertices():
    outputFile.write(str(myGraph.getVertex(n)))
    outputFile.write("\n")
######################################################