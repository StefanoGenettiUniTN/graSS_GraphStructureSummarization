import vertex

class Graph:
    def __init__(self):
        """
        vertList: dictionary{key:=vertex unique identifier ; value:=corresponding vertex object}
        numVertices: number of nodes which populate the data structure
        """
        self.vertList = {}
        self.numVertices = 0
    
    def addVertex(self,key):
        """
        Add a vertex to the Graph network with the id
        of key.
        """

        #add 1 to the number of vertices attribute
        self.numVertices += 1

        #instantiate a new Vertex class
        newVertex = vertex.Vertex(key)

        #add the vertex with the key to the vertList dictionary
        self.vertList[key] = newVertex

        #return the NewVertex created
        return newVertex
    
    def getVertex(self, key):
        """
        If vertex with key is in Graph then return
        the Vertex.
        """

        #use the get method to return the Vertex if it
        #exists otherwise it will return None
        return self.vertList.get(key)
    
    def __contains__(self, key):
        """
        Check whether vertex with key is in the Graph.
        """

        #returns True or False depending if in list
        return key in self.vertList

    def addEdge(self, f, t, weight=0):
        """
        Add an edge to connect two vertices of t and f
        with weight assuming directed graph
        """

        #add vertices if they do not exist
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        
        #then add Neighbor from f to t with weight
        self.vertList[f].addNeighbor(self.vertList[t], weight)

    def getVertices(self):
        """
        Return all the vertices in the graph
        """

        return self.vertList.keys()

    def getAdjacency(self, v, u):
        """
        Return the entry of the adjacency matrix [v,u]
        """
        vertexV = self.getVertex(v)
        vertexU = self.getVertex(u)

        if vertexU in vertexV.getConnections():
            return 1
        
        return 0