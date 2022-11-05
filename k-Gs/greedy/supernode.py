class Supernode:
    def __init__(self, key):
        """
        id: unique identifier of the supernode
        connectedTo: dictionary{key:=id of the adjacent supernode ; value:= number of edges between the two connected supernodes}
        cardinality: number of components which populate the supernode
        internalEdges: number of connections among the components which populate the supernode
        components: set of components ids which populate the supernode
        """

        self.id = key
        self.connectedTo = {}
        self.cardinality = 0
        self.internalEdges = 0
        self.components = set()
    
    def __str__(self):
        return f"supernode [{str(self.id)}] \n cardinality: {str(self.cardinality)} \n connected to (id, number_of_edges): {str([(x, self.connectedTo[x]) for x in self.connectedTo])} \n components: {str(self.components)} \n internal edges: {str(self.internalEdges)}"

    def addNeighbor(self, nbr, weight=1):
        self.connectedTo[nbr] = weight

    def updateNeighbor(self, nbr, weight):
        if(nbr in self.connectedTo.keys()):
            self.connectedTo[nbr] += weight
        else:
            self.connectedTo[nbr] = weight

    def removeNeighbor(self, nbr):
        if(nbr in self.connectedTo.keys()):
            del self.connectedTo[nbr]
    
    def getConnections(self):
        return self.connectedTo.keys()
    
    def getId(self):
        return self.id
    
    def getWeight(self, nbr):
        if nbr in self.connectedTo:
            return self.connectedTo.get(nbr)
        return 0

    def addComponent(self, key):
        """
        Increment the cardinality of the supernode
        """
        self.cardinality += 1
        self.components.add(key)

    def getComponents(self):
        return self.components

    def getInternalEdges(self):
        return self.internalEdges
    
    def incrementInternalEdges(self, numEdges):
        self.internalEdges += numEdges