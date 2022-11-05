class Vertex:
    def __init__(self, key):
        """
        id: unique identifier of the vertex
        connectedTo: dictionary{key:=id of the adjacent vertex ; value:= weight of the directed edge}
        """

        self.id = key
        self.connectedTo = {}
    
    def __str__(self):
        return f"{str(self.id)} connected to: {str([x.id for x in self.connectedTo])}"

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight

    def getConnections(self):
        return self.connectedTo.keys()
    
    def getId(self):
        return self.id
    
    def getWeight(self, nbr):
        return self.connectedTo.get(nbr)