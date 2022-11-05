import supernode

class Summary:
    def __init__(self):
        """
        superList: dictionary{key:=supernode unique identifier ; value:=corresponding supernode object}
        numSupernodes: progressive identifier of the supernodes which populate the data structure. This counter is used to assign auto-increment unique integer identifiers. DO NOT use this attribute to count the number of supernodes which populate the data structure.
        components: dictionary{key:=id of the node of the original graph ; value:=supernode id where the node is in the summary}
        numComponents: number of components of the original graph represented by the summary
        """
        self.superList = {}
        self.numSupernodes = 0
        self.components = {}
        self.numComponents = 0

    def addComponent(self, key):
        """
        Add a new component to the summary.
        Consequently a new supernode is created to host the newcomer.
        """
        
        #add 1 to the number of components
        self.numComponents += 1

        #create a supernode with one component
        newSupernode = supernode.Supernode(self.numSupernodes)
        newSupernode.addComponent(key)
        self.superList[self.numSupernodes] = newSupernode

        #link components "key" to the corresponding supernode
        self.components[key] = self.numSupernodes

        #increment the number of supernodes
        self.numSupernodes += 1
    
    
    def getSupernode(self, key):
        """
        If supernode with key is in the summary then return
        the Supernode.
        """

        #use the get method to return the Supernode if it
        #exists otherwise it will return None
        return self.superList.get(key)
    
    def getComponentSupernode(self, componentId):
        """
        Find the supernode which containes the component with id componentId.
        If there it does not exist, return None.
        """
        if(self.components[componentId]):
            return self.superList.get(components[componentId])
        
        return None
    
    def __contains__(self, key):
        """
        Check whether components with key is in some supernode of the summary.
        """

        #returns True or False depending if in list
        return key in self.components

    def getVertices(self):
        """
        Return all the supernodes in the summary
        """

        return self.superList.keys()

    def getComponents(self):
        """
        Return all the components represented by the summary
        """

        return self.components.keys()

    def merge(self, s1, s2):
        """
        Merge supernode identified by id s1 with supernode identified by id s2.
        Returns the identifier of the new supernode.
        """
        #Check if supernode s1 != s2
        if(s1==s2):
            print(f"Error function merge({s1},{s2}): cannot merge {s1} with itself")
            return

        #Get supernode s1
        supernodeS1 = self.getSupernode(s1)
        if(supernodeS1==None):
            print(f"Error function merge({s1},{s2}): supernode {s1} does not exist")
            return

        #Get supernode s2
        supernodeS2 = self.getSupernode(s2)
        if(supernodeS2==None):
            print(f"Error function merge({s1},{s2}): supernode {s2} does not exist")
            return
        
        #Create new supernode
        supernodeS3 = supernode.Supernode(self.numSupernodes)
        self.superList[self.numSupernodes] = supernodeS3
        self.numSupernodes += 1

        #Add to supernodeS3 all the components of supernodeS1
        for c in supernodeS1.components:
            supernodeS3.addComponent(c)
            self.components[c] = supernodeS3.id

        #Add to supernodeS3 all the components of supernodeS2
        for c in supernodeS2.components:
            supernodeS3.addComponent(c)
            self.components[c] = supernodeS3.id
        
        #Set internal_edges[supernodeS3] = internal_edges[supernodeS1] + internal_edges[supernodeS3] + edges between supernodeS1 and supernodeS3
        supernodeS3.incrementInternalEdges(supernodeS1.getInternalEdges())
        supernodeS3.incrementInternalEdges(supernodeS2.getInternalEdges())
        supernodeS3.incrementInternalEdges(supernodeS1.getWeight(supernodeS2.id))

        #Copy all the neighbours of S1 (different from S2) to S3
        for n in supernodeS1.getConnections():
            if(n != supernodeS2.id):
                w = supernodeS1.getWeight(n)
                supernodeS3.addNeighbor(n, w)

                #Delete and update node n neighbourhood with respect to supernodeS1
                neighbourSupernode = self.getSupernode(n)
                neighbourSupernode.removeNeighbor(supernodeS1.id)
                neighbourSupernode.addNeighbor(supernodeS3.id, w)

        #Copy all the neighbours of S2 (different from S2) to S3: if a neighbour n has been already added in the previous step, increment the weight by 1
        for n in supernodeS2.getConnections():
            if(n != supernodeS1.id):
                w = supernodeS2.getWeight(n)
                supernodeS3.updateNeighbor(n, w)

                #Delete and update node n neighbourhood with respect to supernodeS1
                neighbourSupernode = self.getSupernode(n)
                neighbourSupernode.removeNeighbor(supernodeS2.id)
                neighbourSupernode.updateNeighbor(supernodeS3.id, w)

        #Delete S1 and S2 from superList
        del self.superList[supernodeS1.id]
        del self.superList[supernodeS2.id]        

        #Return S3 id
        return supernodeS3.id
