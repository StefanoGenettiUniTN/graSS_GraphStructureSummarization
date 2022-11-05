import summary
import graph
import supernode
from itertools import combinations
import numpy as np
import random

def kgs_greedy(graph, summary, k):
    '''
    Given an input graph G(V,E) and integer k, find a summary graph S
    for G with at most k supernodes V (|V| <= k),
    such that the Re(G|S) is minimized.
    '''
    supernodeCounter = len(summary.superList)

    while(supernodeCounter>k):
        #Check all the possible couples of supernodes to find the couple to be merged
        minReIncrement = float('inf')
        possibleSuperCouple = list(combinations(summary.superList.keys(), 2))
        for c in possibleSuperCouple:
            s1 = summary.getSupernode(c[0])
            s2 = summary.getSupernode(c[1])

            '''
            Simulate merge and compute reconstruction error
            '''
            reconstructionError = 0

            #Create new supernode
            supernodeS3 = supernode.Supernode(summary.numSupernodes)

            ###Compute internal reconstruction error###

            #Set internal_edges[supernodeS3] = internal_edges[supernodeS1] + internal_edges[supernodeS3] + edges between supernodeS1 and supernodeS3
            supernodeS3.incrementInternalEdges(s1.getInternalEdges())
            supernodeS3.incrementInternalEdges(s2.getInternalEdges())
            supernodeS3.incrementInternalEdges(s1.getWeight(s2.id))

            #Total number of components in supernode S3
            supernodeS3TotNodes = s1.cardinality + s2.cardinality

            #Internal adjacency of S3
            internal_adj = (2*supernodeS3.getInternalEdges())/(supernodeS3TotNodes*(supernodeS3TotNodes-1))

            #For each possible couple of S3 components, we update the reconstruction error
            #with respect to the ground truth
            for c1 in s1.getComponents():
                for c2 in s2.getComponents():
                    trueAdjacency = graph.getAdjacency(c1, c2)
                    reconstructionError += abs((trueAdjacency)-(internal_adj))

            ###end compute internal reconstruction error###

            ###Compute external reconstruction error###

            #Copy all the neighbours of S1 (different from S2) to S3
            for n in s1.getConnections():
                if(n != s2.id):
                    w = s1.getWeight(n)
                    supernodeS3.addNeighbor(n, w)

            #Copy all the neighbours of S2 (different from S2) to S3: if a neighbour n has been already added in the previous step, increment the weight by 1
            for n in s2.getConnections():
                if(n != s1.id):
                    w = s2.getWeight(n)
                    supernodeS3.updateNeighbor(n, w)
            
            #For each neighbour of s3 update the reconstruction error
            for n in supernodeS3.getConnections():
                
                supernodeN = summary.getSupernode(n)

                #External adjacency between nodes in S3 and nodes in its neighbour n
                external_adj = supernodeS3.getWeight(n)/(supernodeS3TotNodes*supernodeN.cardinality)

                #For each possible couple between S3 and components and n components, we update the reconstruction error
                #with respect to the ground truth
                for c1 in supernodeN.getComponents():
                    for c2 in s1.getComponents():
                        trueAdjacency = graph.getAdjacency(c1, c2)
                        reconstructionError += abs((trueAdjacency)-(external_adj))

                    for c2 in s2.getComponents():
                        trueAdjacency = graph.getAdjacency(c1, c2)
                        reconstructionError += abs((trueAdjacency)-(external_adj))                

            ###end compute external reconstruction error###

            if reconstructionError<minReIncrement:
                minReIncrement = reconstructionError
                bestS1 = s1.getId()
                bestS2 = s2.getId()

            '''
            end simulate merge
            '''

        ###Merge the couple which leads to the smaller Re increment
        summary.merge(bestS1, bestS2)
        
        #Decrease number of supernode
        supernodeCounter -=1

def kgs_sample_pairs_constant(graph, summary, k, num_pair):
    '''
    Given an input graph G(V,E) and integer k, find a summary graph S
    for G with at most k supernodes V (|V| <= k),
    such that the Re(G|S) is minimized.
    '''

    '''
    In this implementation of the SamplePairs algorithm, we pick
    a constant number of pairs in every round K. The number of pairs
    which are picked at every step is set by parameter num_pair
    '''

    supernodeCounter = len(summary.superList)

    while(supernodeCounter>k):
        #Check all the possible couples of supernodes to find the couple to be merged
        minReIncrement = float('inf')

        possibleSuperCouple = np.array(list(combinations(summary.superList.keys(), 2)))

        #Sample num_pair couples at random with uniform probability
        if num_pair < possibleSuperCouple.size: #if the number of possible couples is less than the number of pairs we want to pick, we have to take into account all the possible pairs
            possibleSuperCouple = possibleSuperCouple[np.random.choice(len(possibleSuperCouple), num_pair)]

        for c in possibleSuperCouple:
            s1 = summary.getSupernode(c[0])
            s2 = summary.getSupernode(c[1])

            '''
            Simulate merge and compute reconstruction error
            '''
            reconstructionError = 0

            #Create new supernode
            supernodeS3 = supernode.Supernode(summary.numSupernodes)

            ###Compute internal reconstruction error###

            #Set internal_edges[supernodeS3] = internal_edges[supernodeS1] + internal_edges[supernodeS3] + edges between supernodeS1 and supernodeS3
            supernodeS3.incrementInternalEdges(s1.getInternalEdges())
            supernodeS3.incrementInternalEdges(s2.getInternalEdges())
            supernodeS3.incrementInternalEdges(s1.getWeight(s2.id))

            #Total number of components in supernode S3
            supernodeS3TotNodes = s1.cardinality + s2.cardinality

            #Internal adjacency of S3
            internal_adj = (2*supernodeS3.getInternalEdges())/(supernodeS3TotNodes*(supernodeS3TotNodes-1))

            #For each possible couple of S3 components, we update the reconstruction error
            #with respect to the ground truth
            for c1 in s1.getComponents():
                for c2 in s2.getComponents():
                    trueAdjacency = graph.getAdjacency(c1, c2)
                    reconstructionError += abs((trueAdjacency)-(internal_adj))

            ###end compute internal reconstruction error###

            ###Compute external reconstruction error###

            #Copy all the neighbours of S1 (different from S2) to S3
            for n in s1.getConnections():
                if(n != s2.id):
                    w = s1.getWeight(n)
                    supernodeS3.addNeighbor(n, w)

            #Copy all the neighbours of S2 (different from S2) to S3: if a neighbour n has been already added in the previous step, increment the weight by 1
            for n in s2.getConnections():
                if(n != s1.id):
                    w = s2.getWeight(n)
                    supernodeS3.updateNeighbor(n, w)
            
            #For each neighbour of s3 update the reconstruction error
            for n in supernodeS3.getConnections():
                
                supernodeN = summary.getSupernode(n)

                #External adjacency between nodes in S3 and nodes in its neighbour n
                external_adj = supernodeS3.getWeight(n)/(supernodeS3TotNodes*supernodeN.cardinality)

                #For each possible couple between S3 and components and n components, we update the reconstruction error
                #with respect to the ground truth
                for c1 in supernodeN.getComponents():
                    for c2 in s1.getComponents():
                        trueAdjacency = graph.getAdjacency(c1, c2)
                        reconstructionError += abs((trueAdjacency)-(external_adj))

                    for c2 in s2.getComponents():
                        trueAdjacency = graph.getAdjacency(c1, c2)
                        reconstructionError += abs((trueAdjacency)-(external_adj))                

            ###end compute external reconstruction error###

            if reconstructionError<minReIncrement:
                minReIncrement = reconstructionError
                bestS1 = s1.getId()
                bestS2 = s2.getId()

            '''
            end simulate merge
            '''

        ###Merge the couple which leads to the smaller Re increment
        summary.merge(bestS1, bestS2)

        #Decrease number of supernode
        supernodeCounter -=1

def kgs_sample_pairs_proportional(graph, summary, k, c_const):
    '''
    Given an input graph G(V,E) and integer k, find a summary graph S
    for G with at most k supernodes V (|V| <= k),
    such that the Re(G|S) is minimized.
    '''

    '''
    In this implementation of the SamplePairs algorithm, at every iteration
    t we may sample pairs P(t) with cardinality proportional to the number
    of supernodes n(t) in the graph summary at iteration t:
    |P(t)| = c_const*(n(t)).
    '''

    supernodeCounter = len(summary.superList)

    while(supernodeCounter>k):
        #Check all the possible couples of supernodes to find the couple to be merged
        minReIncrement = float('inf')

        possibleSuperCouple = np.array(list(combinations(summary.superList.keys(), 2)))

        #Sample c_const*(n(t)) couples at random with uniform probability
        num_pair = int(c_const*len(summary.superList.keys()))
        if num_pair < possibleSuperCouple.size: #if the number of possible couples is less than the number of pairs we want to pick, we have to take into account all the possible pairs
            possibleSuperCouple = possibleSuperCouple[np.random.choice(len(possibleSuperCouple), num_pair)]

        for c in possibleSuperCouple:
            s1 = summary.getSupernode(c[0])
            s2 = summary.getSupernode(c[1])

            '''
            Simulate merge and compute reconstruction error
            '''
            reconstructionError = 0

            #Create new supernode
            supernodeS3 = supernode.Supernode(summary.numSupernodes)

            ###Compute internal reconstruction error###

            #Set internal_edges[supernodeS3] = internal_edges[supernodeS1] + internal_edges[supernodeS3] + edges between supernodeS1 and supernodeS3
            supernodeS3.incrementInternalEdges(s1.getInternalEdges())
            supernodeS3.incrementInternalEdges(s2.getInternalEdges())
            supernodeS3.incrementInternalEdges(s1.getWeight(s2.id))

            #Total number of components in supernode S3
            supernodeS3TotNodes = s1.cardinality + s2.cardinality

            #Internal adjacency of S3
            internal_adj = (2*supernodeS3.getInternalEdges())/(supernodeS3TotNodes*(supernodeS3TotNodes-1))

            #For each possible couple of S3 components, we update the reconstruction error
            #with respect to the ground truth
            for c1 in s1.getComponents():
                for c2 in s2.getComponents():
                    trueAdjacency = graph.getAdjacency(c1, c2)
                    reconstructionError += abs((trueAdjacency)-(internal_adj))

            ###end compute internal reconstruction error###

            ###Compute external reconstruction error###

            #Copy all the neighbours of S1 (different from S2) to S3
            for n in s1.getConnections():
                if(n != s2.id):
                    w = s1.getWeight(n)
                    supernodeS3.addNeighbor(n, w)

            #Copy all the neighbours of S2 (different from S2) to S3: if a neighbour n has been already added in the previous step, increment the weight by 1
            for n in s2.getConnections():
                if(n != s1.id):
                    w = s2.getWeight(n)
                    supernodeS3.updateNeighbor(n, w)
            
            #For each neighbour of s3 update the reconstruction error
            for n in supernodeS3.getConnections():
                
                supernodeN = summary.getSupernode(n)

                #External adjacency between nodes in S3 and nodes in its neighbour n
                external_adj = supernodeS3.getWeight(n)/(supernodeS3TotNodes*supernodeN.cardinality)

                #For each possible couple between S3 and components and n components, we update the reconstruction error
                #with respect to the ground truth
                for c1 in supernodeN.getComponents():
                    for c2 in s1.getComponents():
                        trueAdjacency = graph.getAdjacency(c1, c2)
                        reconstructionError += abs((trueAdjacency)-(external_adj))

                    for c2 in s2.getComponents():
                        trueAdjacency = graph.getAdjacency(c1, c2)
                        reconstructionError += abs((trueAdjacency)-(external_adj))                

            ###end compute external reconstruction error###

            if reconstructionError<minReIncrement:
                minReIncrement = reconstructionError
                bestS1 = s1.getId()
                bestS2 = s2.getId()

            '''
            end simulate merge
            '''

        ###Merge the couple which leads to the smaller Re increment
        summary.merge(bestS1, bestS2)
        
        #Decrease number of supernode
        supernodeCounter -=1

def kgs_linear_check(graph, summary, k):
    '''
    Given an input graph G(V,E) and integer k, find a summary graph S
    for G with at most k supernodes V (|V| <= k),
    such that the Re(G|S) is minimized.
    '''
    supernodeCounter = len(summary.superList)

    while(supernodeCounter>k):
        #Check all the possible couples of supernodes to find the couple to be merged
        minReIncrement = float('inf')

        #Pick uniformly a single supernode s1
        superListKeys = list(summary.superList.keys())
        randomKey = random.choice(superListKeys)
        s1 = summary.getSupernode(randomKey)
        
        possibleSuperCouple = list()
        for supernodeId in superListKeys:
            if supernodeId!=randomKey:
                possibleSuperCouple.append(supernodeId)

        for c in possibleSuperCouple:
            s2 = summary.getSupernode(c)

            '''
            Simulate merge and compute reconstruction error
            '''
            reconstructionError = 0

            #Create new supernode
            supernodeS3 = supernode.Supernode(summary.numSupernodes)

            ###Compute internal reconstruction error###

            #Set internal_edges[supernodeS3] = internal_edges[supernodeS1] + internal_edges[supernodeS3] + edges between supernodeS1 and supernodeS3
            supernodeS3.incrementInternalEdges(s1.getInternalEdges())
            supernodeS3.incrementInternalEdges(s2.getInternalEdges())
            supernodeS3.incrementInternalEdges(s1.getWeight(s2.id))

            #Total number of components in supernode S3
            supernodeS3TotNodes = s1.cardinality + s2.cardinality

            #Internal adjacency of S3
            internal_adj = (2*supernodeS3.getInternalEdges())/(supernodeS3TotNodes*(supernodeS3TotNodes-1))

            #For each possible couple of S3 components, we update the reconstruction error
            #with respect to the ground truth
            for c1 in s1.getComponents():
                for c2 in s2.getComponents():
                    trueAdjacency = graph.getAdjacency(c1, c2)
                    reconstructionError += abs((trueAdjacency)-(internal_adj))

            ###end compute internal reconstruction error###

            ###Compute external reconstruction error###

            #Copy all the neighbours of S1 (different from S2) to S3
            for n in s1.getConnections():
                if(n != s2.id):
                    w = s1.getWeight(n)
                    supernodeS3.addNeighbor(n, w)

            #Copy all the neighbours of S2 (different from S2) to S3: if a neighbour n has been already added in the previous step, increment the weight by 1
            for n in s2.getConnections():
                if(n != s1.id):
                    w = s2.getWeight(n)
                    supernodeS3.updateNeighbor(n, w)
            
            #For each neighbour of s3 update the reconstruction error
            for n in supernodeS3.getConnections():
                
                supernodeN = summary.getSupernode(n)

                #External adjacency between nodes in S3 and nodes in its neighbour n
                external_adj = supernodeS3.getWeight(n)/(supernodeS3TotNodes*supernodeN.cardinality)

                #For each possible couple between S3 and components and n components, we update the reconstruction error
                #with respect to the ground truth
                for c1 in supernodeN.getComponents():
                    for c2 in s1.getComponents():
                        trueAdjacency = graph.getAdjacency(c1, c2)
                        reconstructionError += abs((trueAdjacency)-(external_adj))

                    for c2 in s2.getComponents():
                        trueAdjacency = graph.getAdjacency(c1, c2)
                        reconstructionError += abs((trueAdjacency)-(external_adj))                

            ###end compute external reconstruction error###

            if reconstructionError<minReIncrement:
                minReIncrement = reconstructionError
                bestS1 = s1.getId()
                bestS2 = s2.getId()

            '''
            end simulate merge
            '''


        ###Merge the couple which leads to the smaller Re increment
        summary.merge(bestS1, bestS2)
        
        #Decrease number of supernode
        supernodeCounter -=1