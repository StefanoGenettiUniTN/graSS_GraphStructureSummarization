import summary
import graph
import supernode
from itertools import combinations
import numpy as np
import random
import math

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
            supernodeS3ComponentSet = s1.getComponents().union(s2.getComponents())
            possibleS3ComponentCouple = list(combinations(supernodeS3ComponentSet, 2))
            
            for s3_couple in possibleS3ComponentCouple:
                c1 = s3_couple[0]
                c2 = s3_couple[1]
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
            supernodeS3ComponentSet = s1.getComponents().union(s2.getComponents())
            possibleS3ComponentCouple = list(combinations(supernodeS3ComponentSet, 2))
            
            for s3_couple in possibleS3ComponentCouple:
                c1 = s3_couple[0]
                c2 = s3_couple[1]
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
            supernodeS3ComponentSet = s1.getComponents().union(s2.getComponents())
            possibleS3ComponentCouple = list(combinations(supernodeS3ComponentSet, 2))
            
            for s3_couple in possibleS3ComponentCouple:
                c1 = s3_couple[0]
                c2 = s3_couple[1]
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
            supernodeS3ComponentSet = s1.getComponents().union(s2.getComponents())
            possibleS3ComponentCouple = list(combinations(supernodeS3ComponentSet, 2))
            
            for s3_couple in possibleS3ComponentCouple:
                c1 = s3_couple[0]
                c2 = s3_couple[1]
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

def gs(graph, summary):
    '''
    Given an input graph G(V,E) with adjacency matrix A, find a summary S of this
    graph with expected adjacency matrix \bar{A} such that the total number of bits

        Tb(\bar{A}) = B(\bar{A})+B(A|\bar{A})

    is minimized.
    '''

    #Number of supernodes
    k = len(summary.superList)

    #Number of vertices original graph
    n = graph.numVertices
    
    #initial Tb
    tb = computeTb(graph, summary, n, k)

    #stopping condition: there is no merging of supernodes that can improve
    #the objective function Tb
    stopping_condition = False

    iterationCounter = 0
    while stopping_condition==False:
        #Check all the possible couples of supernodes to find the couple to be merged
        best_tb = tb
        stopping_condition = True
        possibleSuperCouple = list(combinations(summary.superList.keys(), 2))
        
        iterationCounter += 1

        for c in possibleSuperCouple:
            s1 = summary.getSupernode(c[0])
            s2 = summary.getSupernode(c[1])

            '''
            Simulate merge and compute the objective function Tb
            '''
            
            tb_tmp = tb

            # STEP 1: update B(\bar{A})
            previousBa = (graph.numVertices+pow(k, 2)+k-1) * math.log(graph.numVertices, 2)
            currentBa = (graph.numVertices+pow(k-1, 2)+k-2) * math.log(graph.numVertices, 2)
            tb_tmp -= previousBa
            tb_tmp += currentBa
            # ...end step 1

            # STEP 2: update B(A|\bar{A})

            #Create new supernode
            supernodeS3 = supernode.Supernode(summary.numSupernodes)

            #Set internal_edges[supernodeS3] = internal_edges[supernodeS1] + internal_edges[supernodeS3] + edges between supernodeS1 and supernodeS3
            supernodeS3.incrementInternalEdges(s1.getInternalEdges())
            supernodeS3.incrementInternalEdges(s2.getInternalEdges())
            supernodeS3.incrementInternalEdges(s1.getWeight(s2.id))

            #Total number of components in supernode S3
            supernodeS3TotNodes = s1.cardinality + s2.cardinality

            #Internal adjacency of S3
            internal_adj = (2*supernodeS3.getInternalEdges())/(supernodeS3TotNodes*(supernodeS3TotNodes-1))

            #For each possible couple of S3 components, we update the B(A | \bar{A})
            supernodeS3ComponentSet = s1.getComponents().union(s2.getComponents())
            possibleS3ComponentCouple = list(combinations(supernodeS3ComponentSet, 2))
            
            for s3_couple in possibleS3ComponentCouple:
                
                c1 = s3_couple[0]
                c2 = s3_couple[1]

                #Compute previous contribution of c1 c2 in the computation
                #of B(A | \bar{A})
                previousBa_a = computeBaa_ij(graph, summary, c1, c2)

                #Compute current contribution of c1 c2 in the computation
                #of B(A | \bar{A})
                if internal_adj==0 or internal_adj==1:
                    currentBa_a = 0
                else:
                    adj_c1_c2 = graph.getAdjacency(c1, c2)
                    currentBa_a = -adj_c1_c2 * math.log(internal_adj, 2)-(1-adj_c1_c2) * math.log((1-internal_adj),2) 

                tb_tmp -= previousBa_a
                tb_tmp += currentBa_a

            #Copy all the neighbours of S1 (different from S2) to S3
            for n in s1.getConnections():
                if n != s2.id:
                    w = s1.getWeight(n)
                    supernodeS3.addNeighbor(n, w)

            #Copy all the neighbours of S2 (different from S2) to S3: if a neighbour n has been already added in the previous step, increment the weight by 1
            for n in s2.getConnections():
                if n != s1.id:
                    w = s2.getWeight(n)
                    supernodeS3.updateNeighbor(n, w)
            
            #For each neighbour of s3 update update the B(A | \bar{A})
            for n in supernodeS3.getConnections():
                
                supernodeN = summary.getSupernode(n)

                #External adjacency between nodes in S3 and nodes in its neighbour n
                external_adj = supernodeS3.getWeight(n)/(supernodeS3TotNodes*supernodeN.cardinality)

                #For each possible couple between S3 and components and n components,
                #update the B(A | \bar{A})
                for c1 in supernodeN.getComponents():
                    for c2 in s1.getComponents():
                        #Compute previous contribution of c1 c2 in the computation
                        #of B(A | \bar{A})
                        previousBa_a = computeBaa_ij(graph, summary, c1, c2)

                        #Compute current contribution of c1 c2 in the computation
                        #of B(A | \bar{A})
                        if external_adj==0 or external_adj==1:
                            currentBa_a = 0
                        else:
                            adj_c1_c2 = graph.getAdjacency(c1, c2)
                            currentBa_a = -adj_c1_c2 * math.log(external_adj, 2)-(1-adj_c1_c2) * math.log((1-external_adj),2) 

                        tb_tmp -= previousBa_a
                        tb_tmp += currentBa_a

                    for c2 in s2.getComponents():
                        #Compute previous contribution of c1 c2 in the computation
                        #of B(A | \bar{A})
                        previousBa_a = computeBaa_ij(graph, summary, c1, c2)

                        #Compute current contribution of c1 c2 in the computation
                        #of B(A | \bar{A})
                        if external_adj==0 or external_adj==1:
                            currentBa_a = 0
                        else:
                            adj_c1_c2 = graph.getAdjacency(c1, c2)
                            currentBa_a = -adj_c1_c2 * math.log(external_adj, 2)-(1-adj_c1_c2) * math.log((1-external_adj),2) 

                        tb_tmp -= previousBa_a
                        tb_tmp += currentBa_a              

            if tb_tmp<best_tb:
                best_tb = tb_tmp
                bestS1 = s1.getId()
                bestS2 = s2.getId()
                stopping_condition = False

            '''
            end simulate merge
            '''

        if stopping_condition == False:
            ###Merge the couple which leads to the smaller Re increment
            summary.merge(bestS1, bestS2)
            
            #Decrease number of supernode
            k -= 1

            #Update tb
            #tb = computeTb(graph, summary, graph.numVertices, k)
            tb = best_tb

def computeTb(graph, summary, n, k):
    '''
    Auxiliary function for gs: computation of the objective function
    '''

    #Compute B(a | bar{A})
    ba_a = 0

    possibleCouple = list(combinations(graph.vertList.keys(), 2))
    
    for c in possibleCouple:
        
        #Graph components
        c1 = c[0]
        c2 = c[1]

        #Corresponding supernodes
        s1 = summary.getComponentSupernode(c1)
        s2 = summary.getComponentSupernode(c2)

        s1_cardinality = s1.cardinality
        s2_cardinality = s2.cardinality

        if s1.getId() == s2.getId():
            #Same supernode
            s1_internalEdges = s1.getInternalEdges()
            expected_adj_c1_c2 = (2*s1_internalEdges)/(s1_cardinality*(s1_cardinality-1))

        else:
            #Distinct supernode
            s1_s2_edges = s1.getWeight(s2.getId())
            expected_adj_c1_c2 = (s1_s2_edges)/(s1_cardinality*s2_cardinality)

        if expected_adj_c1_c2==0 or expected_adj_c1_c2==1:
            ba_a += 0
        else:
            adj_c1_c2 = graph.getAdjacency(c1, c2)
            ba_a += -adj_c1_c2 * math.log(expected_adj_c1_c2, 2)-(1-adj_c1_c2) * math.log((1-expected_adj_c1_c2),2)

    #...end compute B(A | bar{A})

    #Compute B(\bar{A})
    ba = (n+pow(k, 2)+k-1) * math.log(n, 2)

    #compute Tb
    tb = ba + ba_a

    return tb

def computeBaa_ij(graph, summary, i, j):
    '''
    Auxiliary function for gs: computation of the contribution of the couple
    i, j in the computation of the number of bits required to describe A given \bar{A}
    '''

    #Compute B(a | bar{A})
    ba_a = 0
        
    #Graph components
    c1 = i
    c2 = j

    #Corresponding supernodes
    s1 = summary.getComponentSupernode(c1)
    s2 = summary.getComponentSupernode(c2)

    s1_cardinality = s1.cardinality
    s2_cardinality = s2.cardinality

    if s1.getId() == s2.getId():
        #Same supernode
        s1_internalEdges = s1.getInternalEdges()
        expected_adj_c1_c2 = (2*s1_internalEdges)/(s1_cardinality*(s1_cardinality-1))

    else:
        #Distinct supernode
        s1_s2_edges = s1.getWeight(s2.getId())
        expected_adj_c1_c2 = (s1_s2_edges)/(s1_cardinality*s2_cardinality)

    if expected_adj_c1_c2==0 or expected_adj_c1_c2==1:
        ba_a += 0
    else:
        adj_c1_c2 = graph.getAdjacency(c1, c2)
        ba_a += -adj_c1_c2 * math.log(expected_adj_c1_c2, 2)-(1-adj_c1_c2) * math.log((1-expected_adj_c1_c2),2)

    #...end compute B(A | bar{A})

    return ba_a

def kcgs_condense(graph, summary, k):
    '''
    Given an input graph G(V,E) and integer k,
    find a summary graph S for G with supernodes
    V such that Re(G|S) is minimized and for every
    V' \in V we have that |V'| >= k.
    '''

    if k<=1:
        print(f"Error kcgs_condense: parameter k cannot be less than 1")
        return

    #set of original nodes that have not yet been
    #assigned to a supernode
    singletonNodes = set()
    for v in graph.getVertices():
        singletonNodes.add(v)

    #Create supernodes with size exactly k
    #until fewer than k nodes remain without
    #being assigned to a supernode
    while len(singletonNodes)>=k:
        #randomly select a node from singletonNodes
        v = random.choice(tuple(singletonNodes))
        s1 = summary.getComponentSupernode(v)
        ballId = s1.getId()
        singletonNodes.remove(v)

        #form a "ball" around v as described in the paper
        for i in range(k-1):
            minReIncrement = float('inf')

            for u in singletonNodes:
                s1 = summary.getSupernode(ballId)
                s2 = summary.getComponentSupernode(u)
                
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
                supernodeS3ComponentSet = s1.getComponents().union(s2.getComponents())
                possibleS3ComponentCouple = list(combinations(supernodeS3ComponentSet, 2))
                
                for s3_couple in possibleS3ComponentCouple:
                    c1 = s3_couple[0]
                    c2 = s3_couple[1]
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
                    bestS2 = s2.getId()
                    bestS2Component = u

                '''
                end simulate merge
                '''


            ###Merge the couple which leads to the smaller Re increment
            ballId = summary.merge(ballId, bestS2)

            ###Remove bestS2 from the set of original nodes that have not
            #yet been assigned to a supernode
            singletonNodes.remove(bestS2Component)

    
    #if in the last iteration fewen than k nodes remain without
    #being assigned to a supernode, we assign them to already
    #existing supernodes
    for c in singletonNodes:
        s1 = summary.getComponentSupernode(c)
        
        minReIncrement = float('inf')

        for u in summary.superList.keys():
            s2 = summary.getSupernode(u)

            if(s2.cardinality>1):
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
                supernodeS3ComponentSet = s1.getComponents().union(s2.getComponents())
                possibleS3ComponentCouple = list(combinations(supernodeS3ComponentSet, 2))
            
                for s3_couple in possibleS3ComponentCouple:
                    
                    c1 = s3_couple[0]
                    c2 = s3_couple[1]
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
                    bestS2 = s2.getId()

                '''
                end simulate merge
                '''


        ###Merge the couple which leads to the smaller Re increment
        summary.merge(s1.getId(), bestS2)
            

