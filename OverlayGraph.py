import pickle
from sage.all import *
import config as cfg
from logger import applogger
import itertools
from multiprocessing.dummy import Pool as ThreadPool
from heapq import  heappop, heappush
import time

#OverlayGraph maintains the k layers of Overlay graphs where _graphLayer[n] is the nth overlay layer
"""
Overlay Graph is a condensed form(lesser vertices of your input roadmap.
This helps you to run your dijsktra queries on this condensed graph rather than a real road map
with possibly millions of node.

The class here is a heirarchy of overlay Graph where each layer is an overlay graph.
At each layer, the number of vertices in the graph reduces, however the edge could also increase
since there is an edge in the overlay graph corresponding to every actual route in the road map,
between the vertices of the overlay graph.

Your basegraph could contain enormous information (number of traffic lights, travel time, distance, turn costs) etc that you may
not want the user to query by.

For all the queriable metrices, add a corresponding entry in "metrics" field in the config file to make it queriable.

Overlay graph at level I is constructed by considering only the vertex covers of the graph at level (i-1)
and adding an edge corresponding to every route between these vertex covers in the immediate lower layer.

This leads to a formation of 2^k-All Path Cover of the base graph, where k is the number of layers


For proof and additional informations :
1. http://www.vldb.org/pvldb/vol7/p893-funke.pdf
2. https://dl.acm.org/citation.cfm?id=2983712
"""

def timeit(f):
    def timed(self,*args, **kw):
        ts = time.time()
        result = f(self,*args, **kw)
        te = time.time()
        if not self._benchmark.get(f.__name__) :
            self._benchmark[f.__name__] = []
        self._benchmark[f.__name__].append(round((te -ts)*1000,1))
        return result
    return timed

class OverlayGraph():
    def __init__(self):
        #_graphLayer[n] is the nth overlay layer
        #_coveredVertices[n] is the set of vertices in _graphLayer[n]
        self._graphLayer = []
        self._coveredVertices = []
        self._metrics = cfg.metrics
        self._numberOfLayers = None
        self._benchmark = {}
        try :
            #Check if we already have created an overlay graph before. The program can load the file and initialize instantly without
            #re-creating overlay graphs from the scratch
            overlayGraphFile = cfg.filelocation["OVERLAYGRAPH"]
            self._graphLayer = load(overlayGraphFile)
            self._coveredVertices = map(lambda diGraphAtlayer : set(diGraphAtlayer.vertices()), self._graphLayer)
            self._numberOfLayers = len(self._graphLayer)
            applogger.debug("Number of layers in loaded graph = %s", self._numberOfLayers)
        except IOError:
            #This occures if a pre-existing overlay graph file doesnt exist on disk. We need to create an overlay graph and store it so that future runs
            #dont have to do this again.
            applogger.info("Creating overlaygraph for the first time. This might take a while")
            self._createOverLayGraph()
            self._writeOverLayGraphToFile()

    def _plotLayer(self, layerNum, saveLocation) :
        if layerNum >= self._numberOfLayers or layerNum < 0:
            applogger.error("Layer doesnt exist")
        else :
            save(self._graphLayer[layerNum].plot(edge_labels=False), saveLocation+".png",axes=False)#,aspect_ratio=True)

    def _plotAllLayer(self, saveLocation) :
        for layerNum in xrange(0, self._numberOfLayers):
            self._plotLayer(layerNum, saveLocation+str(layerNum))


    def _writeOverLayGraphToFile(self) :
        graphLayerList = []

        try :
            overlayGraphFile = cfg.filelocation["OVERLAYGRAPH"]
            applogger.debug("Writing overlayGraph to file")
            save(self._graphLayer, overlayGraphFile)
        except IOError:
            applogger.error("Error while writing overlayGraphFile")

    #Helper function to get the outgoing edges(including parallel edges) of a vertex and extract path information from them.
    def _getAllNeighbourDetails(self, vertex, layerNum) :
        for (_, neighbourVertex, edgeLabel) in self._graphLayer[layerNum].edges_incident(vertex):
            if neighbourVertex != vertex : #If there is a loop, ignore it
            #Overlay Graph at layer i is a condensed graph of layer i-1.
            #Graph (A->C) at layer i have been could have been created out of graph(A->B->C) in layer(i-1)
            #So along the edge A->C, we need to store the actual intermediate nodes along the route, so that we can
            #use this information to display the actual route during query stage.
                intermediateVertices = edgeLabel.get("intermediateVertices", [])
                edgeMetrics = {}
                edgeMetrics["intermediateVertices"] = intermediateVertices
                for metric in self._metrics :
                    edgeMetrics[metric] = edgeLabel[metric]
                yield (neighbourVertex, edgeMetrics)

    #Given a vertex v (included in the cover), this function creates an edge for every path from v to any other vertex that is also included
    #in the cover.
    def createEdges(self, vertex, layerNum) :
        for (neighbourVertex, edgeMetricsBetweenSourceAndNeighbour) in self._getAllNeighbourDetails(vertex, layerNum - 1):
            #Graph (A->B->C). Imagine both A as well as B are included in the vertex cover(not optimum)
            #In the overlay graph, we add a edge from A to B and retain the intermediateVertices between A->B as it is in the lower layer
            if neighbourVertex in self._coveredVertices[layerNum]:
             #   edgeLabel = {"intermediateVertices":intermediateVerticesBetweenSourceAndNeighbour}
                self.overlayLayer.add_edge(vertex, neighbourVertex, edgeMetricsBetweenSourceAndNeighbour)
                self.overlayTopology.add_edge(vertex, neighbourVertex)
            else :
                ##If immediate neighbour isnt in the vertex cover, the neighbour of neighbour is definitely in the vertex cover
                #or else the vertex cover property is violated.
                #Imagine Graph (A-B-C), A and C are in the vertex cover, so we add an edge from A to C in the overlay Graph
                #However the new edge's intermediateVertices are (intermediateVertices from A to B + Node B + intermediateVertices from B to C)
                for (neighbourOfNeighbourVertex, edgeMetricsBetweenNeighbours) in self._getAllNeighbourDetails(neighbourVertex, layerNum - 1) :
                    #Graph (A-B-A) could created an infinite loop. So we check if the neighbour of neighbour isnt the source vertex itself
                    if neighbourOfNeighbourVertex != vertex:
                        intermediateVerticesBetweenSourceAndNeighbour = edgeMetricsBetweenSourceAndNeighbour["intermediateVertices"]
                        intermediateVerticesBetweenNeighbours = edgeMetricsBetweenNeighbours["intermediateVertices"]
                        #Experiemntal. If they have a common intermediate node, useless path
                        if not set(intermediateVerticesBetweenSourceAndNeighbour).isdisjoint(intermediateVerticesBetweenNeighbours):
                            continue
                        newEdgeMetrics = { k: edgeMetricsBetweenSourceAndNeighbour[k] + edgeMetricsBetweenNeighbours[k] for k in set(edgeMetricsBetweenSourceAndNeighbour) if k != "intermediateVertices"}
                        newIntermediateVertices = []
                        newIntermediateVertices.extend(intermediateVerticesBetweenSourceAndNeighbour)
                        newIntermediateVertices.append(neighbourVertex)
                        newIntermediateVertices.extend(intermediateVerticesBetweenNeighbours)
                        newEdgeMetrics["intermediateVertices"] = newIntermediateVertices
                        self.overlayLayer.add_edge(vertex, neighbourOfNeighbourVertex, newEdgeMetrics)
                        self.overlayTopology.add_edge(vertex, neighbourOfNeighbourVertex)
    #This function creates an overlay graph by considering every route from one covered vertex to another covered vertex in the base graph
    def _createAllPathLayer(self, layerNum) :
        self.overlayLayer = DiGraph(multiedges=True)
        self.overlayTopology = Graph()
        pool = None
        #Edge creation between nodes on the overlay graph can be parallelized. Hence using a threadpool.
        #https://stackoverflow.com/questions/2846653/how-to-use-threading-in-python
        pool = ThreadPool()
        pool.map(lambda v : self.createEdges(v,layerNum), self._coveredVertices[layerNum])
        pool.close()
        pool.join()
        #At this point overlayLayer at level i is complete. That is all routes between vertex covers of layer (i-1) is connected via an edge
        return self.overlayLayer

    def _createShortestPathlayer(self, layerNum) :
        graphLayer = DiGraph()
        for coveredVertex in self._coveredVertices[layerNum] :
            (visited, accessNodes) = self._createBestEdges(coveredVertex, layerNum)
            #print accessNodes
            for accessnode in accessNodes :
                path,_ = self._getNodesAlongShortestPath(accessnode, visited, False)
                path.reverse()
                edgeMetrics = {'dist':visited.get(accessnode).get("cost"), 'intermediateVertices' : path[1:]}
                graphLayer.add_edge(coveredVertex, accessnode, edgeMetrics)
        return graphLayer

    def _createBestEdges(self, sourceVertex, layerNum):
        userWieghts = {'dist' : 1}
        visited={}
        accessNodes = set()
        queue = [] #priority queue
        #push to Priority Q values (cost, (currentVertex, parentOfCurrentVertex))
        #Cost is set to 0 for the source vertex so that it is popped first
        heappush(queue, (0,(sourceVertex, None)))
        #Function _runPartialDijsktra is commonly used by Forward and backward Dijsktra.
        #Using the forwardDirection boolean switch, we decide if we consider the outgoing(for forward) edges or
        #incoming(for backward) edges
        directedNeighbors = self._graphLayer[0].neighbors_out
        while queue:
            path_cost, (v, parent) = heappop(queue)
            if visited.get(v) is None: # v is unvisited
                visited[v] = {"cost":path_cost, "parent": parent}
                #If v is not an accessnode
                if parent not in self._coveredVertices[layerNum] or parent == sourceVertex:
                    for neighbourVertex in directedNeighbors(v):
                        if visited.get(neighbourVertex) is None:
                            edges = self._graphLayer[0].edge_label(v , neighbourVertex)
                            edge_cost, _ = self._getMinimumWeightedEdge(edges, userWieghts)
                            #push to Q the new cost to neighbour and set myself as its parent
                            heappush(queue, (path_cost + edge_cost, (neighbourVertex, v)))
                else :
                    accessNodes.add(parent)
        return (visited, accessNodes)



    def _getVertexCoverOflayer(self, layerNum):
        if self.overlayTopology is None:
            undirectedGraph = self._graphLayer[layerNum]
        else :
            applogger.debug("Loaded topology map")
            undirectedGraph = self.overlayTopology
        vertexAndDegreeTupleList = [vertexAndDegreeTuple for vertexAndDegreeTuple in undirectedGraph.degree_iterator(labels=True)]
        #Sort considering the order of the second key in the tuple ie degree, So that we have
        #this list sorted by ascending order of degree
        vertexAndDegreeTupleList.sort(key = lambda tup:tup[1])
        vertexCover = set()
        for vertex,_ in vertexAndDegreeTupleList:
            if vertex not in vertexCover:
                vertexCover.update(undirectedGraph.neighbors(vertex))
        return vertexCover

    #This function Creates the ith layer of the graph using the (i-1)th layer
    @timeit
    def _createLayer(self, layerNum):
        #Find the vertex cover of the graph at the lower layer. Vertex cover works only on for undirected graph in Sage
        #So, converting the lower layer into a undirected graph and also removing the multiple edges in the lower graph
        applogger.debug("Starting vertex cover of layer %s"%(layerNum-1))
        self._coveredVertices[layerNum] = self._getVertexCoverOflayer(layerNum - 1)
        applogger.debug("Retrieved vertex cover of layer %s"%(layerNum-1))
        applogger.debug("VC Size = %s"%len(self._coveredVertices[layerNum]))
        #This is a switch in-case I decide to store only the shortest path later instead of all routes from one node to another
        if cfg.ALLPATH:
            self._graphLayer[layerNum] = self._createAllPathLayer(layerNum)
        else :
            self._graphLayer[layerNum] = self._createShortestPathlayer(layerNum)

    def _createOverLayGraph(self):
        try :
            #BaseGraph is the raw graph (road network).
            baseGraphFile = cfg.filelocation["BASEGRAPH"]
            self._numberOfLayers = cfg.numberOfLayers
            #initialize graphLayer . ie start with every layer as an empty list. Later on graphLayer[n] will be the nth layer of the overlay graph
            self._graphLayer = [None for k in xrange(self._numberOfLayers)]
            self._coveredVertices = [None for k in xrange(self._numberOfLayers)]


            #Initialize the first layer as the raw graph data
            self._graphLayer[0] = load(baseGraphFile)
            applogger.debug("Loaded baseGraph")
            #Initialize the cover in the first layer as all the vertices of the raw graph
            self._coveredVertices[0] = set(self._graphLayer[0].vertices())
            self.overlayTopology = None
            applogger.debug("Retrieved baseGraph vertices")
            applogger.debug("Layer %s Generated. Number of vertices = %s, Number of edges = %s"%(0, self._graphLayer[0].order(), self._graphLayer[0].size()))
            for layerNum in xrange(1, self._numberOfLayers) :
                applogger.debug("Generating Layer %s"%layerNum)
                self._createLayer(layerNum)
                applogger.debug("Layer %s Generated. Number of vertices = %s, Number of edges = %s"%(layerNum, self._graphLayer[layerNum].order(), self._graphLayer[layerNum].size()))
            applogger.debug("OverlayGraph creation complete")
        except IOError :
            print "No base file located"
            import sys
            sys.exit()
    #To run, Dijsktra on the overlay graph, we initialize the priority Q with every access node obtained in the forward
    #search and its cost from source. By the end of the Dijstra in Overlay graph, we get the cost from source to every
    #access node of the target
    @timeit
    def _runDijsktraOnOverlay(self,forwardAccessNodePathCost, targetAccessNodes, userWieghts) :
        #print forwardAccessNodePathCost
        #print targetAccessNodes
        queue = []
        for accessNodeOfSource, costToAccessNode in forwardAccessNodePathCost:
            #Initialize the priority Q with all accessNodes
            heappush(queue, ((costToAccessNode, 0),(accessNodeOfSource, None, [])))
        visited = {}
        overlayGraph = self._graphLayer[self._numberOfLayers - 1]
        #We stop when all the targetAccessNodes are settled.
        #Everytime we encounter a target node, remove it from targetAccessNodes.
        #When all of them have been settled, targetAccessNodes will become empty and we
        #can stop the dijsktra
        while queue and targetAccessNodes:
            (path_cost, hops), (vertex, parentOfVertex, edgeIntermediateVerticesToParent) = heappop(queue)
            if visited.get(vertex) is None:
                visited[vertex] = {"cost":path_cost, "parent" : parentOfVertex , "intermediateVerticesToParent" : edgeIntermediateVerticesToParent, "hops" : hops}
                if vertex in targetAccessNodes :
                    targetAccessNodes.remove(vertex)
                for neighbourOverlayNode in overlayGraph.neighbors_out(vertex):
                    if visited.get(neighbourOverlayNode) is None:
                        edges = overlayGraph.edge_label(vertex, neighbourOverlayNode)
                        #In overlay graphs, there could be multiple edges between 2 nodes
                        #Hence we need to retrieve edgeIntermediateVertices, to get the actual
                        #list of intermediate vertices between overlay nodes
                        edge_cost, edgeIntermediateVertices = self._getMinimumWeightedEdge(edges, userWieghts)
                        heappush(queue, ((path_cost + edge_cost, hops+len(edgeIntermediateVertices)), (neighbourOverlayNode, vertex, edgeIntermediateVertices)))
        return visited

    def getWeightedEdgeCost(self, edge, userWieghts) :
        weightedEdgeCost = 0
        for metric in userWieghts :
            weightedEdgeCost += edge.get(metric) * userWieghts[metric]
        return weightedEdgeCost


    #If edges is a single element, return the cost of the edge as per userWieght
    #If edges is a list of edges, returns the cost of the edge with the minimum cost
    def _getMinimumWeightedEdge(self, edges, userWieghts) :
        if type(edges) is list :
            optimumEdge = min(edges, key = lambda d: self.getWeightedEdgeCost(d, userWieghts))
            return (self.getWeightedEdgeCost(optimumEdge, userWieghts), optimumEdge.get("intermediateVertices", []))
        return (self.getWeightedEdgeCost(edges, userWieghts), edges.get("intermediateVertices", []))

    #Runs a Dijsktra algorithm from sourceVertex but without expanding accessNodes.
    #Thus when the priority Q is empty, we will have the optimum path to every access node
    #reachable from sourceVertex.
    @timeit
    def _runPartialDijsktra(self, sourceVertex, userWieghts, forwardDirection):
        numberOfLayers = self._numberOfLayers
        visited={}
        accessNodes = set()
        queue = [] #priority queue
        #push to Priority Q values (cost, (currentVertex, parentOfCurrentVertex))
        #Cost is set to 0 for the source vertex so that it is popped first
        heappush(queue, (0,(sourceVertex, None)))
        #Function _runPartialDijsktra is commonly used by Forward and backward Dijsktra.
        #Using the forwardDirection boolean switch, we decide if we consider the outgoing(for forward) edges or
        #incoming(for backward) edges
        if forwardDirection :
            directedNeighbors = self._graphLayer[0].neighbors_out
        else :
            directedNeighbors = self._graphLayer[0].neighbors_in
        while queue:
            path_cost, (v, parent) = heappop(queue)
            if visited.get(v) is None: # v is unvisited
                visited[v] = {"cost":path_cost, "parent": parent}
                #If v is not an accessnode
                if parent not in self._coveredVertices[numberOfLayers-1] :
                    for neighbourVertex in directedNeighbors(v):
                        if visited.get(neighbourVertex) is None:
                            if forwardDirection:
                                edges = self._graphLayer[0].edge_label(v , neighbourVertex)
                            else :
                                edges = self._graphLayer[0].edge_label(neighbourVertex, v)
                            edge_cost, _ = self._getMinimumWeightedEdge(edges, userWieghts)
                            #push to Q the new cost to neighbour and set myself as its parent
                            heappush(queue, (path_cost + edge_cost, (neighbourVertex, v)))
                else :
                    accessNodes.add(parent)
        return (visited, accessNodes)

    @timeit
    def _runNormalDijsktra(self, sourceVertex, targetVertex, userWieghts):
        applogger.debug("Running normal dijsktra")
        numberOfLayers = self._numberOfLayers
        visited={}
        accessNodes = set()
        queue = [] #priority queue
        #push to Priority Q values (cost, (currentVertex, parentOfCurrentVertex))
        #Cost is set to 0 for the source vertex so that it is popped first
        heappush(queue, (0,(sourceVertex, None)))
        #Function _runPartialDijsktra is commonly used by Forward and backward Dijsktra.
        #Using the forwardDirection boolean switch, we decide if we consider the outgoing(for forward) edges or
        #incoming(for backward) edges
        directedNeighbors = self._graphLayer[0].neighbors_out
        while queue:
            path_cost, (v, parent) = heappop(queue)
            if v == targetVertex :
                #print path_cost, parent
                #return
            if visited.get(v) is None: # v is unvisited
                visited[v] = {"cost":path_cost, "parent": parent}
                #If v is not an accessnode
                for neighbourVertex in directedNeighbors(v):
                    if visited.get(neighbourVertex) is None:
                        edges = self._graphLayer[0].edge_label(v , neighbourVertex)
                        edge_cost, _ = self._getMinimumWeightedEdge(edges, userWieghts)
                        #push to Q the new cost to neighbour and set myself as its parent
                        heappush(queue, (path_cost + edge_cost, (neighbourVertex, v)))
        return visited.get(targetVertex)

    def _getNodesAlongShortestPath(self, startFrom, verticesVisitedDuringSearch, addFirstNode = True):
        currentNode = startFrom
        path = []
        if addFirstNode:
            path.append(currentNode)
        parent = verticesVisitedDuringSearch[currentNode].get("parent")
        while parent :
            path.extend(reversed(verticesVisitedDuringSearch[currentNode].get("intermediateVerticesToParent", [])))
            path.append(parent)
            currentNode = parent
            parent = verticesVisitedDuringSearch[currentNode].get("parent")
        return path, currentNode

    @timeit
    def findOptimumRoute(self, sourceVertex, targetVertex, userWieghts):
        """Given two nodes on the graph, returns the optimum path along them
            @sourceVertex - Starting node
            @targetVertex - Destination node
            @UserWeights - Dictionary of user preference
                eg : {'dist' : 1, 'traffic_lights:0'} will find the optimum path considering shortest distance between source and target
                     {'dist' : 0, 'traffic_lights:1'} will find the optimum path considering least amount of traffic lights between source and target
        """
        args = [(sourceVertex, True), (targetVertex, False)]
        pool = ThreadPool(2)
        #Run a bi-directional dijsktra parallely : A forward Dijsktra from source and a backward dijsktra from the target
        (forwardResult, backwardWardResult) = pool.map(lambda arg : self._runPartialDijsktra(arg[0], userWieghts, arg[1]), args)
        #Sample Forward Result :
        #({1: {'cost': 0, 'parent': None}, 2: {'cost': 1, 'parent': 1}, 3: {'cost': 2, 'parent': 2}, 4: {'cost': 3, 'parent': 3}, 10: {'cost': 1, 'parent': 1}}, set([10, 4]))
        #Sample BackWard Result :
        #({8: {'cost': 0, 'parent': None}, 6: {'cost': 2, 'parent': 7}, 7: {'cost': 1, 'parent': 8}}, set([6]))
        pool.close()
        pool.join()
        verticesVisitedDuringForwardSearch, forwardAccessNodes = forwardResult
        verticesVisitedDuringBackwardSearch, backWardAccessNodes = backwardWardResult
        #print "forwardAccessNodes : " , len(forwardAccessNodes)
        #print "backWardAccessNodes : " , len(backWardAccessNodes)

        #TargetVertex was already settled during the forward search. We have nothing more to do here.
        if targetVertex in verticesVisitedDuringForwardSearch :
            path,_ = self._getNodesAlongShortestPath(targetVertex, verticesVisitedDuringForwardSearch)
            path.reverse()
            jsonDict = {"minCost" : verticesVisitedDuringForwardSearch.get(targetVertex).get("cost"), "route" : path}
            return jsonDict
        #To run, Dijsktra on the overlay graph, we initialize the priority Q with every access node obtained in the forward
        #search and its cost from source. By the end of the Dijstra in Overlay graph, we get the cost from source to every
        #access node of the target
        forwardAccessNodePathCost = [(accessNode, verticesVisitedDuringForwardSearch.get(accessNode).get("cost")) for accessNode in forwardAccessNodes]
        verticesVisitedDuringOverlaySearch = self._runDijsktraOnOverlay(forwardAccessNodePathCost, backWardAccessNodes.copy(), userWieghts)
        #Sample verticesVisitedDuringOverlaySearch :
        #{10: {'cost': 1, 'intermediateVerticesToParent': [], 'parent': None}, 4: {'cost': 3, 'intermediateVerticesToParent': [], 'parent': None}, 6: {'cost': 4, 'intermediateVerticesToParent': [], 'parent': 4}}
        #Pick the target access node that minimizes the cost of (source to access node) + (access node to target)
        distToTargetViaAcessNodes = map(lambda accessNode : (verticesVisitedDuringOverlaySearch[accessNode].get("cost") + verticesVisitedDuringBackwardSearch[accessNode].get("cost"), verticesVisitedDuringOverlaySearch[accessNode].get("hops"), accessNode), backWardAccessNodes)
        #print distToTargetViaAcessNodes
        minimumCost, _, minimumAccessNode = min(distToTargetViaAcessNodes)
        #Here , we create the shortest route from the source to target.
        #For , overlay search, we use the parent heirarchy and the edgeIntermediate vertices to retrieve the actual path
        #print minimumAccessNode
        #print verticesVisitedDuringOverlaySearch[minimumAccessNode]
        pathAlongOverLayGraph,lastNode = self._getNodesAlongShortestPath(minimumAccessNode, verticesVisitedDuringOverlaySearch, True)
        pathAlongForwardSearch,_ = self._getNodesAlongShortestPath(lastNode, verticesVisitedDuringForwardSearch, False)
        pathAlongBackWardSearch,_ = self._getNodesAlongShortestPath(minimumAccessNode, verticesVisitedDuringBackwardSearch, False)
        path = []
        path.extend(pathAlongOverLayGraph)
        path.extend(pathAlongForwardSearch)
        #print list(reversed(pathAlongForwardSearch))
        #print list(reversed(pathAlongOverLayGraph))
        #print pathAlongBackWardSearch
        #The parent heirarchy lists the route in reverse order. So we reverse it to obtain the forward order
        path.reverse()
        #Backward search operated in the backward direction. So the parent heirarchy will already list vertices in the actual forward order
        #So, no need to reverse it
        path.extend(pathAlongBackWardSearch)
        jsonDict = {"minCost":minimumCost, "route" : path}
        return jsonDict

    def _modifyEdgeLabel(self,u,v, layerNum, currentEdgeLabel, delta) :
        modifiedEdgeLabel = currentEdgeLabel.copy()
        for metric in delta :
            modifiedEdgeLabel[metric] = modifiedEdgeLabel.get(metric, 0) + delta.get(metric)
        self._graphLayer[layerNum].delete_edge(u, v, currentEdgeLabel)
        self._graphLayer[layerNum].add_edge(u, v, modifiedEdgeLabel)
        #Sage doesnt support editing in place the edge weight between u,v when there are multiple edges between them
        #So, I delete the old edge and then add the new edge with new weights(to emulate a modification)

    #Recursive internal function to update the edge weight
    #Edge weights need to propogated from the baselayer upto the highest overlay layer
    #So we call this function recursively for each layer
    def _updateHeirarchicalInternal(self, sourceCoveredVertex, targetCoveredVertex, delta, layerNum, changedU, changedV, alreadyDone):
        #We iterate through all routes between two covered vertices , sourceCoveredVertex and targetCoveredVertex
        if alreadyDone.get((sourceCoveredVertex, targetCoveredVertex, layerNum), False):
            applogger.debug("Skip Modifying %s, %s, %s"%(sourceCoveredVertex, targetCoveredVertex, layerNum))
            return
        for _, _, currentEdgeLabel in self._graphLayer[layerNum].edge_boundary([sourceCoveredVertex],[targetCoveredVertex]) :
            intermediateVertices = currentEdgeLabel.get("intermediateVertices", [])
            #if between u and v, there is a direct edge(No intermediate vertices) betweeen them change the cost of the direct edge
            if changedU == sourceCoveredVertex and changedV == targetCoveredVertex and not intermediateVertices :
                self._modifyEdgeLabel(sourceCoveredVertex,targetCoveredVertex, layerNum, currentEdgeLabel, delta)
            elif changedU == sourceCoveredVertex:
                #If sourceVertex(First vertex) of this route is u, the edge uv exist only in this route if the first intermediate node is v.
                #We modify such a route, if it exists
                if intermediateVertices and changedV  == intermediateVertices[0]:
                    self._modifyEdgeLabel(sourceCoveredVertex, targetCoveredVertex, layerNum, currentEdgeLabel, delta)
            elif changedV == targetCoveredVertex :
                #If targetVertex(last vertex) of this route is v, the edge uv exist only in this route if the last intermediate node is u.
                #We modify such a route, if it exists
                if intermediateVertices and changedU  == intermediateVertices[-1]:
                    self._modifyEdgeLabel(sourceCoveredVertex, targetCoveredVertex, layerNum, currentEdgeLabel, delta)
            else :
            #If neither u or v are present at the first and last node, edge uv exists in this route only
            #if they are present side by side along the intermediate path
                for i in xrange(len(intermediateVertices)-1):
                    if intermediateVertices[i] == changedU :
                        if intermediateVertices[i+1] == changedV :
                            self._modifyEdgeLabel(sourceCoveredVertex , targetCoveredVertex, layerNum, currentEdgeLabel, delta)
                        break
        alreadyDone[(sourceCoveredVertex, targetCoveredVertex, layerNum)] = True
        if layerNum >= self._numberOfLayers - 1:
            return
        #In this layer, we have modified the edge weight of a route between sourceCoveredVertex and targetCoveredVertex
        #If both sourceCoveredVertex and targetCoveredVertex are covered vertices in the next layer as well, the modified edge exist
        #between them in the next layer as well.
        if sourceCoveredVertex in self._coveredVertices[layerNum + 1] and targetCoveredVertex in self._coveredVertices[layerNum + 1] :
                self._updateHeirarchicalInternal(sourceCoveredVertex, targetCoveredVertex, delta, layerNum + 1, changedU, changedV, alreadyDone)
        #If targetCoveredVertex in this layer , is not a covered vertex on the next layer, the edge modified at this layer is going to be in one of
        #the routes between sourceCoveredVertex and an outneighbour of targetCoveredVertex.
        #Remember : Since covered vertex at layer i+1 is a vertex cover of layer i, if targetCoveredVertex is not in the layer i+1's cover,
        #one of its neighbour surely is.
        elif sourceCoveredVertex in self._coveredVertices[layerNum + 1] : ##target not in  vc
            for neighbouroftargetCoveredVertex in self._graphLayer[layerNum].neighbors_out(targetCoveredVertex) :
                if neighbouroftargetCoveredVertex != sourceCoveredVertex :
                    self._updateHeirarchicalInternal(sourceCoveredVertex,neighbouroftargetCoveredVertex, delta, layerNum + 1, changedU, changedV, alreadyDone)
        else : #source is not in vc , target is
        #If sourceCoveredVertex in this layer , is not a covered vertex on the next layer, the edge modified at this layer is going to be in one of
        #the routes between in_neighbour of sourceCoveredVertex and an targetCoveredVertex.
        #Remember : it is impossible for both sourceCoveredVertex and targetCoveredVertex to be not in the covered vertices of the next layer
        #since that will violate the vertex cover property.
            for inNeighbourOfSource in self._graphLayer[layerNum].neighbors_in(sourceCoveredVertex) :
                self._updateHeirarchicalInternal(inNeighbourOfSource,targetCoveredVertex, delta, layerNum + 1, changedU, changedV, alreadyDone)

    #Public function that is called to update the weight of the edge uv.
    #delta is a dict with {metric1:newcost, metric2:newCost}.
    #If metric1 and metric2 already exist on the edge, their costs are updated or else a new metric is added
    @timeit
    def updateWeight(self, u, v, newWeight) :
        delta = {}
        response={"success":True}
        try :
            uv = self._graphLayer[0].edge_label(u,v)
            for metric in newWeight:
                delta[metric] = newWeight[metric] - uv.get(metric, 0)
            self._updateHeirarchicalInternal(u, v, delta, 0, u, v, {})
        except :
            response["success"] = False
        return response
