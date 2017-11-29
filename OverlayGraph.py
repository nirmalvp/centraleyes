import pickle
from sage.all import *
import config as cfg
from logger import applogger
import itertools
from multiprocessing.dummy import Pool as ThreadPool
from heapq import  heappop, heappush

#OverlayGraph maintains the k layers of Overlay graphs where _graphLayer[n] is the nth overlay layer
class OverlayGraph():
    def __init__(self):
        #_graphLayer[n] is the nth overlay layer
        #_coveredVertices[n] is the set of vertices in _graphLayer[n]
        self._graphLayer = []
        self._coveredVertices = []
        self._metrics = cfg.metrics
        self._numberOfLayers = None
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
            save(self._graphLayer[layerNum].plot(), saveLocation+".png",axes=False)#,aspect_ratio=True)

    def _plotAllLayer(self, saveLocation) :
        for layerNum in xrange(0, self._numberOfLayers):
            self._plotLayer(layerNum, saveLocation+str(layerNum))


    def _writeOverLayGraphToFile(self) :
        graphLayerList = []
        try :
            overlayGraphFile = cfg.filelocation["OVERLAYGRAPH"]
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
                #applogger.debug("EdgeLabel : %s"%edgeLabel)
                for metric in self._metrics :
                    edgeMetrics[metric] = edgeLabel[metric]
                yield (neighbourVertex, edgeMetrics)

    #Given a vertex v (included in the cover), this function creates an edge for every path from v to any other vertex that is also included
    #in the cover.
    def createEdges(self, vertex, layerNum) :
        #applogger.debug("considering vertex : %s,%s"%(vertex))
        for (neighbourVertex, edgeMetricsBetweenSourceAndNeighbour) in self._getAllNeighbourDetails(vertex, layerNum - 1):
            #Graph (A->B->C). Imagine both A as well as B are included in the vertex cover(not optimum)
            #In the overlay graph, we add a edge from A to B and retain the intermediateVertices between A->B as it is in the lower layer
            if neighbourVertex in self._coveredVertices[layerNum]:
             #   edgeLabel = {"intermediateVertices":intermediateVerticesBetweenSourceAndNeighbour}
                self.overlayLayer.add_edge(vertex, neighbourVertex, edgeMetricsBetweenSourceAndNeighbour)
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
                        newEdgeMetrics = { k: edgeMetricsBetweenSourceAndNeighbour[k] + edgeMetricsBetweenNeighbours[k] for k in set(edgeMetricsBetweenSourceAndNeighbour) if k != "intermediateVertices"}
                        newIntermediateVertices = []
                        newIntermediateVertices.extend(intermediateVerticesBetweenSourceAndNeighbour)
                        newIntermediateVertices.append(neighbourVertex)
                        newIntermediateVertices.extend(intermediateVerticesBetweenNeighbours)
                        newEdgeMetrics["intermediateVertices"] = newIntermediateVertices
                        self.overlayLayer.add_edge(vertex, neighbourOfNeighbourVertex, newEdgeMetrics)
    #This function creates an overlay graph by considering every route from one covered vertex to another covered vertex in the base graph
    def _createAllPathLayer(self, layerNum) :
        self.overlayLayer = DiGraph(multiedges=True)
        pool = None
        #Edge creation between nodes on the overlay graph can be parallelized. Hence using a threadpool.
        #https://stackoverflow.com/questions/2846653/how-to-use-threading-in-python
        pool = ThreadPool()
        pool.map(lambda v : self.createEdges(v,layerNum), self._coveredVertices[layerNum])
        pool.close()
        pool.join()
        #At this point overlayLayer at level i is complete. That is all routes between vertex covers of layer (i-1) is connected via an edge
        return self.overlayLayer

    def _getVertexCoverOflayer(self, layerNum):
        undirectedGraph = Graph(self._graphLayer[layerNum])
        undirectedGraph.remove_multiple_edges()
        undirectedGraph.allow_multiple_edges(False)
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
            applogger.debug("Retrieved baseGraph vertices")
            for layerNum in xrange(1, self._numberOfLayers) :
                applogger.debug("Generating Layer %s"%layerNum)
                self._createLayer(layerNum)
                applogger.debug("Layer %s Generated. Number of vertices = %s, Number of edges = %s"%(layerNum, self._graphLayer[layerNum].order(), self._graphLayer[layerNum].size()))
        except IOError :
            print "No base file located"
            import sys
            sys.exit()

    def _runDijsktraOnOverlay(self,forwardAccessNodePathCost, targetAccessNodes, userWieghts) :
        queue = []
        for accessNodeOfSource, costToAccessNode in forwardAccessNodePathCost:
            heappush(queue, (costToAccessNode,(accessNodeOfSource, None, [])))
        visited = {}
        overlayGraph = self._graphLayer[self._numberOfLayers - 1]
        #print overlayGraph.edges()
        while queue and targetAccessNodes:
            path_cost, (vertex, parentOfVertex, edgeIntermediateVertices) = heappop(queue)
            if visited.get(vertex) is None:
                visited[vertex] = {"cost":path_cost, "parent" : parentOfVertex , "intermediateVerticesToParent" : edgeIntermediateVertices}
                if vertex in targetAccessNodes :
                    targetAccessNodes.remove(vertex)
                for neighbourOverlayNode in overlayGraph.neighbors_out(vertex):
                    if visited.get(neighbourOverlayNode) is None:
                        edges = overlayGraph.edge_label(vertex, neighbourOverlayNode)
                        edge_cost, edgeIntermediateVertices = self._getMinimumWeightedEdge(edges, userWieghts)
                        heappush(queue, (path_cost + edge_cost, (neighbourOverlayNode, vertex, edgeIntermediateVertices)))
        return visited


    #If edges is a single element, return the cost of the edge as per userWieght
    #If edges is a list of edges, returns the cost of the edge with the minimum cost
    def _getMinimumWeightedEdge(self, edges, userWieghts) :
        def getWeightedEdgeCost(edge) :
            weightedEdgeCost = 0
            for metric in userWieghts :
                weightedEdgeCost += edge.get(metric) * userWieghts[metric]
            return weightedEdgeCost
        if type(edges) is list :
            optimumEdge = min(edges, key = lambda d: getWeightedEdgeCost(d))
            return (getWeightedEdgeCost(optimumEdge), optimumEdge.get("intermediateVertices", []))
        return (getWeightedEdgeCost(edges), edges.get("intermediateVertices", []))

    #Runs a Dijsktra algorithm from sourceVertex but without expanding accessNodes.
    #Thus when the priority Q is empty, we will have the optimum path to every access node
    #reachable from sourceVertex.
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
                if v not in self._coveredVertices[numberOfLayers-1] :
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
                    accessNodes.add(v)
        return (visited, accessNodes)

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
        #TargetVertex was already settled during the forward search. We have nothing more to do here.
        if targetVertex in verticesVisitedDuringForwardSearch :
            path = self._getNodesAlongShortestPath(targetVertex, verticesVisitedDuringForwardSearch)
            path.reverse()
            return path
        #To run, Dijsktra on the overlay graph, we initialize the priority Q with every access node obtained in the forward
        #search and its cost from source. By the end of the Dijstra in Overlay graph, we get the cost from source to every
        #access node of the target
        forwardAccessNodePathCost = [(accessNode, verticesVisitedDuringForwardSearch.get(accessNode).get("cost")) for accessNode in forwardAccessNodes]
        verticesVisitedDuringOverlaySearch = self._runDijsktraOnOverlay(forwardAccessNodePathCost, backWardAccessNodes.copy(), userWieghts)
        #Sample verticesVisitedDuringOverlaySearch :
        #{10: {'cost': 1, 'intermediateVerticesToParent': [], 'parent': None}, 4: {'cost': 3, 'intermediateVerticesToParent': [], 'parent': None}, 6: {'cost': 4, 'intermediateVerticesToParent': [], 'parent': 4}}
        distToTargetViaAcessNodes = map(lambda accessNode : (verticesVisitedDuringOverlaySearch[accessNode].get("cost") + verticesVisitedDuringBackwardSearch[accessNode].get("cost"), accessNode), backWardAccessNodes)
        minimumCost, minimumAccessNode = min(distToTargetViaAcessNodes)
        pathAlongOverLayGraph,lastNode = self._getNodesAlongShortestPath(minimumAccessNode, verticesVisitedDuringOverlaySearch, True)
        pathAlongForwardSearch,_ = self._getNodesAlongShortestPath(lastNode, verticesVisitedDuringForwardSearch, False)
        pathAlongBackWardSearch,_ = self._getNodesAlongShortestPath(minimumAccessNode, verticesVisitedDuringBackwardSearch, False)
        path = []
        path.extend(pathAlongOverLayGraph)
        path.extend(pathAlongForwardSearch)
        path.reverse()
        path.extend(pathAlongBackWardSearch)
        return path

overlay = OverlayGraph()
overlay.findOptimumRoute(1, 8, {'dist' : 1})
