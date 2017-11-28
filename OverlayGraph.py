import pickle
from sage.all import *
import config as cfg
from logger import applogger
import itertools
from multiprocessing.dummy import Pool as ThreadPool

#OverlayGraph maintains the k layers of Overlay graphs where _graphLayer[n] is the nth overlay layer
class OverlayGraph():
    def __init__(self):
        #_graphLayer[n] is the nth overlay layer
        #_coveredVertices[n] is the set of vertices in _graphLayer[n]
        self._graphLayer = []
        self._coveredVertices = []
        try :
            #Check if we already have created an overlay graph before. The program can load the file and initialize instantly without
            #re-creating overlay graphs from the scratch
            overlayGraphFile = cfg.filelocation["OVERLAYGRAPH"]
            self._graphLayer = load(overlayGraphFile)
            #OverlayGraphs are pickled as a list of networkx graphs. So COnverting each networkx element in the list  into a DiGraph object
        except IOError:
            #This occures if a pre-existing overlay graph file doesnt exist on disk. We need to create an overlay graph and store it so that future runs
            #dont have to do this again.
            applogger.info("Creating overlaygraph for the first time. This might take a while")
            self._createOverLayGraph()
            self._writeOverLayGraphToFile()

    def _plotLayer(self, layerNum, saveLocation) :
        if layerNum >= cfg.numberOfLayers or layer < 0:
            applogger.error("Layer doesnt exist")
        else :
            save(self._graphLayer[layerNum], saveLocation,axes=False,aspect_ratio=True)

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
                yield (neighbourVertex, intermediateVertices)

    #Given a vertex v (included in the cover), this function creates an edge for every path from v to any other vertex that is also included
    #in the cover.
    def createEdges(self, vertex, layerNum) :
        #applogger.debug("considering vertex : %s,%s"%(vertex))
        for (neighbourVertex, intermediateVerticesBetweenSourceAndNeighbour) in self._getAllNeighbourDetails(vertex, layerNum - 1):
            #Graph (A->B->C). Imagine both A as well as B are included in the vertex cover(not optimum)
            #In the overlay graph, we add a edge from A to B and retain the intermediateVertices between A->B as it is in the lower layer
            if neighbourVertex in self._coveredVertices[layerNum]:
                self.overlayLayer.add_edge(vertex, neighbourVertex, {"intermediateVertices":intermediateVerticesBetweenSourceAndNeighbour})
            else :
                ##If immediate neighbour isnt in the vertex cover, the neighbour of neighbour is definitely in the vertex cover
                #or else the vertex cover property is violated.
                #Imagine Graph (A-B-C), A and C are in the vertex cover, so we add an edge from A to C in the overlay Graph
                #However the new edge's intermediateVertices are (intermediateVertices from A to B + Node B + intermediateVertices from B to C)
                for (neighbourOfNeighbourVertex, intermediateVerticesBetweenNeighbours) in self._getAllNeighbourDetails(neighbourVertex, layerNum - 1) :
                    newIntermediateVertices = []
                    #Graph (A-B-A) could created an infinite loop. So we check if the neighbour of neighbour isnt the source vertex itself
                    if neighbourOfNeighbourVertex != vertex:
                        newIntermediateVertices.extend(intermediateVerticesBetweenSourceAndNeighbour)
                        newIntermediateVertices.append(neighbourVertex)
                        newIntermediateVertices.extend(intermediateVerticesBetweenNeighbours)
                        self.overlayLayer.add_edge(vertex, neighbourOfNeighbourVertex, {"intermediateVertices":newIntermediateVertices})

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
            numberOfLayers = cfg.numberOfLayers
            #initialize graphLayer . ie start with every layer as an empty list. Later on graphLayer[n] will be the nth layer of the overlay graph
            self._graphLayer = [[] for k in xrange(numberOfLayers)]
            self._coveredVertices = [[] for k in xrange(numberOfLayers)]
            #Initialize the first layer as the raw graph data
            self._graphLayer[0] = load(baseGraphFile)
            applogger.debug("Loaded baseGraph")
            #Initialize the cover in the first layer as all the vertices of the raw graph
            self._coveredVertices[0] = self._graphLayer[0].vertices()
            applogger.debug("Retrieved baseGraph vertices")
            for layerNum in xrange(1, numberOfLayers) :
                applogger.debug("Generating Layer %s"%layerNum)
                self._createLayer(layerNum)
                applogger.debug("Layer %s Generated. Number of vertices = %s, Number of edges = %s"%(layerNum, self._graphLayer[layerNum].order(), self._graphLayer[layerNum].size()))
                #pickle.dump(self._graphLayer[i].networkx_graph(),open("/home/nirmal/%s.pickle"%i, "wb" ))
        except IOError :
            print "No base file located"
            import sys
            sys.exit()



overlay = OverlayGraph()
