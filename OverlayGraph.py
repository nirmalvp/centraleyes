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
            self._graphLayer = pickle.load(open( overlayGraphFile, "rb" ))
            #OverlayGraphs are pickled as a list of networkx graphs. So COnverting each networkx element in the list  into a DiGraph object
            self._graphLayer = map(DiGraph, self._graphLayer)
        except IOError:
            #This occures if a pre-existing overlay graph file doesnt exist on disk. We need to create an overlay graph and store it so that future runs
            #dont have to do this again.
            applogger.info("Creating overlaygraph for the first time. This might take a while")
            self._createOverLayGraph()
            self._writeOverLayGraphToFile()

    def _writeOverLayGraphToFile(self) :
        graphLayerList = []
        #WOrkaround since Digraph object can't be directly pickled. However, pickle works on networkx objects
        for graphLayer in self._graphLayer :
            graphLayerList.append(graphLayer.networkx_graph())
        try :
            overlayGraphFile = cfg.filelocation["OVERLAYGRAPH"]
            pickle.dump(graphLayerList, open(overlayGraphFile, "wb" ))
        except IOError:
            applogger.error("Error while writing overlayGraphFile")

    #Helper function to get the outgoing edges(including parallel edges) of a vertex and extract path information from them.
    def _getAllNeighbourDetails(self, vertex, i) :
        for (_, neighbourVertex, edgeLabel) in self._graphLayer[i-1].edges_incident(vertex):
            if neighbourVertex != vertex : #If there is a loop, ignore it
            #Overlay Graph at layer i is a condensed graph of layer i-1.
            #Graph (A->C) at layer i have been could have been created out of graph(A->B->C) in layer(i-1)
            #So along the edge A->C, we need to store the actual intermediate nodes along the route, so that we can
            #use this information to display the actual route during query stage.
                intermediateVertices = edgeLabel.get("intermediateVertices", [])
                yield (neighbourVertex, intermediateVertices)

    #Given a vertex v (included in the cover), this function creates an edge for every path from v to any other vertex that is also included
    #in the cover.
    def createEdges(self, vertex, i) :
        applogger.debug("considering vertex : %s,%s"%(vertex))
        for (neighbourVertex, intermediateVerticesBetweenSourceAndNeighbour) in self._getAllNeighbourDetails(vertex, i):
            #Graph (A->B->C). Imagine both A as well as B are included in the vertex cover(not optimum)
            #In the overlay graph, we add a edge from A to B and retain the intermediateVertices between A->B as it is in the lower layer
            if neighbourVertex in self._coveredVertices[i]:
                self.overlayLayer.add_edge(vertex, neighbourVertex, {"intermediateVertices":intermediateVerticesBetweenSourceAndNeighbour})
            else :
                ##If immediate neighbour isnt in the vertex cover, the neighbour of neighbour is definitely in the vertex cover
                #or else the vertex cover property is violated.
                #Imagine Graph (A-B-C), A and C are in the vertex cover, so we add an edge from A to C in the overlay Graph
                #However the new edge's intermediateVertices are (intermediateVertices from A to B + Node B + intermediateVertices from B to C)
                for (neighbourOfNeighbourVertex, intermediateVerticesBetweenNeighbours) in self._getAllNeighbourDetails(neighbourVertex, i) :
                    newIntermediateVertices = []
                    #Graph (A-B-A) could created an infinite loop. So we check if the neighbour of neighbour isnt the source vertex itself
                    if neighbourOfNeighbourVertex != vertex:
                        newIntermediateVertices.extend(intermediateVerticesBetweenSourceAndNeighbour)
                        newIntermediateVertices.append(neighbourVertex)
                        newIntermediateVertices.extend(intermediateVerticesBetweenNeighbours)
                        self.overlayLayer.add_edge(vertex, neighbourOfNeighbourVertex, {"intermediateVertices":newIntermediateVertices})

    #This function retrieves all the edges between two nodes in an overlay graph. THere will be one edge per every route from a vertex to another
    def _createAllPathLayer(self, i) :
        self.overlayLayer = DiGraph(multiedges=True)
        #applogger.debug("Cover vertices = %s" %self._coveredVertices[i])
        pool = None
        #Edge creation between nodes on the overlay graph can be parallelized. Hence using a threadpool.
        #https://stackoverflow.com/questions/2846653/how-to-use-threading-in-python
        pool = ThreadPool()
        pool.map(lambda v : self.createEdges(v,i), self._coveredVertices[i])
        pool.close()
        pool.join()
        #At this point overlayLayer at level i is complete. That is all routes between vertex covers of layer (i-1) is connected via an edge
        return self.overlayLayer

    #This function Creates the ith layer of the graph using the (i-1)th layer
    def _createLayer(self, i):
        #Find the vertex cover of the graph at the lower layer. Vertex cover works only on for undirected graph in Sage
        #So, converting the lower layer into a undirected graph and also removing the multiple edges in the lower graph
        undirectedGraph = Graph(self._graphLayer[i-1])
        undirectedGraph.remove_multiple_edges()
        undirectedGraph.allow_multiple_edges(False)
        ##Remove
        applogger.debug("Has multiple edge = %s" %undirectedGraph.has_multiple_edges())
        applogger.debug("Real has multiple edge = %s" %self._graphLayer[i-1].has_multiple_edges())
        applogger.debug("Number of vertices = %s"%len(undirectedGraph))
        applogger.debug("Number of edges = %s"%undirectedGraph.size())
        applogger.debug("NRandom edges = %s, %s, %s"%undirectedGraph.random_edge())
        ##EndOfRemove
        #g = undirectedGraph.plot()
        #save(g,'/tmp/dom%s.png'%(i-1),axes=False,aspect_ratio=True)
        #os.system('display /tmp/dom%s.png'%(i-1))
        applogger.debug("Starting vertex cover of layer %s"%(i-1))
        #coveredVertices ie the vertices of ith layer are the vertex cover of the (i-1)th layer
        self._coveredVertices[i] = undirectedGraph.vertex_cover()
        #self._coveredVertices[i] = undirectedGraph.vertex_cover(reduction_rules=False, algorithm='MILP')
        applogger.debug("Retrieved vertex cover of layer %s"%(i-1))
        #This is a switch in-case I decide to store only the shortest path later instead of all routes from one node to another
        if cfg.ALLPATH:
            self._graphLayer[i] = self._createAllPathLayer(i)

    def _createOverLayGraph(self):
        try :
            #BaseGraph is the raw graph (road network).
            baseGraphFile = cfg.filelocation["BASEGRAPH"]
            numberOfLayers = cfg.numberOfLayers
            #initialize graphLayer . ie start with every layer as an empty list. Later on graphLayer[n] will be the nth layer of the overlay graph
            self._graphLayer = [[] for k in xrange(numberOfLayers)]
            self._coveredVertices = [[] for k in xrange(numberOfLayers)]
            #Initialize the first layer as the raw graph data
            self._graphLayer[0] = DiGraph(pickle.load(open( baseGraphFile, "rb" )))
            applogger.debug("Loaded baseGraph")
            #Initialize the cover in the first layer as all the vertices of the raw graph
            self._coveredVertices[0] = self._graphLayer[0].vertices()
            applogger.debug("Retrieved baseGraph vertices")
            for i in xrange(1, numberOfLayers) :
                applogger.debug("Generating Layer %s"%i)
                self._createLayer(i)
                ##Remove
                applogger.debug("allows_multiple_edges : %s"%self._graphLayer[i].allows_multiple_edges())
                applogger.debug("has_multiple_edges : %s"%self._graphLayer[i].has_multiple_edges())
                applogger.debug("has_loops : %s"%self._graphLayer[i].has_loops())
                pickle.dump(self._graphLayer[i].networkx_graph(),open("/home/nirmal/%s.pickle"%i, "wb" ))
                ##End of remove
        except IOError :
            print "No base file located"
            import sys
            sys.exit()



overlay = OverlayGraph()
