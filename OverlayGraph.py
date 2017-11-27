import pickle
from sage.all import *
import config as cfg
class OverlayGraph():
    def __init__(self):
        self._graphLayer = []
        self._coveredVertices = []
        try :
            overlayGraphFile = cfg.filelocation["OVERLAYGRAPH"]
            self._graphLayer = pickle.load(open( overlayGraphFile, "rb" ))
        except IOError:
            self._createOverLayGraph()

    def _getAllNeighbourDetails(self, vertex, i) :
        for (_, neighbourVertex, edgeLabel) in self._graphLayer[i-1].edges_incident(vertex):
            if neighbourVertex != vertex : #A loop
                intermediateVertices = edgeLabel.get("intermediateVertices", [])
                yield (neighbourVertex, intermediateVertices)

    def _createAllPathLayer(self, i) :
        overlayLayer = DiGraph(multiedges=True)
        print "Cover vertices = " , self._coveredVertices[i]
        for vertex in self._coveredVertices[i]:
            for (neighbourVertex, intermediateVerticesBetweenSourceAndNeighbour) in self._getAllNeighbourDetails(vertex, i):
                if neighbourVertex in self._coveredVertices[i]:
                    print "In cover" , vertex, intermediateVerticesBetweenSourceAndNeighbour, neighbourVertex
                    overlayLayer.add_edge(vertex, neighbourVertex, {"intermediateVertices":intermediateVerticesBetweenSourceAndNeighbour})
                ##If neighbour isnt in the cover, the neighbour of neightbour definitely is
                else :
                    for (neighbourOfNeighbourVertex, intermediateVerticesBetweenNeighbours) in self._getAllNeighbourDetails(neighbourVertex, i) :
                        newIntermediateVertices = []
                        if neighbourOfNeighbourVertex != vertex:
                            newIntermediateVertices.extend(intermediateVerticesBetweenSourceAndNeighbour)
                            newIntermediateVertices.append(neighbourVertex)
                            newIntermediateVertices.extend(intermediateVerticesBetweenNeighbours)
                            print "not cover" , vertex, newIntermediateVertices, neighbourOfNeighbourVertex
                            overlayLayer.add_edge(vertex, neighbourOfNeighbourVertex, {"intermediateVertices":newIntermediateVertices})
        return overlayLayer

    def _createLayer(self, i):
        undirectedGraph = Graph(self._graphLayer[i-1])
        self._coveredVertices[i] = undirectedGraph.vertex_cover()
        if cfg.ALLPATH:
            self._graphLayer[i] = self._createAllPathLayer(i)

    def _createOverLayGraph(self):
        try :
            baseGraphFile = cfg.filelocation["BASEGRAPH"]
            numberOfLayers = cfg.numberOfLayers
            self._graphLayer = [[] for k in xrange(numberOfLayers)]
            self._coveredVertices = [[] for k in xrange(numberOfLayers)]
            self._graphLayer[0] = DiGraph(pickle.load(open( baseGraphFile, "rb" )))
            self._coveredVertices[0] = self._graphLayer[0].vertices()
            for i in xrange(1, numberOfLayers) :
                self._createLayer(i)
        except IOError :
            print "No base file located"
            import sys
            sys.exit()

overlay = OverlayGraph()
