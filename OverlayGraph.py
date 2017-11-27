import pickle
from sage.all import *
import config as cfg
from logger import applogger
import itertools
from multiprocessing.dummy import Pool as ThreadPool


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

    def createEdges(self, vertex, i) :

        for (neighbourVertex, intermediateVerticesBetweenSourceAndNeighbour) in self._getAllNeighbourDetails(vertex, i):
            if neighbourVertex in self._coveredVertices[i]:
                applogger.debug("In cover %s : %s : %s"%(vertex, intermediateVerticesBetweenSourceAndNeighbour, neighbourVertex))
                #print "In cover " , vertex, intermediateVerticesBetweenSourceAndNeighbour, neighbourVertex
                self.overlayLayer.add_edge(vertex, neighbourVertex, {"intermediateVertices":intermediateVerticesBetweenSourceAndNeighbour})
            ##If neighbour isnt in the cover, the neighbour of neightbour definitely is
            else :
                for (neighbourOfNeighbourVertex, intermediateVerticesBetweenNeighbours) in self._getAllNeighbourDetails(neighbourVertex, i) :
                    newIntermediateVertices = []
                    if neighbourOfNeighbourVertex != vertex:
                        newIntermediateVertices.extend(intermediateVerticesBetweenSourceAndNeighbour)
                        newIntermediateVertices.append(neighbourVertex)
                        newIntermediateVertices.extend(intermediateVerticesBetweenNeighbours)
                        applogger.debug("not cover %s : %s : %s"%(vertex, newIntermediateVertices, neighbourOfNeighbourVertex))
                        #print "not cover " , vertex, newIntermediateVertices, neighbourOfNeighbourVertex
                        self.overlayLayer.add_edge(vertex, neighbourOfNeighbourVertex, {"intermediateVertices":newIntermediateVertices})

    def _createAllPathLayer(self, i) :
        self.overlayLayer = DiGraph(multiedges=True)
        applogger.debug("Cover vertices = %s" %self._coveredVertices[i])
        pool = None
        pool = ThreadPool()
        pool.map(lambda v : self.createEdges(v,i), self._coveredVertices[i])
        pool.close()
        pool.join()
        return self.overlayLayer

    def _createLayer(self, i):

        undirectedGraph = Graph(self._graphLayer[i-1])
        undirectedGraph.remove_multiple_edges()
        #g = undirectedGraph.plot()
        #save(g,'/tmp/dom%s.png'%(i-1),axes=False,aspect_ratio=True)
        #os.system('display /tmp/dom%s.png'%(i-1))
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
