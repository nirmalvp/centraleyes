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

    def _createOverLayGraph(self):
        try :
            baseGraphFile = cfg.filelocation["BASEGRAPH"]
            numberOfLayers = cfg.numberOfLayers
            self._graphLayer = [[] for k in xrange(numberOfLayers)]
            self._coveredVertices = [[] for k in xrange(numberOfLayers)]
            self._graphLayer[0] = pickle.load(open( baseGraphFile, "rb" ))
            self._coveredVertices[0] = self._graphLayer[0].vertices()
            for i in xrange(1, numberOfLayers) :
                self._createLayer(i)
        except IOError :
            print "No base file located"
            import sys
            sys.exit()

    def _createlayer(self, i):
        undirectedGraph = Graph(self._graphLayer[i-1])
        self._coveredVertices[i] = undirectedGraph.vertex_cover()
        if cfg.ALLPATH:
            self._createAllPathLayer()

    def _createAllPathLayer(self) :
        overlayLayer = DiGraph(multiedges=True)









overlay = OverlayGraph()
