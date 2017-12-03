# centraleyes

Centraleyes allowes you to find optimum paths between two points on a graph at the scale of real road networks, in milliseconds
using the concept of OverlayGraph Heirarchy. Centraleyes can find the optimum path according to the metrics user provides, ie shortest path vs shortest travel time for example.

Overlay Graph is a condensed form(lesser vertices of your input roadmap).This helps you to run your dijsktra queries on this condensed graph rather than a real road map with possibly millions of node.

The class here is a heirarchy of overlay Graph where each layer is an overlay graph. At each layer, the number of vertices in the graph reduces, however the edge could also increase since there is an edge in the overlay graph corresponding to every actual route in the road map, between the vertices of the overlay graph.

Your basegraph could contain enormous information (number of traffic lights, travel time, distance, turn costs) etc that you may
not want the user to query by. For all the queriable metrices, add a corresponding entry in "metrics" field in the config file to make it queriable.

Overlay graph at level i is constructed by considering only the vertex covers of the graph at level (i-1) and adding an edge corresponding to every route between these vertex covers in the immediate lower layer.

This leads to a formation of 2^k-All Path Cover of the base graph, where k is the number of layers.

For proof and additional informations :
1. http://www.vldb.org/pvldb/vol7/p893-funke.pdf
2. https://dl.acm.org/citation.cfm?id=2983712

## HOW TO RUN :

### Prerequisite :
	1. A sage Digraph saved into file in the same folder as script. This could be your real road network graph.
    2. Edit config file to point "BASEGRAPH" to the location of file in Point 1.
	3. Edit Config files to set the number of layers for the overlay graph heirarchy
    4. Edit config file to set "metrics" to the list of metrics you want your users to query optimum path by. eg : slope, dist, travel_time etc. Note : These metrices should be available in your baseGraph. This config merely controls whether these metrics should
    also be present in the overlaygraph.

Run : sage -python sock.py

This starts a server listening to a port. Your client can then send the server the starting and the edestination vertex and the server will respond back with the list of vertices in the optimum path between the points.
