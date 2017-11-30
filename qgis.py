from qgis.core import *
import qgis.utils
layer = qgis.utils.iface.activeLayer()
selection = layer.selectedFeatures()
for s in selection:
    geom = s.geometry()
    a =geom.asPoint()
    print (a.x(), a.y())
uri = "/home/nirmal/Learning/pfe/centraleyes/route.csv?delimiter=%s&xField=%s&yField=%s" % (",", "x", "y")
vlayer = QgsVectorLayer(uri, "layername", "delimitedtext")
#iface.addVectorLayer(vlayer)
QgsMapLayerRegistry.instance().addMapLayer(vlayer)
#print vlayer.isValid()
#print vlayer.pendingFields()