import socket
import sys
from thread import *
import json
from sage.all import *
from OverlayGraph import OverlayGraph

HOST = ''   # Symbolic name meaning all available interfaces
PORT = 12345 # Arbitrary non-privileged port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print 'Socket created'
print socket.gethostbyname(socket.gethostname())

overlay = OverlayGraph()
#Bind socket to local host and port
try:
    s.bind(('', PORT))
except socket.error , msg:
    print 'Bind failed. Error Code : ' + str(msg[0]) + ' Message ' + msg[1]
    sys.exit()

print 'Socket bind complete'

#Start listening on socket
s.listen(10)
print 'Socket now listening on PORT:', PORT

#Function for handling connections. This will be used to create threads
def clientthread(conn):
    #Sending message to connected client
    #infinite loop so that function do not terminate and thread do not end.
    while True:
        #Receiving from client
        try :
            data = conn.recv(1024)
            data = data.strip()
            if not data or data[0] != '{' :
                break
            data = json.loads(data)
            args = data.get("args")
            sourceVertex = tuple(args.get("sourceVertex"))
            destVertex = tuple(args.get("destVertex"))
            jsonDict = overlay.findOptimumRoute(sourceVertex, destVertex, {'dist' : 1})
            response = json.dumps(jsonDict)
        except:
            print "error occured"
            response = "Sorry, an error occured"
        conn.sendall(response)
    conn.close()

#now keep talking with the client
while 1:
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    print 'Connected with ' + addr[0] + ':' + str(addr[1])
    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    start_new_thread(clientthread ,(conn,))

s.close()
