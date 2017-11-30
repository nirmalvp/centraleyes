import socket
import sys
from thread import *
#from sage.all import *
#from OverlayGraph import OverlayGraph

HOST = ''   # Symbolic name meaning all available interfaces
PORT = 12345 # Arbitrary non-privileged port

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print 'Socket created'
print socket.gethostbyname(socket.gethostname())

#overlay = OverlayGraph()
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
    conn.send('Welcome to the server. Type something and hit enter\n') #send only takes string

    #infinite loop so that function do not terminate and thread do not end.
    while True:

        #Receiving from client
        data = conn.recv(1024)
        #path = overlay.findOptimumRoute(overlay._graphLayer[0].random_vertex(), overlay._graphLayer[0].random_vertex(), {'dist' : 1})
        #route = path[1]
        #reply = str(route)
        #print repr(data)
        #print data
        if data == '\r\n':
            break
        #conn.sendall(reply)
        conn.sendall("HI")
    #came out of loop
    print "out of loop"
    conn.close()

#now keep talking with the client
while 1:
    #wait to accept a connection - blocking call
    conn, addr = s.accept()
    print 'Connected with ' + addr[0] + ':' + str(addr[1])

    #start new thread takes 1st argument as a function name to be run, second is the tuple of arguments to the function.
    start_new_thread(clientthread ,(conn,))

s.close()
