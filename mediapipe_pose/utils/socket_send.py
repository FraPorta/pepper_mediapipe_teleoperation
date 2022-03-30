import zmq
import time
import json
import sys
import socket
import urllib.request


## class SocketSend
#
# This class creates a ZeroMQ socket to send a Python dictionary over TCP 
class SocketSend:
    ctx = None
    sock = None

    ## method init
    # 
    # class initialization 
    def __init__(self):
        # initialize socket
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        
        external_ip = urllib.request.urlopen('https://ident.me').read().decode('utf8')   
        print(external_ip)

        # try socket bind
        try: 
            self.sock.bind("tcp://*:1234")
            # self.sock.bind("tcp://130.251.13.116:1234")
        except zmq.error.ZMQError as e:
            print(e)
            sys.exit(-1)
    
    ## method send
    #
    # convert dictionary to json and send it via tcp socket
    def send(self, msg_dict):
        msg_json = json.dumps(msg_dict)  
        # print(msg_json)
        self.sock.send_string(msg_json)

    ## method close
    # 
    # close socket
    def close(self):
        self.sock.close()
        self.ctx.term()