import zmq  
import threading
from cereal.messaging.utils import get_zmq_socket_path
from common.params import Params
from system.swaglog import cloudlog


ctx = zmq.Context()
sock_get = ctx.socket(zmq.REP)
sock_get_path = get_zmq_socket_path("6001")
cloudlog.info("sock_get_path: %s", sock_get_path)
sock_get.bind(sock_get_path) # get socket

sock_put = ctx.socket(zmq.REP)
sock_put_path = get_zmq_socket_path("6002")
cloudlog.info("sock_put_path: %s", sock_put_path)
sock_put.bind(sock_put_path) # put socket

sock_del = ctx.socket(zmq.REP)
sock_del_path = get_zmq_socket_path("6003")
cloudlog.info("sock_del_path: %s", sock_del_path)
sock_del.bind(sock_del_path) # del socket

params = Params()

class ParamsServer:
    exit_event = threading.Event()
    threads = []

    @staticmethod
    def put_thread(exit_event):
        while not exit_event.is_set():
            key, val = sock_put.recv_multipart()
            cloudlog.info(f"keyvald SET: {key} = {val}")
            params.put(key, val)
            sock_put.send(b"1")
        
    @staticmethod
    def get_thread(exit_event):
        while not exit_event.is_set():
            key = sock_get.recv()
            data = params.get(key)
            cloudlog.info(f"keyvald GET: {key} = {data}")
            sock_get.send(data if data is not None else b"")

    @staticmethod
    def delete_thread(exit_event):
        while not exit_event.is_set():
            key = sock_del.recv()
            params.remove(key)
            sock_del.send(b"1")
    
    @staticmethod 
    def start():
        ParamsServer.exit_event.clear()
        if ParamsServer.threads:
            return
        ParamsServer.threads.append(threading.Thread(target=ParamsServer.put_thread, args=(ParamsServer.exit_event,), daemon=True))
        ParamsServer.threads.append(threading.Thread(target=ParamsServer.get_thread, args=(ParamsServer.exit_event,), daemon=True))
        ParamsServer.threads.append(threading.Thread(target=ParamsServer.delete_thread, args=(ParamsServer.exit_event,), daemon=True))
        
        for thread in ParamsServer.threads:
            thread.start()
            
    @staticmethod 
    def stop():
        ParamsServer.exit_event.set()
        ParamsServer.threads.clear()
    
    @staticmethod
    def wait():
        for thread in ParamsServer.threads:
            thread.join()
        
def main():
    try:
        ParamsServer.start()
        ParamsServer.wait()
    except KeyboardInterrupt:
        ParamsServer.stop()
          
if __name__ == "__main__":
    main()
