import threading
import queue

class Subprocesos:
	
	def __init__(self):
		self.threads = queue.Queue()
		
	def ejecutar(self, funcion, parametros):
		th = threading.Thread(target=funcion, args=parametros)
		th.start()
		self.threads.put(th)
		
	def esperar(self):
		while not self.threads.empty():
			th = self.threads.get()
			th.join()
		