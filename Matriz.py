from itertools import cycle

class Matriz:
	
	def __init__(self, n, m, patron):
		self.n = n
		self.m = m
		datos = cycle(patron)
		r = []
		for i in range(n):
			fila = []
			for j in range(m):
				fila.append(next(datos))
			r.append(fila)
		self.datos = r
		
	def dim_n(self):
		return self.n

	def dim_m(self):
		return self.m
		
	def dim(self):
		return str(self.dim_n()) + 'x' + str(self.dim_m())
	
	def get(self, i, j):
		return self.datos[i][j]
		
	def getRow(self, i):
		return self.datos[i]
	
	def getRows(self):
		return self.datos
	
	def getColumn(self, j):
		r = []
		for i in range(self.dim_n()):
			r.append(self.get(i,j))
		return r
	
	def set(self,i,j,val):
		self.datos[i][j] = val
		
	def __str__(self):
		string = ''
		for fila in self.datos:
			string = string + str(fila) + '\n'
		return string
	
	def _producto_interno(self, v1, v2):
		r = 0
		for e1,e2 in zip(v1,v2):
			r = r + e1 * e2
		return r
	
	def expandirColumnas(self, m):
		n = self.dim_n()
		r = Matriz(n, m, [0])
		for i in range(n):
			for j in range(m):
				val = self.get(i, j % self.dim_m())
				r.set(i,j,val)
		return r
	
	def transpuesta(self):
		r = Matriz(self.dim_m(), self.dim_n(), [0])
		for i in range(self.dim_n()):
			for j in range(self.dim_m()):
				val = self.get(i,j)
				r.set(j,i,val)
		return r
	
	def mul_directa_mat(self, mat):
		r = []
		for i in range(self.n):
			for j in range(self.m):
				r.append(self.get(i, j) * mat.get(i, j))
		return Matriz(self.n, self.m, r)
		
	def sum_mat(self, mat):
		r = []
		for i in range(self.n):
			for j in range(self.m):
				r.append(self.get(i, j) + mat.get(i, j))
		return Matriz(self.n, self.m, r)
		
	def mul_mat(self, mat):
		dim_n = self.dim_n()
		dim_m = mat.dim_m()
		mat = mat.transpuesta()
		r = []
		for fila in self.getRows():
			for columna in mat.getRows():
				pi = self._producto_interno(fila, columna)
				r.append(pi)
		return Matriz(dim_n, dim_m, r)
	
	def evaluar(self, f):
		r = []
		for i in range(self.dim_n()):
			for j in range(self.dim_m()):
				op = f(self.get(i,j))
				r.append(op)
		return Matriz(self.dim_n(), self.dim_m(), r)
		
	def vector_medio(self):
		r = []
		for fila in self.datos:
			sumatoria = 0
			for e in fila:
				sumatoria = sumatoria + e
			r.append(sumatoria/len(fila))
		return Matriz(self.dim_n(), 1, r)
		
def testMatriz():
	m1 = Matriz(3,3,[0,1])
	m2 = Matriz(3,3,[1,0])
	print(m1.sum_mat(m2))
	m1 = Matriz(3,3,[0,1])
	m2 = Matriz(3,3,[1,0])
	print(m1.mul_directa_mat(m2))
	m = Matriz(4,2,[5,3,6])
	print(m)
	print(m.transpuesta())
	print(m.transpuesta().dim())
	m1 = Matriz(2,3,[2,1])
	m2 = Matriz(3,3,[3,0])
	print(m1)
	print(m2)
	print(m1.mul_mat(m2))
	print(m1.mul_mat(m2).dim())
	m1 = Matriz(2,2,[1,2])
	m2 = Matriz(2,1,[3,11])
	print(m1)
	print(m2)
	print(m1.mul_mat(m2))
	m = Matriz(2,3,[5,2,4,6])
	print(m)
	print(m.expandirColumnas(5))
	
	
#testMatriz()
	
