def producto_interno(v1,v2):
	resultado = 0
	for i,e1 in enumerate(v1):
		e2 = v2[i]
		resultado += e1 * e2
	return resultado
