class Test:
	def test(resultado_obtenido, valor_esperado, silent=True):
		if not silent:
			print(resultado_obtenido)
			print(resultado_obtenido == valor_esperado)
			return
		if resultado_obtenido != valor_esperado:
			print('> ERROR! SE OBTUVO:')
			print('>',resultado_obtenido)
			print('> PERO SE ESPERABA:')
			print('>',valor_esperado)
