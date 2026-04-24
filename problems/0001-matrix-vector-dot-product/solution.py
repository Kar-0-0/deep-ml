def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	if len(a[0]) != len(b):
		return -1
	
	# A * b = NxM * MxK = NxK
	out = []
	for i in range(len(a)):
		acc = 0
		for k in range(len(b)):
			acc += a[i][k] * b[k]

		out.append(acc)

	return out


			
	