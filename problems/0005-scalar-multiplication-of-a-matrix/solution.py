def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
	out = [[0 for i in range(len(matrix[0]))] for j in range(len(matrix))]
	for i in range(len(matrix)):
		for j in range(len(matrix[0])):
			out[i][j] = scalar * matrix[i][j]
	
	return out
