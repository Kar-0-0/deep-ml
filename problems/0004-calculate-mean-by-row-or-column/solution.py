def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    means = []
    if mode == 'row':
        for row in matrix:
            means.append(sum(row)/len(row))
    else:
        for i in range(len(matrix[0])):
            column = [matrix[j][i] for j in range(len(matrix))]
            means.append(sum(column) / len(column))

	return means