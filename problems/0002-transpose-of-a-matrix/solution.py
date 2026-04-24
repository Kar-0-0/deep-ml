def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    # transpose(NxM) = MxN 
    out = [[0 for i in range(len(a))] for j in range(len(a[0]))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            out[j][i] = a[i][j]
    
    return out