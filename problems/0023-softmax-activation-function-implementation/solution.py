import math

def softmax(scores: list[float]) -> list[float]:
    out = [0 for i in range(len(scores))]
    vec_max = max(scores)
    for i in range(len(scores)):
        scores[i] -= vec_max
    for i in range(len(scores)):
        out[i] = (math.e ** scores[i]) 
    
    vec_sum = sum(out)
    for i in range(len(out)):
        out[i] /= vec_sum
    
    
    return out
