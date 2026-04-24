import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	out = [0 for i in range(len(features))]

	for i in range(len(features)):
		dot_acc = 0
		for k in range(len(weights)):
			dot_acc += (features[i][k] * weights[k]) 
		out[i] = dot_acc + bias

	for i in range(len(out)):
		x = out[i]
		out[i] = 1 / (1 + math.exp(-x))

	probs = out

	acc = 0
	for i in range(len(probs)):
		acc += (probs[i] - labels[i]) ** 2

	mse = acc / len(labels)

	return probs, mse










