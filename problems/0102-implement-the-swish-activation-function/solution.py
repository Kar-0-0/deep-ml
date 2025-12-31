def swish(x: float) -> float:
	"""
	Implements the Swish activation function.

	Args:
		x: Input value

	Returns:
		The Swish activation value
	"""
    neg_x = -x
	out = x * (1/(1+2.71828**neg_x))

    return out