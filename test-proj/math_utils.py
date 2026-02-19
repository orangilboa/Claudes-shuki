def clamp(value, min_val, max_val, verbose=True):
    result = max(min_val, min(max_val, value))
    print(f"clamp({value}, {min_val}, {max_val}) -> {result}")
    return result


def factorial(n, verbose=True):
    if n < 0:
        print(f"factorial({n}) -> error: negative input")
        raise ValueError("factorial is not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    if verbose:
        print(f"factorial({n}) -> {result}")
    return result


def is_prime(n, verbose=True):
    if n < 2:
        result = False
    elif n == 2:
        result = True
    elif n % 2 == 0:
        result = False
    else:
        result = all(n % i != 0 for i in range(3, int(n**0.5) + 1, 2))
    if verbose:
        print(f"is_prime({n}) -> {result}")
    return result


def average(numbers, verbose=True):
    if not numbers:
        print("average([]) -> None (empty list)")
        return None
    result = sum(numbers) / len(numbers)
    if verbose:
        print(f"average({numbers}) -> {result}")
    return result


def percentage(part, total, verbose=True):
    if total == 0:
        if verbose:
            print(f"percentage({part}, {total}) -> None (division by zero)")
        return None
    result = (part / total) * 100
    if verbose:
        print(f"percentage({part}, {total}) -> {result:.2f}%")
    return result
