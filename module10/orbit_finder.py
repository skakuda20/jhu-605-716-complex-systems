from fractions import Fraction

def tent_map(x):
    if x <= Fraction(1, 2):
        return 2 * x
    else:
        return 2 - 2 * x

def tent_map_float(x):
    if x <= 0.5:
        return 2 * x
    else:
        return 2 - 2 * x

def iterate(f, x0, n):
    x = x0
    for _ in range(n):
        x = f(x)
    return x

def iterate_range(f, x0, start, end):
    x = iterate(f, x0, start)
    results = [x]
    for _ in range(end - start):
        x = f(x)
        results.append(x)
    return results

# Initial conditions
x0_a = Fraction(2, 5)
x0_b = Fraction(2, 9)


for x0 in [x0_a, x0_b]:
    print(f"\nInitial condition  x₀ = {x0}  ({float(x0):.6f})")
    float_vals = iterate_range(tent_map_float, float(x0), 1001, 1005)
    for i, val in enumerate(float_vals):
        n = 1001 + i
        print(f"x_{n:<4}  {val:.15f}")


        