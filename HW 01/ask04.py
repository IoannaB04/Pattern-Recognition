import numpy as np
from matplotlib import pyplot as plt
import sympy as sp

def f(x):
  return x**2 * (1-x)


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

x = np.linspace(0, 1, 1000)

plt.plot(x, f(x), color='purple')
plt.show()

def P_error_ideal():
  x = np.linspace(0, 1, 1000)
  max_error = max(f(x))
  index = np.argmax(f(x))

  print("The maximum error is " + str(max_error) + " which occurs for x = " + str(x[index]))
  return x[index], max_error


def derivatedat(value):
  x = sp.symbols('x')
  df = sp.diff(f(x), x)
  a = df.subs(x, value).evalf()
  return a

def P_error(p, p_true):
  p_ideal, max_error = P_error_ideal()

  if round(p_ideal, 4) == p:
    print("Ideal")
    return max_error
  else:
    x = sp.symbols('x')
    df = sp.diff(f(x), x)
    a = df.subs(x, p).evalf()
    b = f(p) - a * p

    print("The tangent's equation is: y = " +str(round(a,4)) + "x" + str(round(b,4)))
    return a * p_true + b

p_estimated = 0.3
p_true = 0.7
error = P_error(p_estimated, p_true)
print("Error for estimated priority " + str(p_estimated)+ " when the truth is " + str(p_true) + ": " + str(error))
