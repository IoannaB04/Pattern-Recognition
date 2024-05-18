import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import sqrt

global start
global step
global end

# Plot between 0 and 4 with .001 step
start = 0
step = 0.001
end = 4
x_axis = np.arange(start, end, step)

series_parameters = [[2, sqrt(0.5)],  # p(x|ω1)
                     [1.8, sqrt(0.2)] # p(x|ω2)
                    ]

px1 = norm.pdf(x_axis,series_parameters[0][0], series_parameters[0][1])
px2 = norm.pdf(x_axis,series_parameters[1][0], series_parameters[1][1])
plt.plot(x_axis, px1)
plt.plot(x_axis, px2)
# plt.show()

########################################################################################################################
#  Define the number of the classes
num_classes = 2

# Define a priori probabilities
P1 = 1/4      # Ρ(ω1)
P2 = 1 - P1   # P(ω2)
p = [P1, P2]

# Define cost table
L = [[0, 1], [3, 0] ]

classes=[[] for i in range(num_classes)]
for x in range(0, len(x_axis)):
  R1 = L[0][0] * px1[x]*P1  +  L[0][1] * px2[x]*P2
  R2 = L[1][0] * px1[x]*P1  +  L[1][1] * px2[x]*P2

  if R1 < R2 :
    classes[0].append(x_axis[x])
  else:
    classes[1].append(x_axis[x])


########################################################################################################################
saved4plot = []
limits = []
for c in range(0, len(classes)):
  # c == 0 --> ω1
  # c == 1 --> ω2

  a = []  # λίστα που αποθηκεύω τα όρια των διαστημάτων των x που ανήκουν σε κάθε κλάση
  x_previous = classes[c][0]

  for x in classes[c]:
    if round(x,4) == round(x_previous+step,4):
      x_previous = x
    else:
      a.append(x_previous)
      a.append(x)
      x_previous = x

  a.pop(0)  # Deleting the first element because it is added two times, one at the beginning and one at the end of the first interval
  a.append(classes[c][-1]) # Adding the last element to determing the end of the last interval
  limits.append(a)   # saving the limits of each class in order to calculate the total cost

  print("Στην κλάση ω" + str(c+1) + " ανήκουν τα x που βρίσκονται στο διάστημα:")

  if len(a) > 2:
    for i in range(0, len(a), 2):
      print("[" + str(a[i]) + "," + str(a[i + 1]) + "]")

  else:
    print("[" + str(a[0]) + "," + str(a[1]) + "]")
    saved4plot.append(a[0])
    saved4plot.append(a[1])
plt.plot(x_axis, px1)
plt.plot(x_axis, px2)
for x in range(0, len(saved4plot)):
  plt.axvline(x = saved4plot[x], color = 'black')
# plt.show()

########################################################################################################################
########################################################################################################################

import scipy.special as sp

def SumOfSeries(m, s, lower_limit, upper_limit):

  if upper_limit + step == end+step:    # inf
    f_upper = sp.erf(float("inf"))
  else:
    f_upper = sp.erf( (round(upper_limit,4)-m)/(sqrt(2)*s) )

  if lower_limit +step == start+step:    # - inf
    f_lower = - sp.erf(float("inf"))
  else:
    f_lower = sp.erf( (lower_limit-m)/(sqrt(2)*s) )

  return 0.5 * ( f_upper - f_lower )

def cost_wrong(series_parameter,P, L, limitsA):
  # Προκύπτει από τα δεδομένα της κατανομής που βρίσκονται στην άλλη περιοχή απόφασης. Για αυτό αντιστρέφω την limits

  cost = 0
  L_wrong = [ L[1][0], L[0][1] ] # κρατάω μονάχα τα μη μηδενικά λ με την σειρά που με βολεύει

  limits = limitsA.copy()
  limits.reverse()
  for p in range(0, len(P)):

    if len(limits[p]) > 2:
      a = 0
      for i in range(0, len(limits[p]), 2):
        aa = SumOfSeries(series_parameter[p][0], series_parameter[p][1], limits[p][i], limits[p][i+1])
        a += aa
    else:
      a = SumOfSeries(series_parameter[p][0], series_parameter[p][1], limits[p][0], limits[p][1])

    b = L_wrong[p]
    cost += P[p] * b * a

  return cost

cost = cost_wrong(series_parameters, p, L, limits)
print("Total cost is "+str(cost))




samples = 1000
w1 = np.random.normal(series_parameters[0][0], series_parameters[0][1],samples)
w2 = np.random.normal(series_parameters[1][0], series_parameters[1][1], samples)
del saved4plot, P1, P2, R1, R2, a, c, classes, start, end, step, num_classes, sqrt, x_previous, norm, series_parameters
del cost, i,


w1_wrong = 0
w1_right = 0
w2_wrong = 0
w2_right = 0

for point in w1:
  if point >= 1.076 and point <= 2.258:
    w1_wrong += 1
  else:
    w1_right += 1

w1_right = w1_right / samples
w1_wrong = w1_wrong / samples
costw1 = p[0] * ( L[0][0]*w1_right + L[1][0]*w1_wrong )

print("\nData from ω1 that classified in ω1: " + str(w1_right) )
print("Data from ω1 that classified in ω2: " + str(w1_wrong) )
print("Total cost in class ω1: " + str(costw1) )

for point in w2:
  if point >= 1.076 and point <= 2.258:
    w2_right += 1
  else:
    w2_wrong += 1

w2_right = w2_right / samples
w2_wrong = w2_wrong / samples
costw2 = p[1] * ( L[1][1]*w2_right + L[0][1]*w2_wrong )

print("\nData from ω2 that classified in ω1: " + str(w2_wrong) )
print("Data from ω2 that classified in ω2: " + str(w2_right) )
print("Total cost in class ω2: " + str(costw2) )

print("\nTotal cost is: " + str(costw1+costw2) )
