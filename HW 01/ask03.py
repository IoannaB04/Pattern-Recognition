import numpy as np
from math import pi, log
import pandas as pd

# Ερώτημα Α

def DiscriminationFunction(d, x, m, sigma, prior):
    x_m = x - m

    if d>1:
        Det_sigma = round(abs(np.linalg.det(s)),4) # rounding is needed cause linalg.det sometimes returns floating points with 1 bit rounding error
        inv_sigma = np.linalg.inv(sigma)
        g = -0.5 * (x_m.T).dot(inv_sigma).dot(x_m) - (d/2) * np.log(2*pi) - 0.5 * np.log(Det_sigma) + np.log(prior)
    else:
        inv_sigma = sigma**-1
        g = -0.5 * x_m**2 * inv_sigma - (d/2)*log(2*pi) - 0.5*log(abs(sigma)) + np.log(prior)
    return g

def EucleidianDistance(d, x1, x2):
    if d == 1:
        distanace = np.sqrt(np.sum((np.square(x1-x2))))
    else:
        distanace = np.sqrt(np.sum((np.square(x1-x2).T.dot(x1-x2) )))
    return distanace

def ΜahalanobisDistanace(d, m, sigma, x):
    x_m = x - m
    if d>1:
        inv_sigma = np.linalg.inv(sigma)
        distance = np.sqrt(x_m.T.dot(inv_sigma).dot(x_m))
    else:
        inv_sigma = sigma**-1
        distance = np.sqrt(x_m*inv_sigma*x_m)
    return distance

########################################################################################################################
# Ερώτημα Β

data = pd.read_csv('data.csv', usecols=[0,1,2])
c = np.transpose(pd.read_csv('data.csv', usecols=[3]).values)
df = np.transpose(data.values)

def maximum_likelihood_estimation(data):
    mean = np.mean(data, axis=0)
    covariance_matrix = np.cov(data, rowvar=False)
    return mean, covariance_matrix


# Ερώτημα Β1
print("----- Ερώτημα Β1 -----")
classes_x1 = [ [], [], []]

for i in range(len(df[0])):
  if c[0][i] == 1:
    classes_x1[0].append([df[0][i]])
  elif c[0][i] == 2:
    classes_x1[1].append([df[0][i]])
  else:
    classes_x1[2].append([df[0][i]])

for i in range (0,len(classes_x1)):
  m, s = maximum_likelihood_estimation(classes_x1[i])
  print("Kλάση "+str(i+1)+": μ = "+str(round(m[0],4))+" \tκαι\tΣ = "+str(s))


# Ερώτημα Β2
print("\n----- Ερώτημα Β2 -----")
classes_x1x2 = [ [], [], []]

for i in range(len(df[0])):
  if c[0][i] == 1:
    classes_x1x2[0].append([df[0][i], df[1][i]])
  elif c[0][i] == 2:
    classes_x1x2[1].append([df[0][i], df[1][i]])
  else:
    classes_x1x2[2].append([df[0][i], df[1][i]])


for i in range (0,len(classes_x1x2)):
  m, s = maximum_likelihood_estimation(classes_x1x2[i])
  print("Kλάση "+str(i+1)+": μ = "+str(np.round(m,4))+" \tκαι\tΣ = "+str(np.round(s[0],4)))
  print("\t\t\t\t\t    "+str(np.round(s[1],4))+'\n')

# Ερώτημα Β3
print("----- Ερώτημα Β3 -----")
classes_x1x2x3 = [ [], [], []]

for i in range(len(df[0])):
  if c[0][i] == 1:
    classes_x1x2x3[0].append([df[0][i], df[1][i], df[2][i]])
  elif c[0][i] == 2:
    classes_x1x2x3[1].append([df[0][i], df[1][i], df[2][i]])
  else:
    classes_x1x2x3[2].append([df[0][i], df[1][i], df[2][i]])

for i in range (0,len(classes_x1x2x3)):
  m, s = maximum_likelihood_estimation(classes_x1x2x3[i])
  print("Kλάση "+str(i+1)+": μ = "+str(np.round(m,4))+" \tκαι\tΣ = "+str(np.round(s[0],4)))
  print("\t\t\t\t\t\t    "+str(np.round(s[1],4)))
  print("\t\t\t\t\t\t    "+str(np.round(s[2],4))+"\n")

########################################################################################################################
# Ερώτημα Γ
print("----- Ερώτημα Γ -----")

def classifier(d, classes, priors):
    m_x = []
    s_x = []
    wrong = 0

    for y in range(0, len(classes)):
        m, s = maximum_likelihood_estimation(classes[y])
        m_x.append(m)  # Need to keep these values in order to calculate the discrimination function below
        s_x.append(s.tolist())


    for cc in range(0, len(classes)):
        for i in range(len(classes[cc])):
            # Wrong classification for features belonging to classI --> Occurs when gI is not minimum
            g = []
            for y in range(0, len(classes)):
                gi = DiscriminationFunction(d, classes[cc][i], m_x[y], s_x[y], priors[y])
                g.append(gi)

            if max(g) != g[cc]:
                wrong += 1

    total = sum(len(v) for v in classes)
    return wrong / total

priors = [0.5 , 0.5]
classes_x1c = classes_x1.copy()
classes_x1c.pop()  # Deleting third class ω3

error_1 = classifier(1, classes_x1c, priors)
print("Classification error using only feature x1: "+str(error_1))
del classes_x1c

# Ερώτημα Δ
print("\n----- Ερώτημα Δ -----")
classes_x1x2d = classes_x1x2.copy()
classes_x1x2d.pop() # Deleting third class ω3

error_2 = classifier(2, classes_x1x2d, priors)
print("Classification error using only features x1 and x2: "+str(error_2))
del classes_x1x2d

classes_x1x2x3d = classes_x1x2x3.copy()
classes_x1x2x3d.pop()   # Deleting third class ω3

error_3 = classifier(3, classes_x1x2x3d, priors)
print("Classification error using all features x1, x2 and x3: " + str(error_3))
del classes_x1x2x3d



# Ερώτημα Ε
print("\n----- Ερώτημα Ε -----")
priors = [0.8, 0.1, 0.1]

error_4 = classifier(3, classes_x1x2x3, priors)
print("Classification error using all features x1, x2 and x3: " + str(error_4))

del i, log, pi,  data, df
print()
