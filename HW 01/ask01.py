D = [[0.5, 0.05, 0.02],
     [0.23, 0.15, 0.13],
     [0.16, 0.4, 0.15],
     [0.1, 0.3, 0.3],
     [0.01, 0.1, 0.4]
    ]

ps = 0.3
pm = 0.1
pn = 1 - ps - pm  # το άθροισμα των πιθανοτήτων είναι 1
P = [pn, ps, pm]
del pn, ps, pm


classes = []
error = []

for d in D:
    evidence = 0
    posteriors = []
    numberators = []

    # Calculating posteriors of each d
    for i in range(0, len(d)):
        evidence = evidence + d[i]*P[i]
        numberators.append(d[i]*P[i])

    for val in numberators:
        posteriors.append(val/evidence)

    # Determining the class of each d according to the maximum posterior
    val = max(posteriors)
    val_index = list.index(posteriors, val)
    classes.append(val_index)
    # 0 --> Normal      1 --> Spam      2 --> Malicious

    # Calculating the error
    error.append(evidence*(1-val))

    del i, val, val_index

# Printing the results
counter = 1
for c in classes:
    match c:
        case 0:
            print("D" + str(counter) + " email belongs to category NORMAL.")
        case 1:
            print("D" + str(counter) + " email belongs to category SPAM.")
        case 2:
            print("D" + str(counter) + " email belongs to category MALICIUS.")

    counter += 1

# Calculating the error
print()
print("Total error is: " + str(sum(error)))


classes = []
error_c = []

for d in D:
    evidence = 0
    posteriors = []
    numberators = []

    # Calculating posteriors of each d
    for i in range(0, len(d)):
        evidence = evidence + d[i]*P[i]
        numberators.append(d[i]*P[i])   # γινόμενο της πιθανοφάνειας με την a priori πιθανότητα

    for val in numberators:
        posteriors.append(val/evidence)

    # Determining the class of each d according to the index of the maximum posterior
    classes.append(list.index(posteriors, max(posteriors)))
    # 0 --> Normal    1 --> Spam    2 --> Malicious

    # Calculating the error
    error_c.append(evidence*(1-max(posteriors)))


# Printing the results    !!! python version >= 3.8.10 !!!
counter = 1
for c in classes:
    match c:
        case 0:
            print("D" + str(counter) + " email belongs to category NORMAL.")
        case 1:
            print("D" + str(counter) + " email belongs to category SPAM.")
        case 2:
            print("D" + str(counter) + " email belongs to category MALICIUS.")
    counter += 1

# Calculating the total error
print("\nNew total error is: " + str(sum(error_c)))

print("\nΤα δύο σφάλματα διαφέρουν κατά " + str(sum(error_c)-sum(error)))