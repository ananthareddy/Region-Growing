import numpy as np
import random as rn
import time
from Function_Eval import feval


def EFO(Positions, fobj, Lb, Ub, Max_iter, val_in,val_tar):
    N, dim = Positions.shape[0], Positions.shape[1]
    ub = Ub[1, :]
    lb = Lb[1, :]

    sense_area = 0.5
    Amplitude = np.random.rand(N, dim)
    Distance = np.zeros(N)
    Probability = np.zeros(N)

    Convergence_curve = np.zeros((Max_iter, dim))
    Fitness = feval(fobj, Positions, val_in,val_tar)

    frequency = np.zeros(Fitness.shape)
    r = np.zeros(Fitness.shape)

    fworst = max(Fitness)
    fbest = min(Fitness)

    magnitude = np.linalg.norm(Positions)
    for i in range(N):
        val = ((fworst - Fitness[i, 0]) / (fworst - fbest))
        frequency[i] = fworst + val[0] * (fbest - fworst)
        Amplitude[i, :] = magnitude * Amplitude[i, :] + (1 - magnitude) * Fitness[i, 0]

    l = 0
    ct = time.time()

    while l < Max_iter:
        print(l)
        for i in range(N):
            sum = 0
            if frequency[i] >= rn.random():
                r[i] = (fbest - fworst) * Amplitude[i, :]
                for j in range(dim - 1):
                    sum = sum + Positions[i, j] - Positions[1 + 1, j]
                Distance[i] = sum
                sum = 0
                S = np.where(Distance < sense_area)
                if S != 0:
                    k = rn.randint(-1, 9)
                    Positions[i, :] = Positions[i, :] + rn.uniform(-1, 1) * (Positions[k, :] - Positions[i, :])
                else:
                    Positions[i, :] = Positions[i, :] + rn.uniform(-1, 1) * (r[i])
            else:
                for j in range(N):
                    numer = Amplitude[j] / Distance[j]
                    denom = np.sum(Amplitude / Distance)
                    Probability[j] = round(numer / denom)
                    snum = 0
                    sden = 0
                    for k in range(Probability[j]):
                        snum = snum + Amplitude[k] / Positions[:, k]
                        sden = sden + Amplitude[k]
                    Positions[i, :] = snum / sden
                    new = Positions[i, :] + rn.uniform(-1, 1) * (Positions[j, :] - Positions[i, :])
                    if Fitness[i, 0] < rn.random():
                        Positions[i, :] = new

        bestsol = Positions[0, :]
        Convergence_curve[l] = fbest
        l = l + 1

    best_fit = Convergence_curve[Max_iter - 1]
    ct = time.time() - ct
    return best_fit, Convergence_curve, bestsol, ct
