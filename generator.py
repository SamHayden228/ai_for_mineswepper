import numpy as np

from mineswepper_the_game import Grid
import solver
import torch
import numpy as np
import random
import csv
from solver import solve

N=[9,16,16,20]
M=[9,16,30,30]
bombs=[10,40,100,130]

сount=0
b=bombs


for l in range(3):
    with open('maps.csv', "a", newline='') as csvfile:
        for i in range(200):
            steps=[0]
            сount+=1
            print(сount)
            t=1
            while t==1:
                grid1 = Grid(N[l], M[l], bombs[l],random.random())

                t=solve(grid1,steps)

            grid1.been_solved()

            data=grid1.get_data()

            mapwriter = csv.writer(csvfile)
            mapwriter.writerow(["#map",сount])
            mapwriter.writerow([data[0], data[1], bombs[l], data[3], data[4],steps[0]])
            for i in range(data[0]):
                mapwriter.writerow(data[i + 5])
            mapwriter.writerow(["#map"])

        for i in range(100):
            сount+=1
            print(сount)
            grid2 = Grid(N[l], M[l], bombs[l],random.random())
            data = grid2.get_data()


            mapwriter = csv.writer(csvfile)
            mapwriter.writerow(["#map",сount])
            mapwriter.writerow([data[0], data[1], bombs[l], data[3], data[4],0])
            for i in range(data[0]):
                mapwriter.writerow(data[i + 5])
            mapwriter.writerow(["#map"])


print()