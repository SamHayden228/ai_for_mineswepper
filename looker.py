import csv
from random import randint

from PIL.ImagePalette import random

from mineswepper_the_game import Grid
import numpy as np
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import random

def graph():
    bombs=[]
    nums=dict()
    strt=[]
    step={}
    def get_sum(mas):
        r=0
        for i in mas:
            for j in i:
                r+=j
        return r
    m=0
    with open("maps.csv") as file:
        mapreader=list(csv.reader(file))
        i=0
        c=0
        st=600
        en=900

        while i<(len(mapreader)) and c<min(en-st,300):

            if len(mapreader[i])==2 and st<int(mapreader[i][1])<=en:
                if len(bombs)==0:
                    bombs=[[0 for j in range(int(mapreader[i+1][1]))] for p in range(int(mapreader[i+1][0]))]
                if len(strt)==0:
                    strt=[[0 for j in range(int(mapreader[i+1][1]))] for p in range(int(mapreader[i+1][0]))]
                for j in range(int(mapreader[i+1][0])):
                    for k in range(len(mapreader[j+i+2])):

                        bombs[j][k]+=int(mapreader[j+i+2][k]=="-1")
                        strt[j][k] += int(mapreader[j + i + 2][k] == "9")
                        if 9>int(mapreader[j+i+2][k])>=0:
                            if list(nums.keys()).count(int(mapreader[j+i+2][k])):
                                nums[int(mapreader[j+i+2][k])]+=1
                            else:
                                nums[int(mapreader[j + i + 2][k])] = 1

                if int(mapreader[i+1][4]) and list(step.keys()).count(int(mapreader[i+1][5])):
                    step[int(mapreader[i+1][5])] += 1
                elif int(mapreader[i+1][4]):
                    step[int(mapreader[i+1][5])] = 1
                i+=int(mapreader[i+1][0])+3
                c+=1
            else:
                i+=1

    print(get_sum(strt))
    colors_list = ['#4366e6',"#d8e643",'#e64343']
    cmap = colors.ListedColormap(colors_list)

    # plt.figure()
    # hm = sns.heatmap(data=bombs,
    #                  annot=True)

    plt.figure()
    plt.title("")

    strtg = sns.heatmap(data=strt,
                     annot=True)



    # # displaying the plotted heatmap
    # plt.figure()
    # bc=sns.barplot(nums)
    #
    #
    # plt.figure()
    # bc2=sns.barplot(step)

    plt.show()

def get_random_map(type,solved):
    type=["small","medium","big"].index(type)
    diap_start = 300*type+200* int(not(solved))
    diap_end =  300 * (type+1) - 100 *solved
    map_num=random.randint(diap_start+1,diap_end)

    print(map_num,"НОМЕР КАРТЫ")
    with open("maps.csv") as file:
        mapreader = list(csv.reader(file))
        i = 0


        while i < (len(mapreader)) :

            if len(mapreader[i]) == 2 :
                if int(mapreader[i][1])==map_num:
                    grid=Grid(int(mapreader[i+1][0]),int(mapreader[i+1][1]),int(mapreader[i+1][2]),float(mapreader[i+1][3]))
                    break
                i += int(mapreader[i + 1][0]) + 3

            else:
                i += 1

    return grid


def get_map(map_num):
    with open("maps.csv") as file:
        mapreader = list(csv.reader(file))
        i = 0


        while i < (len(mapreader)) :

            if len(mapreader[i]) == 2 :
                if int(mapreader[i][1])==map_num:
                    grid=Grid(int(mapreader[i+1][0]),int(mapreader[i+1][1]),int(mapreader[i+1][2]),float(mapreader[i+1][3]))
                    break
                i += int(mapreader[i + 1][0]) + 3

            else:
                i += 1
    return grid

