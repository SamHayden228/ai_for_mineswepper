import random
import numpy
import re
from torch import tensor





class Node:
    type=int
    bombs_near=0
    opn=False
    flag=False
    start=False
    wrong=False
    been_flaged=False
    flags_near=0
    def __init__(self):
        self.type=int
        self.bombs_near=0


    def reset(self):
        self.opn = False
        self.flag = False
        self.start = False
        self.wrong = False

    def incr(self):
        self.bombs_near+=1

    def decr(self):
        self.bombs_near=max(self.bombs_near-1,0)

    def add_flag(self):
        self.flags_near += 1

    def set_start(self):
        self.start=True
    def unset_start(self):
        self.start=False


    def rem_flag(self):
        self.flags_near=max(self.flags_near-1,0)

    def set_to_bomb(self):
        self.type="bomb"

    def get(self):
        if self.start:
            return "X"
        if self.type==int:
            return str(self.bombs_near)
        return "B"

    def get_nn(self):
        if self.start:
            return 100
        if self.wrong:
            return -9
        if self.flag:
            return 9
        if not(self.opn):
            return 10
        if self.type=="bomb":
            return -1
        if self.bombs_near==0:
            return 0
        return (self.bombs_near)

    def __str__(self):
        if self.start:
            return "X"
        if self.wrong:
            return "⚠️"
        if self.flag:
            return "❗"
        if not(self.opn):
            return "#"
        if self.type=="bomb":
            return "🔴"
        if self.bombs_near==0:
            return "_"
        return str(self.bombs_near)

    def open_node(self):
        self.opn = True
        if self.type=="bomb" and not(self.flag):
            raise Exception(f"explosives")

        if self.bombs_near==0:
            return 0



    def set_flag(self):
        self.flag=True
        self.been_flaged=True
        self.opn=True

    def unset_flag(self):
        if self.type=="bomb":
            self.opn=False
        self.flag=False


class Grid:
    dx = [-1, 0, 1]
    dy = [-1, 0, 1]
    def __init__(self,xsize,ysize,bombs,seed=random.random()):

        self.xsize=xsize
        self.ysize=ysize
        self.savebombs=bombs
        self.bombs=bombs
        self.seed=seed
        self.solved=False
        self.last_action = -1
        random.seed(self.seed)
        self.area=[[Node() for j in range(ysize)] for i in range(xsize)]
        self.startx=random.randint(0,xsize-1)
        self.starty = random.randint(0, ysize-1)
        self.area[self.startx][self.starty].set_start()
        self.fill()
        self.move_bombs(self.startx,self.starty)


    @classmethod
    def replicate(cls, data):
        bombs=0
        obj = cls(data[0],data[1],0,data[3])
        data=data[4:]
        for i in range(len(data)):
            for j in range (len(data[i])):
                if data[i][j]==-1:
                    bombs+=1
                    obj.area[i][j].set_to_bomb()
                elif data[i][j]==9:
                    obj.area[i][j].set_start()
                    obj.startx=i
                    obj.starty=j
                else:
                    obj.area[i][j].bombs_near=data[i][j]
        obj.bombs=bombs
        return obj



    def been_solved(self):
        self.solved=True

    def fill(self):

        for i in range(self.bombs):
            l=random.randint(0,self.xsize-1)
            w=random.randint(0,self.ysize-1)

            while self.area[l][w].type=="bomb":
                l = random.randint(0, self.xsize - 1)
                w = random.randint(0, self.ysize - 1)
            for wi in range(-1,2):
                if not(wi+w<0 or wi+w>=self.ysize):
                    for li in range(-1, 2):
                        if not (li+l < 0 or li+l >= self.xsize):

                            self.area[l+li][w+wi].incr()
            self.area[l][w].set_to_bomb()



    def debug(self):
        res=[]

        for i in self.area:
            res .append([])
            for k in i:
                res[-1].append(k.get())

        return res


    def print_debug(self):
        deb=self.debug()
        c = 0
        p=1
        if self.ysize>=10:
            p=2
        res = "   "
        for i in range(len(deb[0])):
            res += str(i)+("_")+((i+1<10)*(p==2)*"_")
        print(res[:-1])
        for i in deb:
            res = str(c)+((c<10)*" ")+"|"
            for k in i:
                res += str(k) + p*" "
            print(res)
            c += 1


    def get(self):
        res = []

        for i in self.area:
            res.append([])
            for k in i:
                res[-1].append(str(k))

        return res

    def get_nn(self):
        res=[[],[],[],[],[],[]]

        for i in self.area:
            res[0].append([])
            res[1].append([])
            res[2].append([])
            res[3].append([])
            res[4].append([])
            res[5].append([])
            for k in i:
                res[0][-1].append(k.get_nn())
                res[1][-1].append(k.bombs_near if k.opn else 10)
                res[2][-1].append(1 if k.opn else 0)
                res[3][-1].append(1 if k.flag else 0)
                res[4][-1].append(self.bombs)
                res[5][-1].append(self.last_action)
        return tensor(res)

    def get_data(self):
        res=[self.xsize,self.ysize,self.bombs,self.seed,int(self.solved)]
        for i in self.area:
            res.append([])
            for k in i:
                if k.type == "bomb":
                    res[-1].append(-1)
                elif str(k)=="X":
                    res[-1].append(9)
                else:
                    res[-1].append(k.bombs_near)
        return res

    def get_opened_minus_flags(self):
        res = []
        for i in self.area:
            res.append([])
            for k in i:
                if (k.opn):
                    res[-1].append(k.bombs_near-k.flags_near)
                else:
                    res[-1].append(str(k))
        return res

    def reset(self):
        self.unmark_mistakes()
        self.bombs=self.savebombs
        for i in self.area:
            for b in i:
                b.reset()

        self.area[self.startx][self.starty].set_start()

        self.move_bombs(self.startx, self.starty)
        return self.get_nn()

    def __str__(self):
        c = 0
        p = 1
        if self.ysize>=10:
            p=2
        res = "   "
        res = "y> "
        for i in range(len(self.area[0])):
            res += str(i)+("_")+((i<10)*(p==2)*"_")
        res+="\n"
        for i in self.area:
            res+= str(c)+((c<10)*" ")+"|"
            for k in i:
                res += str(k) + p*" "
            res+="\n"
            c += 1
        return res

    def open(self,x,y):
        if self.area[x][y].opn:
            return 3*self.open_opened(x,y)

        t=self.area[x][y].open_node()
        if not(0<=x<self.xsize and 0<=y<self.ysize):
            return

        if self.area[x][y].type=="bomb":

            raise Exception("explosives")
        k=1
        if t==0:
            for i in self.dx:
                if not (x+i < 0 or x+i>= self.xsize):
                    for j in self.dy:
                        if not (y+j < 0 or y+j>= self.ysize) and not(self.check_if_open(x+i,y+j)):
                            k+=self.open(x+i,y+j)
        return k

    def open_opened(self,x,y):
        if (self.area[x][y].bombs_near-self.area[x][y].flags_near):
            return 0
        k=0
        for i in range(-1, 2):
            if not (x+ i < 0 or x + i >= self.xsize):
                for j in range(-1, 2):
                    if not (y+j < 0 or y+j >= self.ysize) and not (self.check_if_open(x + i, y + j)):

                        t=self.area[x+i][y+j].open_node()
                        k=1
                        if t == 0:
                            for ix in self.dx:
                                for jy in self.dy:
                                    if not (x+ i+ix < 0 or x + i+ix >= self.xsize) and not (y+j+jy < 0 or y+j+jy >= self.ysize) and not (self.check_if_open(x + i+ix, y + j+jy)):
                                        k+=self.open(x + i+ix, y + j+jy)
        return k

    def open_solver(self,x,y):


        try:
            t=self.area[x][y].open_node()
        except Exception:

            self.mark_mistakes()

            raise Exception(f"explosives {x} {y}")
        if not(0<=x<self.xsize and 0<=y<self.ysize):
            return

        if self.area[x][y].type=="bomb":

            raise Exception(f"explosives {x} {y}")
        t=self.area[x][y].bombs_near-self.area[x][y].flags_near

        if t==0:
            for i in self.dx:
                if not (x+i < 0 or x+i>= self.xsize):
                    for j in self.dy:
                        if not (y+j < 0 or y+j>= self.ysize) and not(self.check_if_open(x+i,y+j)):
                            self.open_solver(x+i,y+j)
                        if not (y+j < 0 or y+j>= self.ysize) and not(self.area[x+i][y+j].bombs_near-self.area[x+i][y+j].flags_near) and not(self.area[x+i][y+j].flag):
                            self.open_solver2(x+i,y+j)

    def open_solver2(self,x,y):




        if self.area[x][y].type=="bomb":

            raise Exception("explosives")



        for i in self.dx:
            if not (x+i < 0 or x+i>= self.xsize):
                for j in self.dy:
                    if not (y+j < 0 or y+j>= self.ysize) and not(str(self.area[x][y])=="#"):
                        self.area[x][y].open_node()



    def check_if_open(self,x,y):
        if not(0<=x<self.xsize and 0<=y<self.ysize):
            return True
        return self.area[x][y].opn

    def flag(self,x,y):

        if not(self.area[x][y].flag):
            self.bombs-=1
            self.area[x][y].set_flag()
            for i in self.dx:
                for j in self.dy:
                    if not (x+ i < 0 or x + i >= self.xsize) and not (y+j < 0 or y+j >= self.ysize):

                        self.area[x+i][y+j].add_flag()
            return True
        else:
            self.bombs += 1
            self.area[x][y].unset_flag()
            for i in self.dx:
                for j in self.dy:
                    if not (x + i < 0 or x + i >= self.xsize) and not (y + j < 0 or y + j >= self.ysize):
                        self.area[x+i][y+j].rem_flag()
            return False

    def unmark_mistakes(self):
        for i in self.area:
            for node in i:
                if node.type=="bomb":
                    node.opn=False
                elif node.flag and node.type!="bomb":
                    node.wrong=False

    def mark_mistakes(self):
        for i in self.area:
            for node in i:
                if node.type=="bomb":
                    node.opn=True
                elif node.flag and node.type!="bomb":
                    node.wrong=True
    def count(self):
        r=0
        for i in self.get():
            r+=i.count("#")
        return r

    def move_bombs(self,x,y):

        if self.area[x][y].type=="bomb":
            for xi in range(-1, 2):
                if not (x + xi < 0 or x + xi >= self.xsize):
                    for yj in range(-1, 2):
                        if not (y + yj < 0 or y + yj >= self.ysize):
                            self.area[x + xi][y + yj].decr()
            self.area[x][y].type=int
            l = random.randint(0, self.xsize - 1)
            w = random.randint(0, self.ysize - 1)

            while self.area[l][w].type == "bomb" or (x - l) ** 2 + (y - w) ** 2 <= 2:
                l = random.randint(0, self.xsize - 1)
                w = random.randint(0, self.ysize - 1)
            self.area[l][w].type = "bomb"
            for wi in range(-1, 2):
                if not (wi + w < 0 or wi + w >= self.ysize):
                    for li in range(-1, 2):
                        if not (li + l < 0 or li + l >= self.xsize):
                            self.area[l + li][w + wi].incr()
        for i in range(-1,2):
            for j in range(-1,2):
                try:
                    if not (x+i< 0 or x+i >= self.xsize) and not (y+j < 0 or y+j >= self.ysize) and self.area[x + i][y + j].type=="bomb":

                        self.area[x + i][y + j].type=int
                        for xi in range(-1, 2):
                            if not (x+i+xi < 0 or x+i+xi >= self.xsize):
                                for yj in range(-1, 2):
                                    if not (y+j+yj < 0 or y+j+yj >= self.ysize):
                                        self.area[x+i+xi][y+j+yj].decr()
                        l = random.randint(0, self.xsize - 1)
                        w = random.randint(0, self.ysize - 1)

                        while self.area[l][w].type == "bomb" or (x-l)**2+(y-w)**2<=2:
                            l = random.randint(0, self.xsize - 1)
                            w = random.randint(0, self.ysize - 1)
                        self.area[l][w].type="bomb"
                        for li in range(-1, 2):
                            if  not(li + l < 0 or li + l >= self.xsize):

                                for wi in range(-1, 2):
                                    if  not(wi + w < 0 or wi + w >= self.ysize):

                                        self.area[l + li][w + wi].incr()

                except Exception:
                    pass
        self.area[x][y].bombs_near = 0

    def deint_action(self,comm):
        m=self.ysize*self.xsize-1
        res=""
        if comm > m:
            res = "f "
            comm-=m
        x=comm//self.xsize
        y=comm%self.xsize
        return f'{res}{x} {y}'

    def get_random_action(self):
        m = self.ysize * self.xsize
        return random.randint(0,m-1)

    def step(self,action):
        self.last_action=action
        command = self.deint_action(action).split()
        x = int(command[0 + (command[0] == "f")])
        y = int(command[1 + (command[0] == "f")])
        if command[0]=="f":
            try:
                if not(self.area[x][y].been_flaged):
                    self.flag(x, y)
                    return self.get_nn(), 5, 0
            except IndexError:
                return self.get_nn(), 0, 0
        try:

            if x == self.startx and y == self.starty and self.area[x][y].start:
                reward = self.open(x, y)
                self.area[x][y].start=False
                return self.get_nn(), 3*reward+10, 0
            elif self.area[self.startx][self.starty].start:
                return self.get_nn(), -10, 0
            reward = self.open(x, y)
            if reward==0:
                return self.get_nn(),-20,0
            if str(self).count("#")-self.bombs==0:
                self.mark_mistakes()
                return self.get_nn(), 1000, 1
            return self.get_nn(), 3*reward, 0
        except Exception as e:
            self.mark_mistakes()
            return self.get_nn(), -100, -1


    def game(self):
        inp=""
        good=False
        amount=str(self).count("#")-self.bombs
        moves=0
        while not(good):
            if amount==0:
                print(self)
                print("Congrats, you win!")
                quit()

            print(self)
            print(f"f if you want to flag, x, y | amount of bombs: {self.bombs}")
            inp=input().rstrip()


            if not(re.fullmatch("([f][ ][\d]+[ ][\d]+)|([\d]+[ ][\d]+)",inp)):
                print("Incorrect input")
                continue
            command=inp.split()
            if not((0<=int(command[0+(command[0]=="f")])<self.xsize) and (0<=int(command[1+(command[0]=="f")])<self.ysize)):
                print("Incorrect cords")
                continue
            x=int(command[0+(command[0]=="f")])
            y=int(command[1+(command[0]=="f")])
            if inp[0]=="f":
                self.flag(x,y)
            else:
                if not(moves):
                    self.area[self.startx][self.starty].unset_start()
                try:
                    self.open(x,y)
                except Exception as e:
                    self.mark_mistakes()
                    print(self)
                    print("KABOOM")
                    quit()
            amount = str(self).count("#") - self.bombs
            moves+=1


if __name__ == "__main__":
    pass

