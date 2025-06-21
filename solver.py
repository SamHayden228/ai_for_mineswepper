import time
from collections import defaultdict
import numpy as np
import copy
import mineswepper_the_game



def solver(grid,times,steps):
	mk=grid.get_opened_minus_flags()
	no=['#',"❗","0"]

	if grid.bombs < 10:
		res=[grid.bombs]
		eqs = [[]]
	else:
		eqs = []
		res=[]


	seen=defaultdict(bool)
	place= {}

	msize=0
	for i in range(grid.xsize):
		for j in range(grid.ysize):
			if ((mk[i][j])==0) and not(grid.area[i][j].flag):


				grid.open_solver(i,j)
	for i in range(grid.xsize):
		for j in range(grid.ysize):
			d = f'{i}:{j}'
			if ((mk[i][j])==0) and not(grid.area[i][j].flag):

				grid.open_solver(i,j)
			elif grid.bombs<10 and mk[i][j]=="#" and not(seen[d]):
				place[d]=len(eqs[0])
				eqs[0].append(1)
				seen[d]=True
			elif not(no.count(str(mk[i][j]))):
				if grid.bombs<10:
					eqs.append([0]*len(eqs[0]))
				else:
					if len(eqs)==0:
						eqs.append([])
					else:
						eqs.append([0] * len(eqs[-1]))
				for xi in range(-1,2):
					if (0<=i+xi<grid.xsize):
						for yj in range(-1,2):
							if (0<=j+yj<grid.ysize) and mk[i+xi][j+yj]=="#":
								c=f'{i+xi}:{j+yj}'
								if not(seen[c]):
									seen[c]=True
									place[c]=len(eqs[-1])
									eqs[-1].append(1)
									if grid.bombs<10:
										eqs[0].append(1)
								else:

									eqs[-1][place[c]]=1
				res.append(mk[i][j])

	for i in eqs:
		for j in range(len(eqs[-1])-len(i)):
			i.append(0)



	for i in list(place.keys()):
		c = i.split(":")
	if grid.bombs < 10:
		for i in eqs:
			for j in range(len(eqs[0])-len(i)):
				i.append(0)
	else:
		for i in eqs:
			for j in range(msize-len(i)):
				i.append(0)

	eqs=np.array(eqs)
	res=np.array(res)


	flag=[]
	open=[]
	a=list(np.linalg.lstsq(eqs,res,)[0])

	reconstructed = eqs @ a
	residual = np.linalg.norm(res - reconstructed)
	spread = max(a) - min(a)
	complexity = len(res) + 1

	# комбинированное динамическое ε
	epsilon = max(0.01, min(0.1, residual / complexity + spread / 20))

	for i in range(len(a)):
		val = float(a[i])
		key = list(place.keys())[i]

		if abs(val - 1) < epsilon:
			flag.append(key)
		elif abs(val) < epsilon:
			open.append(key)



	s = set()
	vars= {}
	assigned=defaultdict(int)


	for i in flag:
		c = i.split(":")
		steps[0] += 1
		grid.flag(int(c[0]),int(c[1]))
	try:
		for i in open:
			c = i.split(":")
			steps[0] += 1

			grid.open_solver(int(c[0]), int(c[1]))
	except Exception:

		raise Exception
	return flag,open



		# if grid.area[int(c[0])][int(c[1])].bombs_near-grid.area[int(c[0])][int(c[1])].flags_near!=0:
		# 	grid.open_opened(int(c[0]), int(c[1]))

def get_route_solver(grid):

	no = ['#', "❗", "0"]
	route=[]
	if grid.bombs < 10:
		res = [grid.bombs]
		eqs = [[]]
	else:
		eqs = []
		res = []

	seen = defaultdict(bool)
	place = {}
	grid2 = copy.deepcopy(grid)
	if not(grid.check_if_open(grid.startx,grid.starty)):

		route.append(grid.startx*grid.ysize+grid.starty)
		grid2.open(grid.startx,grid.starty)

	msize = 0

	mk = grid2.get_opened_minus_flags()
	for x in range(grid2.xsize):
		for y in range(grid2.ysize):
			act=(x*grid2.ysize+y)

			if ((mk[x][y]) == 0) and not (grid2.area[x][y].flag):

				rr=grid2.open(x,y)

				if rr>0:
					route.append(act)

	mk = grid2.get_opened_minus_flags()

	for i in range(grid.xsize):
		for j in range(grid.ysize):
			d = f'{i}:{j}'

			if grid.bombs < 10 and mk[i][j] == "#" and not (seen[d]):
				place[d] = len(eqs[0])
				eqs[0].append(1)
				seen[d] = True
			elif not (no.count(str(mk[i][j]))):
				if grid.bombs < 10:
					eqs.append([0] * len(eqs[0]))
				else:
					if len(eqs) == 0:
						eqs.append([])
					else:
						eqs.append([0] * len(eqs[-1]))
				for xi in range(-1, 2):
					if (0 <= i + xi < grid.xsize):
						for yj in range(-1, 2):
							if (0 <= j + yj < grid.ysize) and mk[i + xi][j + yj] == "#":
								c = f'{i + xi}:{j + yj}'
								if not (seen[c]):
									seen[c] = True
									place[c] = len(eqs[-1])
									eqs[-1].append(1)
									if grid.bombs < 10:
										eqs[0].append(1)
								else:

									eqs[-1][place[c]] = 1
				res.append(mk[i][j])

	for i in eqs:
		for j in range(len(eqs[-1]) - len(i)):
			i.append(0)

	for i in list(place.keys()):
		c = i.split(":")
	if grid.bombs < 10:
		for i in eqs:
			for j in range(len(eqs[0]) - len(i)):
				i.append(0)
	else:
		for i in eqs:
			for j in range(msize - len(i)):
				i.append(0)

	eqs = np.array(eqs)
	res = np.array(res)

	flag = []
	open = []
	a = list(np.linalg.lstsq(eqs, res)[0])
	if len(a)==0:

		return route
	reconstructed = eqs @ a
	residual = np.linalg.norm(res - reconstructed)
	spread = max(a) - min(a)
	complexity = len(res) + 1

	# комбинированное динамическое ε
	epsilon = max(0.01, min(0.1, residual / complexity + spread / 20))

	for i in range(len(a)):
		val = float(a[i])
		key = list(place.keys())[i]

		if abs(val - 1) < epsilon:
			flag.append(key)
		elif abs(val) < epsilon:
			open.append(key)

	s = set()
	vars = {}
	assigned = defaultdict(int)

	for i in flag:
		c = i.split(":")

		route.append((int(c[0])*grid.ysize+int(c[1]))+grid.ysize*grid.xsize-1)

	try:
		for i in open:

			c = i.split(":")

			route.append((int(c[0])*grid.ysize+int(c[1])))

	except Exception:

		raise Exception
	return route

def solve(grid,steps=[0]):
	steps[0]=0
	c=1
	grid.open(grid.startx,grid.starty)

	try:

		while True:
			print(grid)
			time.sleep(2.5)
			flag,open =solver(grid,c,steps)
			print(grid)
			time.sleep(2.5)
			amount = str(grid).count("#") - grid.bombs

			if len(flag)==0 and len(open)==0:
				c+=1
			else:
				c=1
			if amount ==0:

				return 0

			if c==10:

				return 1
	except Exception:

		return 1












if __name__ == "__main__":
	# Given input
	N = 30
	M = 30
	bombs=200
	grid=mineswepper_the_game.Grid(N,M,bombs)

	steps=[0]
	#0.1384367715993029
	print(solve(grid,steps))
	print(steps[0])






	# Function call to perform generate and solve a minesweeper

