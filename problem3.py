import numpy as np
import matplotlib.pyplot as plt

w = np.ones((1,2))
b = 1
inp = [{'p':[], 't':-1} for i in range(8)]

def init():
	inp[0]['p'] = np.array([[1],[4]])
	inp[0]['t'] = 0
	inp[1]['p'] = np.array([[1],[5]])
	inp[1]['t'] = 0
	inp[2]['p'] = np.array([[2],[4]])
	inp[2]['t'] = 0
	inp[3]['p'] = np.array([[2],[5]])
	inp[3]['t'] = 0
	inp[4]['p'] = np.array([[3],[1]])
	inp[4]['t'] = 1
	inp[5]['p'] = np.array([[3],[2]])
	inp[5]['t'] = 1
	inp[6]['p'] = np.array([[4],[1]])
	inp[6]['t'] = 1
	inp[7]['p'] = np.array([[4],[2]])
	inp[7]['t'] = 1

			
def plot():
	for i in range(len(inp)):
		if(inp[i]['t']==0):
			plt.plot(inp[i]['p'][0][0], inp[i]['p'][1][0], 'k*')
		else:
			plt.plot(inp[i]['p'][0][0], inp[i]['p'][1][0], 'go')
	x = np.linspace(-2,8)
	y = (-(b/w[0][1])/(b/w[0][0]))*x - b/w[0][1] 
	plt.plot(x, y)
	plt.plot(0,-b/w[0][1],'b^')
	plt.plot(-b/w[0][0],0,'b^')
	plt.axis([-2, 6, -2, 6])
	plt.show()

def hardlim(x):
	if (x>=0):
		return 1
	else:
		return 0
	
def train():
	global b
	global w
	ite = 0
	
	while(1):
		w_prev = w
		for i in range(len(inp)):
			a = hardlim(w.dot(inp[i]['p']) + b)
			s = inp[i]['t'] - a
			b = b + s
			plott = np.array([inp[i]['p'][0][0], inp[i]['p'][1][0]])
			w = w + plott.dot(s)
			ite += 1
		if(np.array_equal(w_prev, w)): break
	print("b(",ite,") =",b)
	print("w(",ite,") =",w)

def add():
	p1 = {}
	p1['p'] = np.array([[1],[3]])
	p1['t'] = 0
	inp.append(p1)
	
	p2 = {}
	p2['p'] = np.array([[5],[3]])
	p2['t'] = 1
	inp.append(p2)
	
	
def main():
	init()
	add()
	train()
	plot()
	
main()
