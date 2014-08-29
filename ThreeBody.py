# Restricted three-body Poincare Mapper
# Alex Rowe
# University of California, Santa Cruz, 2014
# Python 3.4

from numpy import *
from itertools import product
from multiprocessing import pool
from scipy import linspace

import scipy.integrate as scp
import matplotlib.pyplot as plt
import time, sys

m_s = .999001 # Mass of the Sun
m_j = .000999 # Mass of Jupiter

r_s = -m_j    # Position of Sun
r_j = m_s	  # Position of Jupiter

w_j = 1       # Angular speed of Jupiter

orbits = 100  # Number of orbits to run the simulation for

tol = 1e-12	  # Error Tolerance

pools = 6     # Maximum number of threads to use

filename = 'lagrange.dat'  # File to either contnue or start

writemode = 'wb' # Default writes to a new file

if len(sys.argv) > 1:
	filename = sys.argv[1]

if len(sys.argv) > 2:
	orbits = int(sys.argv[2])

if len(sys.argv) > 3:
	pools = int(sys.argv[3])


def main():

	x0 = [.55]

	cj = [3.07]

	init = product(x0,cj)

	# trace(X(0.5,3.07))

	p = pool.Pool(pools)
	p.map(orbitCalc, init)

	# continueCalc will append orbits to a data file already started
	#p.map(continueCalc, init)


def continueCalc():
	global writemode
	writemode = 'ab'

	print("Continuing calculation for",filename)
	print("Adding",orbits,"orbits")

	data = loadtxt(filename)

	x0 = data[-1][5]
	cj = data[-1][6]

	init_vectors = list()
	counted = list()
	mx=0
	for row in reversed(data):
		if x0 >= mx:
			mx=x0
		x0 = row[5]
		cj = row[6]
		if not (x0,cj) in counted:
			counted.append((x0,cj))
			init_vectors.append(row)
	return init_vectors


# Calculates a particular initial condition and saves it to a file
# init - (x0, Cj) a tuple containing the initial starting x and Cj value
# init - x optionally use a starting vector with x0 and cj at end of vector
# init - (x0, Cj, x) optionally use an initial vector at end of tuple
# orbits - the rough number of orbits to calculate
# tolerance - integrator tolerance
def orbitCalc(init):
	if(len(init)==7):
		x = init[0:5]
		x0 = init[5]
		cj = init[6]

	if(len(init)==2):
		x0 = init[0]
		cj = init[1]
		x = X( x0, cj )

	if len(init)==3:
		x0 = init[0]
		cj = init[1]
		xv = init[2]
		x= X(x0,cj,xv)
		x0 = xv 

	if len(init)==5:
		x = init
		x0 = init[0]
		cj = init[1]

	o_time = orbitTimer(x) # time the average orbit 

	h = o_time/10 #cuts the orbit into 10 pieces (minimum should be 4 or 5 but 10 is safe)
	T = orbits*o_time ## final time

	print('Calculating x0=',x0,', Cj=',cj)

	t = time.time() # time the process 

	x_mat = ode(x,h,T) # magic happens here

	print('Counting orbits...')

	orbit_mat = locateCrossing(x_mat)

	#orbit_mat = vstack([x_mat[0],orbit_mat])

	print(len(orbit_mat)," orbits found")
	print("Took ",time.time()-t,"seconds")

	orbit_mat = col_append( orbit_mat, x0 )
	orbit_mat = col_append( orbit_mat, cj )

	f = open(filename, 'wb')
	savetxt(f, orbit_mat, delimiter=' ')
	f.close()

	return orbit_mat


# Calculates Cj for debugging purposes
def Cj(x):

	ds = dist([ x[0],x[1] ], [ r_s, 0 ])
	dj = dist([ x[0],x[1] ], [ r_j, 0 ])


	C = 2 * (m_s/ds+m_j/dj)
	C += x[0]**2 + x[1]**2
	C += -(x[2]**2+x[3]**2)

	return C

# Solves for a particular y-velocity and Cj
def ydot(x0, Cj, xd=0):

	r1 = x0-r_s #distance from sun
	r3 = r_j-x0 # distance from jupiter

	T= 2*pi/w_j
	n = 2*pi/T


	Yd = sqrt( - Cj 
		+ n**2 * x0**2 
		+ 2*(m_s/r1 + m_j/r3) 
		- xd**2)


	return Yd

def find_max(Cj=3.07):

	xd = -1
	x0=.9

	while xd < 0:
		xp=x0
		r1 = xp-r_s #distance from sun
		r3 = r_j-xp # distance from jupiter

		xd = - Cj + xp**2 + 2*(m_s/r1 + m_j/r3) 

		x0*=.999999

	xd=sqrt(xd)
	print(xd)
	return xp


# This sets up an initial vector
# x: inital x value
# Cj: Cj constant 
# vy: use if you want to specify a specific vy 
def X(x0, Cj=3.07, vx=0):
	y=0
	vy=ydot(x0,Cj,vx)

	return array([x0,y,vx,vy,0])

# This is a shortcut for the ode call
# x - vector
# h - stepsize
# final_time - time to stop at 
# tol - Error
def ode(x, h, final_time):
	space = linspace(0,final_time,final_time/h)
	mat = scp.odeint(f, x, space, atol=tol, rtol = tol*2)
	return mat

# method called by the integrator
def f(x,t):
	X  = x[0]
	Y  = x[1]
	VX = x[2]
	VY = x[3]

	ds = dist([ X,Y ], [ r_s, 0 ])
	dj = dist([ X,Y ], [ r_j, 0 ])

	grav_j = m_j / dj**3
	grav_s = m_s / ds**3

	return(array([
		VX, # velocity body1
		VY,

		  grav_s * (r_s-X) # x-gravity from sun
		+ grav_j * (r_j-X) # x-gravity from jupiter
		+ 2*w_j*VY         # y-coriolis force
		+ w_j**2*X         # y-centrifugal force
		,

		  grav_s * (0-Y) # y-gravity from sun
		+ grav_j * (0-Y) # y-gravity from jupiter
		+ -2*w_j*VX      # y-coriolis force
		+ w_j**2*Y ,     # y-centrifugal force
		1                # time
		]))


# Locates all crossings of the y axis
# requires a positively oriented rotation (ccw)
def locateCrossing(x_mat, precision = 1e-6):

	def locate(tup):
		v1 = tup[0]
		v2 = tup[1]

		if v1[1] == 0:
			return v1

		if v1[1] < 0 and v2[1] > 0 and v1[3] > 0 and v2[3] > 0:
			return locateZero(v1, v2, precision)
		else:
			return None
	
	# Copy the matrix without the first entry to shift it over
	shifted_mat = delete(x_mat,0,0)

	mat = zip(x_mat, shifted_mat) # create a list of tuples
	mat = map(locate, mat)

	return array([zero for zero in mat if zero is not None])


# interpolate a zero to a tolerance 
# x_mat1 - Should be a vector with a negative y 
# x_mat2 - should be a vector with a positive y
# prec - the zero will be located to within this precision
def locateZero(x_mat1, x_mat2, precision):

	# if x_mat1 and 2 are in the in tolerance range, return the interpolation between them
	if( abs(x_mat1[1])+abs(x_mat2[1]) <= precision):
		return (x_mat1+x_mat2)/2.0

	# Starting and ending time
	t = x_mat1[4]
	t2 = x_mat2[4]


	# Calculate the intermediate grid, cut up into 128 pieces
	x_mat = scp.odeint(f, x_mat1, linspace(t,t2,128), atol = tol*100, rtol = tol*1000)

	# get the y values
	x = x_mat.T[1]

	# Error message if somehow there is no zero
	if not (x[0] <= 0 and x[-1] > 0):
		print("Zero Not found Error")
		return None


	# This part quickly locates a single zero in the matrix
	w = len(x_mat)
	i = 0
	while w != 1:
	 	w = int(w/2.0)
	 	if (x[i+w]<0):
	 		i += w

	return locateZero(x_mat[i], x_mat[i+w], precision)


# Estimates the time of an orbit 
# x - initial vector
# time - time to run test for
# Returns time per orbit
def orbitTimer(x,  time = 100):
	h=0.2
	x_mat=ode(x,h,time)

	orbit_mat = locateCrossing(x_mat)

	return time/len(orbit_mat) 


def trace(x, time=1000,name=None, xd=0):
	result = ode( x, 0.01, time)
	plt.plot(result.T[0],result.T[1],color=(0,0,0))
	plt.xlabel("x")
	plt.ylabel("y")
	plt.legend()

	if name is not None:
		plt.savefig("figures/"+name,dpi=300)
	plt.show()
	plt.clf()

# simple method to add a column to a matrix
def col_append(x,a):
	tmp = append(x[0],a)

	for row in x:
		row = append(row,a)
		tmp = vstack([tmp,row])

	tmp = delete(tmp,0,0)
	return tmp

# simple method to find distance between two vectors
def dist(a1,b1):
	a = array(a1)
	b = array(b1)
	c = linalg.norm(a-b)
	return c


if __name__=='__main__' :
	main()
