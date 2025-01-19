#python nbody.py config_1.txt
import numpy as np
import matplotlib.pyplot as plt
import sys

#given input file
input_fname = sys.argv[1]

class OrbitState:
    """
     state of a body orbit have 
     name tag of body
     mass of the body
     x position 
     y position 
     x component of velocity
     y component of velocity
    """
    def _init__(self, state):
        self.name = state[0]
        self.mass = state[1]
        self.x = state[2]
        self.y = state[3]
        self.vx = state[4]
        self.vy = state[5]


class Solver:
    """
    Given all bodies states in a vector form
    stepsize of integration
    and final time of the simulation
    get state of body from start till the final time
    """
    def __(self, Yvec, dt, finalT):
        self.Yvec = Yvec
        self.dt = dt
        self.finalT = finalT

    def RK4_solver(self, Yvec, dt, TotalTime):
        # Yvec is an array of arrays and each array has 4 components of planets position/ velocities
        # Mvals = np.ones(len(Yvec)) # keep M same for each planet
        tarray = []
        Orbit_state = []

        # initialize time
        t = 0

        # store the initial conditions
        tarray.append(t)
        Orbit_state.append(Yvec)

        # main timestep loop
        while t < TotalTime:
            old_orbit_state = Orbit_state[-1]
            # need to fix this
            temp_orbit_state = old_orbit_state.copy()
            new_orbit_state = old_orbit_state.copy()
            # make sure that the last step does not take us past T
            if t + dt > TotalTime:
                dt = TotalTime - t

            # step1 of rk4
            # k1 = self.One_Planet(old_orbit_state) #dtYvec
            # k1 = self.Two_Body(old_orbit_state, r12) #dtYvec
            k1 = self.N_Body(old_orbit_state)  # dtYvec

            # we want to change just pos and velocities
            temp_orbit_state[:, 2:6] = old_orbit_state[:, 2:6] + 0.5 * dt * k1[:, 2:6]
            # k2 = self.One_Planet(temp_orbit_state) #dtYvec
            # k2 = self.Two_Body(temp_orbit_state, r12) #dtYvec
            k2 = self.N_Body(temp_orbit_state)

            temp_orbit_state[:, 2:6] = old_orbit_state[:, 2:6] + 0.5 * dt * k2[:, 2:6]
            # k3 = self.One_Planet(temp_orbit_state) #dtYvec
            # k3 = self.Two_Body(temp_orbit_state, r12) #dtYvec
            k3 = self.N_Body(temp_orbit_state)  # dtYvec

            temp_orbit_state[:, 2:6] = old_orbit_state[:, 2:6] + dt * k3[:, 2:6]

            # k4 = self.One_Planet(temp_orbit_state) #dtYvec
            # k4 = self.Two_Body(temp_orbit_state, r12) #dtYvec
            k4 = self.N_Body(temp_orbit_state)  # dtYvec

            # do the final update
            new_orbit_state[:, 2:6] = old_orbit_state[:, 2:6] + dt / 6.0 * (
                k1[:, 2:6] + 2 * k2[:, 2:6] + 2 * k3[:, 2:6] + k4[:, 2:6]
            )

            t += dt

            # store the state
            tarray.append(t)
            Orbit_state.append(new_orbit_state)

            # print(Orbit_state[-1])
        return tarray, Orbit_state

  

class CelestialRhapsody:
    """
    Given Planets initial positions
    and initial velocities get their
    motions under tha action of gravity
    of other planets around them

    """

    def __init__(self, x, y, vx, vy, mass_ratio=1.0, Mtot=1.0, G=1.0):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

    def N_Body(self, Yvec, G=1.0):
        """
        given Yvec = [B1, B2, B3 ,... Bn]
        where B1 = ["planetname", mass, x,y,vx,vy]
        return dYvec/dt based on Newtonian Gravity
        for each body we compute dB/dt anf oncatenate it
        """
        dtYvec = np.empty_like(Yvec)
        for i in range(len(Yvec)):
            # print("Planet info = ", Yvec[i])
            dtYvec[i][:2] = Yvec[i][:2]  # no change in planet name and mass

            Planet_i = Yvec[i]
            M_i = Planet_i[1]  # mass
            x_i, y_i = Planet_i[2:4]  # posx, posy of planet i
            # print("p1, x, y = ", Planet_i[0] , x_i, y_i)
            dotx, doty = Planet_i[4:6]  # vx, vy
            dotvx = 0
            dotvy = 0
            for k in range(len(Yvec)):
                if k != i:
                    Planet_k = Yvec[k]
                    M_k = Planet_k[1]
                    x_k, y_k = Planet_k[2:4]
                    x_ik = x_i - x_k
                    y_ik = y_i - y_k
                    r_ik = np.sqrt((x_i - x_k) ** 2 + (y_i - y_k) ** 2)
                    # print("xi, y_i, x_k, y_k", x_i, y_i, x_k, y_k)
                    # Maybe need for collision
                    # if r_ik == 0.0:
                    #    r_ik == np.sqrt(0.01)
                    r3_k = r_ik**3
                    dotvx -= G * M_k * x_ik / r3_k
                    dotvy -= G * M_k * y_ik / r3_k

            dtYvec[i][2:6] = dotx, doty, dotvx, dotvy
        return dtYvec

    def One_Planet(self, Yvec, Mtot=1.0, G=1.0):
        """
        for Fixed Sun and orbiting Planet problem
        we will have Yvec which have x, y, Vx and Vy vals
        as position and velocity components of a body
        """

        # Yvec = np.array([self.x, self.y, self.vx, self.vy])
        xdot = Yvec[2]
        ydot = Yvec[3]

        r = np.sqrt(Yvec[0] ** 2 + Yvec[1] ** 2)
        r3 = r**3

        vxdot = -Yvec[0] * G * Mtot / r3
        vydot = -Yvec[1] * G * Mtot / r3

        dtYvec = np.array([xdot, ydot, vxdot, vydot])
        return dtYvec

    def Two_Body(self, Yvec, r12, M1=0.5, M2=0.5, G=1.0):  # p1vec, p2vec):
        """
        Now we have two fixed bodies with positions
        p1vec, p2vec let M1= M2 = 0.5 and G=1
        Now we see how the motion of planet get effeted
        Now Yvec contained r12x, r12 y info about middle two bodies

        r12 = x1-x2, y1-y2
        """
        x, y = Yvec[0], Yvec[1]

        xdot = Yvec[2]
        ydot = Yvec[3]

        # discuss with David
        # rx12 = Yvec[4]
        # ry12 = Yvec[5]

        x1 = r12[0] / 2.0
        x2 = -x1
        # 1: Using midpoint (x1+x2)/2.0 = 0
        # x1 = -x2
        # and Using r12 = x1 - x2
        # we solve for  x1 = 1/2 r12 and x2 = -1/2r12
        y1, y2 = 0.0, 0.0
        r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2)
        r1cube = r1**3
        r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2)
        r2cube = r2**3

        vxdot = -G * ((x - x1) * M1 / r1cube + (x - x2) * M2 / r2cube)
        vydot = -G * ((y - y1) * M1 / r1cube + (y - y2) * M2 / r2cube)

        dtYvec = np.array([xdot, ydot, vxdot, vydot])
        return dtYvec

    def Euler_solver(self, Yvec, dt, TotalTime):
        tarray = []
        Orbit_state = []
        t = 0  # initial time
        tarray.append(t)
        Orbit_state.append(Yvec)

        # Euler integral
        while t < TotalTime:
            old_orbit_state = Orbit_state[-1]
            # make sure final time is not T+dt
            if t + dt > TotalTime:
                dt = TotalTime - t
            dtYvec = self.One_Planet(old_orbit_state)
            # Euler step
            new_orbit_state = old_orbit_state + dt * dtYvec
            t += dt

            tarray.append(t)
            Orbit_state.append(new_orbit_state)

        return tarray, Orbit_state

    def RK2_solver(self, Yvec, dt, TotalTime):
        tarray = []
        Orbit_state = []

        # initialize time
        t = 0

        # store the initial conditions
        tarray.append(t)
        Orbit_state.append(Yvec)

        # main timestep loop
        while t < TotalTime:
            old_orbit_state = Orbit_state[-1]

            # make sure that the last step does not take us past T
            if t + dt > TotalTime:
                dt = TotalTime - t

            # get the RHS
            dtYvec = self.One_Planet(old_orbit_state)

            # RK2step:  state at the midpoint
            temp_orbit_state = old_orbit_state + 0.5 * dt * dtYvec

            # evaluate the RHS at the midpoint
            dtYvec = self.One_Planet(temp_orbit_state)

            # RK2step:final update
            new_orbit_state = old_orbit_state + dt * dtYvec
            t += dt

            # store the state
            tarray.append(t)
            Orbit_state.append(new_orbit_state)

        return tarray, Orbit_state

    def RK4_solver(self, Yvec, dt, TotalTime):
        # Yvec is an array of arrays and each array has 4 components of planets position/ velocities
        # Mvals = np.ones(len(Yvec)) # keep M same for each planet
        tarray = []
        Orbit_state = []

        # initialize time
        t = 0

        # store the initial conditions
        tarray.append(t)
        Orbit_state.append(Yvec)

        # main timestep loop
        while t < TotalTime:
            old_orbit_state = Orbit_state[-1]
            # need to fix this
            temp_orbit_state = old_orbit_state.copy()
            new_orbit_state = old_orbit_state.copy()
            # make sure that the last step does not take us past T
            if t + dt > TotalTime:
                dt = TotalTime - t

            # step1 of rk4
            # k1 = self.One_Planet(old_orbit_state) #dtYvec
            # k1 = self.Two_Body(old_orbit_state, r12) #dtYvec
            k1 = self.N_Body(old_orbit_state)  # dtYvec

            # we want to change just pos and velocities
            temp_orbit_state[:, 2:6] = old_orbit_state[:, 2:6] + 0.5 * dt * k1[:, 2:6]
            # k2 = self.One_Planet(temp_orbit_state) #dtYvec
            # k2 = self.Two_Body(temp_orbit_state, r12) #dtYvec
            k2 = self.N_Body(temp_orbit_state)

            temp_orbit_state[:, 2:6] = old_orbit_state[:, 2:6] + 0.5 * dt * k2[:, 2:6]
            # k3 = self.One_Planet(temp_orbit_state) #dtYvec
            # k3 = self.Two_Body(temp_orbit_state, r12) #dtYvec
            k3 = self.N_Body(temp_orbit_state)  # dtYvec

            temp_orbit_state[:, 2:6] = old_orbit_state[:, 2:6] + dt * k3[:, 2:6]

            # k4 = self.One_Planet(temp_orbit_state) #dtYvec
            # k4 = self.Two_Body(temp_orbit_state, r12) #dtYvec
            k4 = self.N_Body(temp_orbit_state)  # dtYvec

            # do the final update
            new_orbit_state[:, 2:6] = old_orbit_state[:, 2:6] + dt / 6.0 * (
                k1[:, 2:6] + 2 * k2[:, 2:6] + 2 * k3[:, 2:6] + k4[:, 2:6]
            )

            t += dt

            # store the state
            tarray.append(t)
            Orbit_state.append(new_orbit_state)

            # print(Orbit_state[-1])
        return tarray, Orbit_state


#fix this
x0, y0, vx0, vy0 = 0, 0, 0,0
# call class and provide dt and final time
rap = CelestialRhapsody(x0, y0, vx0, vy0)
tend = 5   
dt = 0.01  

from util_input import read_config
g, f = read_config(input_fname)
#G = g[0]
#dt = g[1]
#tend = int(g[2])

yvec = np.array(f, dtype=object)
tarr, newYvecRK4 = rap.RK4_solver(yvec, dt, tend)
# convert appended list of lists to an np.array
newYvecRK4 = np.asarray(newYvecRK4)

#clean it 
minX = []
maxX = []
minY = []
maxY = []

Xt = []
Yt = []
lineplotlist = []
pointplotlist = []

for i in range(newYvecRK4.shape[1]):
    print(i)
    p_orbit = newYvecRK4[:, i]
    xt, yt = p_orbit[:, 2], p_orbit[:, 3]
    Xt.append(xt)
    Yt.append(yt)
    minx, maxx = np.min(xt), np.max(xt)
    miny, maxy = np.min(yt), np.max(yt)
    minX.append(minx)
    minY.append(miny)
    maxX.append(maxx)
    maxY.append(maxy)
   

xmin = np.min(np.array(minX))
ymin = np.min(np.array(minY))
xmax = np.max(np.array(maxX))
ymax = np.max(np.array(maxY))


plt.close("all")
fig = plt.figure()
ax = plt.axes(xlim=(xmin - 0.1, xmax + 0.1), ylim=(ymin - 0.1, ymax + 0.1))
for i in range(newYvecRK4.shape[0]):
    (my_line,) = ax.plot([], [], c='k', lw=2)
    (my_point,) = ax.plot([], [], marker="o", ms=9)
    lineplotlist.append(my_line)
    pointplotlist.append(my_point)

Xt = np.array(Xt)
Yt = np.array(Yt)

from matplotlib import animation

def get_step(n,X0, Y0, this_line, this_point,):
    for i in range(len(X0)):
        x, y = X0[i,:], Y0[i, :]
        lineplotlist[i].set_data(x[: n + 1], y[: n + 1])
        pointplotlist[i].set_data(x[n], y[n])


mymovie = animation.FuncAnimation(fig, get_step, frames=np.arange(1, len(tarr)),
    fargs=(Xt,Yt, lineplotlist,pointplotlist), )
mymovie.save("movie.mp4", fps=30)
