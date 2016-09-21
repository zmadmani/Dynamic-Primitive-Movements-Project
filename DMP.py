import math
import numpy as np
from functionApproximators import interpolate_function
from functionApproximators import GBF
# Parameters
ALPHA = -1*math.log(0.01) # Alpha that leads to 99% convergence
T = 4
K = 70
D = 2*math.sqrt(K)
numPoints = 40
numFuncs = 40*2+2
centers = []
widths = []
curr = 1
width = 1.0
for i in range(numFuncs):
    centers.append(curr)
    curr = curr/float(1.258925412)
    widths.append(width)
    width = width/float(1.06)

# obstacle
OBSTACLE = [3,0]

# Returns DMP parameters
# trajectory --> n rows (num points) and m columns (dimension of data)
# K and D --> parameters
def DMPLearning(trajectory, K, D):
    # Error Checking
    num_demos = len(trajectory)
    if(num_demos == 0):
        return None

    # Initialize our retunr value
    dimensions = len(trajectory[0][0]['coord'])
    ans = []

    # Need to make one differential equation per dimension
    for dim in range(dimensions):
        elem = {
            'num_demos': num_demos,
            'f': None
        }
        f_target = []
        for traj in trajectory:
            x = [ i['coord'][dim] for i in traj ]
            v = [0]*len(x)
            v_dot = [0]*len(x)
            for i in range(1,len(x)):
                v[i] = T*((x[i] - x[i-1])/float(traj[i]['t'] - traj[i-1]['t']))
                v_dot[i-1] = (v[i] - v[i-1])/float(traj[i]['t'] - traj[i-1]['t'])

            v_dot[len(x)-1] = 0
            v[len(x)-1] = 0

            #print(v)
            # print(v_dot)
            # Solution to canonical system is s(t) = math.exp(-t*alpha/T)
            g = x[len(x)-1]
            x_0 = x[0]
            for i in range(len(x)):
                t = traj[i]['t']
                entry = {}
                entry["s"] = math.exp(-t*ALPHA/T)
                entry["value"] = (T*v_dot[i] + D*v[i])/float(K) - (g - x[i]) + (g - x_0)*entry["s"]
                f_target.append(entry)
        if(num_demos > 1):
            inputs = [entry["s"] for entry in f_target]
            for i in range(len(inputs)):
                centers[i] = inputs[i]
            outputs = [entry["value"] for entry in f_target]
            # if(dim == 0):
            #     print("dim " + str(dim))
            #     for i in range(len(inputs)):
            #         print(str(inputs[i]) + " " + str(outputs[i]))
            gbf = GBF(numFuncs, centers, widths)
            gbf.train(inputs, outputs)
            # if(dim == 0):
            #     print("====")
            #     for i in range(len(inputs)):
            #         print(str(inputs[i]) + " " + str(gbf.predict(inputs[i])))
            elem["f"] = gbf
        else:
            elem["f"] = f_target
        ans.append(elem)
    return ans

def DMPPlanning(parameters, startPos, startVel, endPos, T_new, dt):
    num_demos = parameters[0]['num_demos']
    plan = []
    velocities = []
    plan.append({'coord':[float(i) for i in startPos], 't': 0})
    velocities.append([float(i) for i in startVel])

    dimensions = len(startPos)
    for i in range(1, math.ceil(T_new/float(dt)+1)):
        S = math.exp(-dt*(i-1)*ALPHA/T_new)
        point = [0]*dimensions
        velocity = [0]*dimensions
        couplingTerm = getAccelerationVector(getDist(plan[i-1]['coord'],OBSTACLE),plan[i-1]['coord'],OBSTACLE)
        # print(couplingTerm)
        for dim in range(dimensions):
            v_dot = 0
            if(num_demos == 1):
                v_dot = (K*(endPos[dim] - plan[i-1]['coord'][dim]) - D*velocities[i-1][dim] - K*(endPos[dim] - plan[0]['coord'][dim])*S + K*interpolate_function(S, parameters[dim]) + couplingTerm[dim])/float(T_new)
            else:
                v_dot = (K*(endPos[dim] - plan[i-1]['coord'][dim]) - D*velocities[i-1][dim] - K*(endPos[dim] - plan[0]['coord'][dim])*S + K*parameters[dim]['f'].predict(S))/float(T_new)
            velocity[dim] = velocities[i-1][dim] + dt*v_dot
            # print(v_dot)
            x_dot = velocity[dim]/float(T_new)
            point[dim] = plan[i-1]['coord'][dim] + dt*x_dot
        plan.append({'coord':point, 't': dt*i})
        velocities.append(velocity)

    return plan


##############################################################
# Helper functions to turn any function into points
##############################################################
# Get Euclidean distance
def getDist(x,y):
    return math.sqrt(math.pow(x[0]-y[0],2) + math.pow(x[1]-y[1],2))

def getAccelerationVector(dist,point,obstacle):
    forceMultiplier = 500
    stdv = 1
    acceleration = forceMultiplier*math.exp(-math.pow(dist,2)/(2*math.pow(stdv,2)))
    displacement = [(point[0] - obstacle[0])*acceleration/dist, (point[1] - obstacle[1])*acceleration/dist]
    return displacement

# Function you are trying to approximate
def f(x):
    return math.sin(x)

def addNoise(x):
    return np.random.normal(scale=0.1) + x

def getCoordinates(func, startx, endx, numPoints, noise):
    dx = (endx - startx)/float(numPoints)
    coords = [0]*(numPoints+1)
    points = 0
    x = startx
    while points <= numPoints:
        coord = {}
        if(noise):
            coord = {"coord" : [addNoise(x),addNoise(func(x))] ,"t": T*points/float(numPoints)}
        else:
            coord = {"coord" : [x,func(x)] ,"t": T*points/float(numPoints)}
        coords[points] = coord
        x = x + dx
        points = points + 1
    return coords

def combineTrajectories(trajectories):
    finalTrajectory = []
    for trajectory in trajectories:
        for item in trajectory:
            finalTrajectory.append(item)
    return finalTrajectory

###############################################################

# gbf = GBF(3, [1,.5,0],[.5,.5,.5])
# input = [1, .5, 0.1]
# output = [1, 1, 1]
# gbf.train(input,output)
# exit()

traj1 = getCoordinates(f, 0, 4*math.pi, numPoints, False)
#traj2 = getCoordinates(f, 0, 4*math.pi, numPoints, True)
#traj = [traj1, traj2]
traj = [traj1]
model = DMPLearning(traj, K, D)
# for item in model:
#     for elem in item['f']:
#         print(elem)
plan = DMPPlanning(model, [0,0], [0,0], [4*math.pi,0], T, 0.1)
print("==========GIVEN TRAJECTORY==========")
print('{:>5}\t{}'.format('t','coord'))
print("-"*20)
for i in traj[0]:
    print('{:>5.2f}\t{}'.format(i["t"],'   '.join(['{:3.3f}'.format(elem) for elem in i["coord"]])))

# for i in traj[1]:
#     print('{:>5.2f}\t{}'.format(i["t"],'   '.join(['{:3.3f}'.format(elem) for elem in i["coord"]])))

print("\n")
print("=" * 10)
print("\n")
print("==========PLANNED TRAJECTORY==========")
print('{:>5}\t{}'.format('t','coord'))
print("-"*20)
for i in plan:
    print('{:>5.2f}\t{}'.format(i["t"],'   '.join(['{:3.3f}'.format(elem) for elem in i["coord"]])))
