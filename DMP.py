import math
import numpy as np

# Parameters
ALPHA = -1*math.log(0.01) # Alpha that leads to 99% convergence
T = 6
K = 36
D = 2*math.sqrt(K)

# Returns DMP parameters
# trajectory --> n rows (num points) and m columns (dimension of data)
# K and D --> parameters
def DMPLearning(trajectory, K, D):
    # Error Checking
    if(len(trajectory) == 0):
        return None

    # Initialize our retunr value
    dimensions = len(trajectory[0]['coord'])
    ans = [{
        "num_demos": 1,
        "f": None
    }]*dimensions

    # Need to make one differential equation per dimension
    for dim in range(dimensions):
        x = [ i['coord'][dim] for i in trajectory ]
        v = [0]*len(x)
        v_dot = [0]*len(x)
        for i in range(1,len(x)):
            v[i] = T*((x[i] - x[i-1])/float(trajectory[i]['t'] - trajectory[i-1]['t']))
            v_dot[i-1] = (v[i] - v[i-1])/float(trajectory[i]['t'] - trajectory[i-1]['t'])

        # print(v)
        # print(v_dot)
        # Solution to canonical system is s(t) = math.exp(-t*alpha/T)
        f_target = []
        g = x[len(x)-1]
        x_0 = x[0]
        for i in range(len(x)):
            t = trajectory[i]['t']
            entry = {}
            entry["s"] = math.exp(-t*ALPHA/T)
            entry["value"] = (T*v_dot[i] + D*v[i])/float(K) - (g - x[i]) + (g - x_0)*entry["s"]
            f_target.append(entry)
        ans[dim]["f"] = f_target
    return ans

def DMPPlanning(parameters, startPos, startVel, endPos, T_new, dt):
    plan = []
    velocities = []
    plan.append({'coord':[float(i) for i in startPos], 't': 0})
    velocities.append([float(i) for i in startVel])

    dimensions = len(startPos)
    for i in range(1, math.ceil(T_new/float(dt))+1):
        S = math.exp(-dt*(i-1)*ALPHA/T_new)
        point = [0]*dimensions
        velocity = [0]*dimensions
        #print("S(" + str(S) + ") ==> " + str(interpolate_function(S, parameters)))
        for dim in range(dimensions):
            v_dot = (K*(endPos[dim] - plan[i-1]['coord'][dim]) - D*velocities[i-1][dim] - K*(endPos[dim] - plan[0]['coord'][dim])*S + K*interpolate_function(S, parameters))/float(T_new)
            velocity[dim] = velocities[i-1][dim] + dt*v_dot
            # print(v_dot)
            x_dot = velocity[dim]/float(T_new)
            point[dim] = plan[i-1]['coord'][dim] + dt*x_dot
        plan.append({'coord':point, 't': dt*i})
        velocities.append(velocity)

    return plan

def interpolate_function(S, parameters):
    f = parameters[0]["f"]
    i = 0
    while(i < len(f) and f[i]["s"] > S):
        i = i + 1
    if(i == len(f)):
        return f[i-1]["value"]
    elif(i == 0):
        return f[i]["value"]
    else:
        return f[i-1]["value"] + (S - f[i-1]["s"])*(f[i]["value"]-f[i-1]["value"])/(f[i]["s"]-f[i-1]["s"])


# Helper functions to turn any function into points
##############################################################
# Function you are trying to approximate
def f(x):
    return x

def getCoordinates(func, startx, endx, numPoints):
    dx = (endx - startx)/float(numPoints-1)
    coords = [0]*numPoints
    points = 0
    x = startx
    while points < numPoints:
        coord = {"coord" : [func(x)] ,"t": x}
        coords[points] = coord
        x = x + dx
        points = points + 1
    return coords

###############################################################

traj = getCoordinates(f, 0, 6, 13)
model = DMPLearning(traj, K, D)
#for i in model[0]["f"]:
#    print(i)
plan = DMPPlanning(model, [0], [0], [10], 10, 0.5)
for i in traj:
    print(i)
print("=" * 10)
print('{:>5} | {}'.format('t','coord'))
print("-"*20)
for i in plan:
    print('{:>5} | {:.2f}'.format(i["t"],i["coord"][0]))
