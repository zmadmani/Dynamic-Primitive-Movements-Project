import math
import numpy as np

# Parameters
ALPHA = -1*math.log(0.01) # Alpha that leads to 99% convergence
T = 5.0

# Returns DMP parameters
# trajectory --> n rows (num points) and m columns (dimension of data)
# K and D --> parameters
def DMPLearning(trajectory, K, D):
    # Error Checking
    if(len(trajectory) == 0):
        return None

    # Initialize our retunr value
    dimensions = len(trajectory[0])
    ans = [{
        "num_demos": 1,
        "f": None
    }]*dimensions

    # Need to make one differential equation per dimension
    for dim in range(dimensions):
        x = trajectory[:,dim]
        v = [0]*len(x)
        v_dot = [0]*len(x)
        for i in range(len(x)-1):
            v[i] =T*(x[i+1] - x[i])
            if(i > 0):
                v_dot[i-1] = v[i] - v[i-1]
                if(i == len(x)-2):
                    v_dot[i] = v[i+1] - v[i]

        # Solution to canonical system is s(t) = math.exp(-t*alpha/T)
        f_target = []
        g = x[len(x)-1]
        x_0 = x[0]
        for i in range(len(x)):
            entry = {}
            entry["s"] = math.exp(-i*ALPHA/T)
            entry["value"] = (T*v_dot[i] + D*v[i])/K - (g - x[i]) + (g - x_0)*entry["s"]
            f_target.append(entry)
        ans[dim]["f"] = f_target
    return ans

def DMPPlanning(parameters, startPos, startVel, endPos, t, dt):
    pass

def approximate_function():
    pass


traj = np.array([[0],[3],[2]])
print(DMPLearning(traj, 1, 2))
