import numpy as np
import math

#quaternion math for correcting jig inaccuracies in robot following
#taken from https://personal.utdallas.edu/~sxb027100/dock/quaternion.html#:~:text=order%20of%20multiplication.-,The%20inverse%20of%20a%20quaternion%20refers%20to%20the%20multiplicative%20inverse,for%20any%20non%2Dzero%20quaternion.

def q_multiply(q1, q2): #quaternion multiplication

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    q1q2 = np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])

    return q1q2

def q_inv(q): #inverse of a quaternion

    w, x, y, z = q
    q_norm = w**2 + x**2 + y**2 + z**2
    q_i = np.array([w/q_norm, -x/q_norm, -y/q_norm, -z/q_norm])

    return q_i

def q2E(q): #quaternion to euler angles

    w, x, y, z = q

    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

#define reference data and theoretical work object data

# q_ref = np.array([0.674868797, 0, 0, -0.737937739]) #reference quaternion taken from location plate (in CAD)

q_ref = np.array([0.73793169,0,0,0.674875412])
q_1 = np.array([0.898811099,-0.052880349,-0.113021082,-0.420200561])
q_2 = np.array([0,0.905893224,-0.423506159,0])
q_3 = np.array([0.199807967,-0.798680576,0.373374019,0.427525386])

p_ref = np.array([101.598088732,721.892992064,724.02]) #reference location taken from corner of location plate (in CAD)
p_1 = np.array([131.126112031,686.573440169,727.904460545])
p_2 = np.array([167.842599783,642.642744774,740.223422667])
p_3 = np.array([168.332010692,642.045987203,726.158955828])

#define translations from reference point to work objects

dp1 = p_1 - p_ref
dp2 = p_2 - p_ref
dp3 = p_3 - p_ref

#define rotations from reference point to work objects

dq1 = q_multiply(q_inv(q_ref), q_1)
dq2 = q_multiply(q_inv(q_ref), q_2)
dq3 = q_multiply(q_inv(q_ref), q_3)

#define actual reference work object (measured by Robot)

p_act = np.array([103.034,720.388,724.107])
q_act = np.array([0.736937,0.000842703,-0.000701218,0.67596])

#define new work object position data

p_1a = p_act + dp1
p_2a = p_act + dp2
p_3a = p_act + dp3

#define new work object quaternion data

q_1a = q_multiply(q_act, dq1)
q_2a = q_multiply(q_act, dq2)
q_3a = q_multiply(q_act, dq3)

print("Actual Workobject 1 = [[" +str(p_1a[0])+ ", " +str(p_1a[1])+ ", " +str(p_1a[2])+ "],[ " +str(q_1a[0])+ ", "  +str(q_1a[1])+ ", "  +str(q_1a[2])+ ", "  +str(q_1a[3])+ "]] " )
print("Actual Workobject 2 = [[" +str(p_2a[0])+ ", " +str(p_2a[1])+ ", " +str(p_2a[2])+ "],[ " +str(q_2a[0])+ ", "  +str(q_2a[1])+ ", "  +str(q_2a[2])+ ", "  +str(q_2a[3])+ "]] " )
print("Actual Workobject 3 = [[" +str(p_3a[0])+ ", " +str(p_3a[1])+ ", " +str(p_3a[2])+ "],[ " +str(q_3a[0])+ ", "  +str(q_3a[1])+ ", "  +str(q_3a[2])+ ", "  +str(q_3a[3])+ "]] " )
