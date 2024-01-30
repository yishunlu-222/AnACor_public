import matplotlib.pyplot as plt
import numpy as np
import pdb
# from sympy import symbols, exp, sin, tan, csc, cot,cos,sec

# def new_left ( mu_value , theta_value , L_value , H_value ) :
#     # pdb.set_trace()
#     H, mu, theta, L = symbols('H mu theta L')
#     expression = exp(-2 * H * mu * csc(theta)) * (-1 + exp(H * mu * csc(theta)) * sin(theta)) - \
#              exp(-mu * (L + 2 * H * csc(theta))) * (-1 + exp(H * mu * (cot(theta) + csc(theta)))) * \
#              tan(theta / 2)
#     evaluated_expression = expression.subs({H: H_value, mu: mu_value, theta: theta_value, L: L_value})
#     return evaluated_expression.evalf()

# def new_right ( mu_value , theta_value , L_value , H_value ) :
#     H, mu, theta, L = symbols('H mu theta L')

#     # Redefine the expression to include the symbol L
#     expression = exp(-L * mu) * (H * mu * cos(theta) * tan(theta/2) - (-1 + exp(H * mu)) * cot(theta/2) * sec(theta)) / (mu**2 * (-1 + cos(theta)))

#     evaluated_expression = expression.subs({H: H_value, mu: mu_value, theta: theta_value, L: L_value})
#     return evaluated_expression.evalf()

# def new_analytical ( mu_value , theta_value , L_value , H_value ) :
#     left = new_left(mu_value, theta_value, L_value, H_value)
#     right = new_right(mu_value, theta_value, L_value, H_value)
#     return (left + right)/(L_value * H_value)

def standard_trans ( mu , radian , width , height ) :
    T = (np.sin( radian ) / (mu ** 2)) * (1 - np.exp( -mu * height )) * (1 - np.exp( -mu * width / np.sin( radian ) ))
    # t1 = (1-np.exp(-mu*width)) #c
    # t2=np.sin(radian) #c
    # t3=(1-np.exp(-mu*height/np.sin(radian))) #c
    # t4=t1*t2*t3/(mu**2)
    return T


def ana_f_l_2 ( mu , radian , width , height ) :
    assert radian > np.arctan( height / width )
    a = height
    b = width
    c = mu
    d = np.cos( radian )
    f = 1 / np.tan( radian )
    k = b - a / np.tan( radian )
    H = c ** 2 * (d - 1) ** 2 * f  # ok
    J = d * np.exp( - (c * (a * d * f + b + (d - 1) * k)) / d )  # ok
    part1 = np.exp( a * c * f ) - np.exp( a * c * f / d )  # ok
    part2 = a * np.exp( -b * c ) / (c - c * d)  # ok
    T2 = d * (J * part1 / H + part2)

    return T2


# def ana_f_l_1(mu,radian,width,height):
#     assert radian > np.arctan(height/width)
#
#     K = width-height/np.tan(radian)
#     J=1/(mu**2)
#     C = -( np.sin(radian)*np.tan(radian) * np.exp(-mu*K) ) / (np.sin(radian)-np.tan(radian))
#     part1 = C * ( np.exp(-height*mu/np.sin(radian))  -  np.exp(-height*mu/np.tan(radian)) )
#     part2 = np.sin(radian) * ( 1 - np.exp(-height*mu/np.sin(radian)) )
#     T1 = J*(part1+part2)
#     return T1
def ana_f_l_1 ( mu , radian , width , height ) :
    assert radian > np.arctan( height / width )
    a = height
    b = width
    c = mu
    d = np.sin( radian )
    f = 1 / np.tan( radian )
    k = b - a / np.tan( radian )

    J = 1 / (c ** 2)
    H = d * np.exp( -c * (a / d + a * f + k) ) / (d * f - 1)
    part1 = H * (np.exp( a * c / d ) - np.exp( a * c * f ))
    part2 = d * (1 - np.exp( -a * c / d ))
    T1 = J * (part1 + part2)
    return T1

def ana_f_longer_height_area ( mu , radian , width , height ) :
    T_s = ana_f_s_1( mu , radian , width , height )
    T_s_2 = ana_f_s_2( mu , radian , width , height )
    return (T_s + T_s_2) / (height * width)

def ana_180 ( mu ,  width , height ) :
    T_2=np.exp(-mu*width) *width*height/(width*height)
    return T_2
def ana_0(mu,width,height):
    T_1 =- ((-1 + np.exp(-2*mu*width))*height/ (2*mu) )/ (width * height)
    return T_1

def ana_f_s_1 ( mu , radian , width , height ) :
    assert radian < np.arctan( height / width )
    a = height
    b = width
    c = mu
    d = np.sin( radian )
    f = np.tan( radian )
    k = a - b * np.tan( radian )

    J = d / (c ** 2)
    H = d / (d - f)
    part1 = np.exp( c * (-a - b * d + b * f + k) / d )
    part2 = np.exp( c * (k - a) / d )

    T1 = J * (H * (part1 - part2) - np.exp( -b * c ) + 1)

    return T1


def ana_f_s_2 ( mu , radian , width , height ) :
    assert radian < np.arctan( height / width )
    a = height
    b = width
    c = mu
    d = np.cos( radian )
    f = np.tan( radian )
    k = a - b * np.tan( radian )

    J = d * f * np.exp( -b * c * (d + 1) / d ) / (c ** 2 * (d - 1) ** 2)
    part1 = d * np.exp( b * c ) - (b * c * (d - 1) + d) * np.exp( b * c / d )
    H = d * k / (c - c * d)
    part2 = np.exp( -b * c ) - np.exp( -b * c / d )
    T2 = J * part1 + H * part2
    return T2

def ana_f_exit_sides ( mu , radian , width , height ) :
    T_l = ana_f_l_1( mu , radian , width , height )
    T_l_2 = ana_f_l_2( mu , radian , width , height )
    return (T_l + T_l_2) / (height * width)

def ana_f_exit_top ( mu , radian , width , height ) :
    # if np.abs(radian-np.pi/2) < 1e-3:
    #     pdb.set_trace()
    T_s = ana_f_s_1( mu , radian , width , height )
    T_s_2 = ana_f_s_2( mu , radian , width , height )
    
    return (T_s + T_s_2) / (height * width)

if __name__ == '__main__':

    mu = 1
    width = 8  # x-axis
    height = 0.01  # y-axis
    t_theta = 2 * 15
    radian = np.pi / 180 * t_theta  # convert to radian as python math.sine adopt radian
    # transmission factor for
    T = standard_trans( mu , radian , width , height )
    T_l = ana_f_l_1( mu , radian , width , height )
    T_l_2 = ana_f_l_2( mu , radian , width , height )

    print( T / (height * width) )
    print( T_l / (height * width) )
    print( T_l_2 / (height * width) )
    print( (T_l + T_l_2) / (height * width) )

    print( 'below is the under degree' )
    mu = 1
    width = 0.5  # x-axis
    height = 1  # y-axis
    t_theta = 2 * 15
    radian = np.pi / 180 * t_theta  # convert to radian as python math.sine adopt radian
    # transmission factor for
    T = standard_trans( mu , radian , width , height )
    T_s = ana_f_s_1( mu , radian , width , height )
    T_s_2 = ana_f_s_2( mu , radian , width , height )
    print( T / (height * width) )
    print( T_s / (height * width) )

    print( T_s_2 / (height * width) )
    print( (T_s + T_s_2) / (height * width) )
