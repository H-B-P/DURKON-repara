import pandas as pd
import numpy as np
import math
import scipy
from scipy.special import erf
#Easy objective functions

def Gauss_grad(pred,act):
 return 2*(pred-act)

def Poisson_grad(pred,act):
 return (pred-act)/pred

def Gamma_grad(pred,act):
 return (pred-act)/(pred*pred)

def Logistic_grad(pred,act):
 return (pred-act)/(pred*(1-pred))

#Easy linkages


def Unity_link(x):
 return x

def Unity_link_grad(x):
 return 1

def Unity_delink(x):
 return x


def Root_link(x):
 return x*x

def Root_link_grad(x):
 return 2*x

def Root_delink(x):
 return x**0.5


def Log_link(x):
 return np.exp(x)

def Log_link_grad(x):
 return np.exp(x)

def Log_delink(x):
 return np.log(x)


def Logit_link(x):
 return 1/(1+np.exp(-x))

def Logit_link_grad(x):
 return np.exp(-x)/((1+np.exp(-x))**2)

def Logit_delink(x):
 return np.log(x/(1-x))

links={"Unity":Unity_link,"Root":Root_link,"Log":Log_link,"Logit":Logit_link}
linkgrads={"Unity":Unity_link_grad,"Root":Root_link_grad,"Log":Log_link_grad,"Logit":Logit_link_grad}
delinks = {"Unity":Unity_delink, "Root":Root_delink, "Log":Log_delink, "Logit":Logit_delink}

#Multiple modelling!

def Take0(*args):
 return args[0]
def Take1(*args):
 return args[1]
def Take2(*args):
 return args[2]
def Take3(*args):
 return args[3]
def Take4(*args):
 return args[4]

def js0(*args):
 return pd.Series([0]*len(args[0]))
def js1(*args):
 return pd.Series([1]*len(args[0]))

def nonefunc(*args):
 return None


def Add_mlink(*args):
 return sum(args) #Amazingly, this works!

def Add_mlink_grad(*args):
 return pd.Series([1]*len(args[0]))


def Max_mlink_2(x1, x2):
 return (x1>x2)*x1+(x1<=x2)*x2 #I am aware that this line in particular is an abomination, and refuse to care. Rewrite it at your whim, dear reader.

def Max_mlink_grad_2_A(x1,x2):
 return (x1>=x2).astype(int)

def Max_mlink_grad_2_B(x1,x2):
 return (x2>=x1).astype(int)


def Min_mlink_2(x1, x2):
 return (x1<x2)*x1+(x1>=x2)*x2 #I am aware that this line in particular is an abomination, and refuse to care. Rewrite it at your whim, dear reader.

def Min_mlink_grad_2_A(x1,x2):
 return (x1>=x2).astype(int)

def Min_mlink_grad_2_B(x1,x2):
 return (x2<=x1).astype(int)


def Mult_mlink_2(x1, x2):
 return x1*x2

def Mult_mlink_grad_2_A(x1,x2):
 return x2

def Mult_mlink_grad_2_B(x1,x2):
 return x1



#lras

def default_LRA(*args):
 return 1

def addsmoothing_LRA_A(x1,x2):
 return (sum(x1)+sum(x2))/sum(x1)

def addsmoothing_LRA_B(x1,x2):
 return (sum(x1)+sum(x2))/sum(x2)

#Tobit



def gnormal_u_diff(u, p, y):
 return -(y*(y-u)/((p**2)*(u**3)) - 1/u)

def gnormal_p_diff(u, p, y):
 return -((y-u)**2/((p**3)*(u**2)) - 1/p)

def PDF(u, p, y):
 return np.exp(-0.5*((y-u)/(p*u))**2) / (p*u*math.sqrt(2*math.pi))

def CDF(u, p, y):
 return 0.5*(1 - erf((y-u)/(p*u*math.sqrt(2))))

def u_diff_censored(u, p, y):
 return -((y/u)*PDF(u,p,y)/CDF(u,p,y))

def p_diff_censored(u, p, y):
 return -(((y-u)/p)*PDF(u,p,y)/CDF(u,p,y))