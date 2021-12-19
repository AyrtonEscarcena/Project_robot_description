import numpy as np
from copy import copy
import rospy

cos=np.cos; sin=np.sin; pi = np.pi


#def dh(d, theta, a, alpha):
#  sth = np.sin(theta)
#  cth = np.cos(theta)
#  sa  = np.sin(alpha)
#  ca  = np.cos(alpha)
#  T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
#              [sth,  ca*cth, -sa*cth, a*sth],
#               [0.0,      sa,      ca,     d],
#                [0.0,     0.0,     0.0,   1.0]])
#  return T
def dh(d, theta, a, alpha):
  cth = cos(theta);sth=sin(theta)
  ca = cos(alpha); sa= sin(alpha)
  T = np.array([[cth, -ca*sth, sa*sth, a*cth],
                [sth, ca*cth, -sa*cth, a*sth],
                [0, sa, ca, d],
                [0,0,0,1]])
 
  return T

#Obtencion de la Matriz Homogenea del origen al efector final (posicion del efector final): (0-T-6)
def fkine_ur5(q):
  # Longitudes (en metros)
  l1=0.0892; l2=0.425; l3=0.392; l4=0.1093; l5=0.09475; l6=0.0825
 
  # Matrices DH (completar), emplear la funcion dh con los parametros DH para cada articulacion
  T1 = dh( l1, q[0], 0, pi/2)
  T2 = dh( l4, q[1]+pi, l2, 0)
  T3 = dh( -l4, q[2], l3, 0)
  T4 = dh( l4, q[3]+pi, 0, pi/2)
  T5 = dh( l5, q[4]+pi, 0, pi/2)
  T6 = dh( l6, q[5], 0, 0)
  # Efector final con respecto a la base
  T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6)
  return T


#Obtencion del Jacobiano (se recibe el q actual y la variacion que se le hara)
def jacobian_ur5(q, delta=0.0001):
  """
  Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
  entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5, q6]
  """
  # Crear una matriz 3x6
  J = np.zeros((3,6))
  # Transformacion homogenea (0-T-6) usando el q dado
  T = fkine_ur5(q)
  # Iteracion para la derivada de cada columna
  for i in xrange(6):
    # Copiar la configuracion articular(q) y almacenarla en dq
    dq = copy(q)
    # Incrementar los valores de cada q sumandoles un delta a cada uno
    dq[i] = dq[i] + delta
    # Obtencion de la nueva Matriz Homogenea con los nuevos valores articulares, luego del incremento (q+delta)
    Td = fkine_ur5(dq)
    # Aproximacion del Jacobiano de posicion usando diferencias finitas
    for j in xrange(3):
      J[j][i] = (Td[j][3]-T[j][3])/delta
  return J


def jacobian_pose(q, delta=0.0001):
  """
  Jacobiano analitico para la posicion y orientacion (usando un
  cuaternion). Retorna una matriz de 7x6 y toma como entrada el vector de
  configuracion articular q=[q1, q2, q3, q4, q5, q6]

  """
  J = np.zeros((7,6))
  J1 = np.zeros((3,6))
  J2 = np.zeros((4,6))
  # Transformacion homogenea (0-T-6) usando el q dado
  T = fkine_ur5(q)
  R = T[0:3,0:3]
  Q = rot2quat(R)
  Q = Q.T

  # Iteracion para la derivada de cada columna
  for i in range(6):
    # Copiar la configuracion articular(q) y almacenarla en dq
    dq = copy(q)
    # Incrementar los valores de cada q sumandoles un delta a cada uno
    dq[i] = dq[i] + delta
    # Obtencion de la nueva Matriz Homogenea con los nuevos valores articulares, luego del incremento (q+delta)
    Td = fkine_ur5(dq)
    #print(Td)
    Rd = Td[0:3,0:3]
    Qd = rot2quat(Rd)
    Qd = Qd.T
    #print(Qd)

    for j in range(0,3):
      J1[j][i] = (Td[j][3]-T[j][3])/delta
    for j in range(0,4):
      J2[j][i] = (Qd[j]-Q[j])/delta
    J = np.hstack((J1, J2))
  return J