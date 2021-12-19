import numpy as np
from copy import copy
#<param name="use_gui" value="$(arg gui)"/>
# <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher"/>

cos=np.cos; sin=np.sin; pi=np.pi


def dh(d, theta, a, alpha):
    # Calcular la matriz de transformacion homogenea asociada con los parametros
    # de Denavit-Hartenberg.

    cth = np.cos(theta)
    sth = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    T = np.array([   [cth, -ca*sth,  sa*sth, a*cth],
                     [sth,  ca*cth, -sa*cth, a*sth],
                     [0,        sa,     ca,      d],
                     [0,         0,      0,      1]])
    return T
    
    

def fkine_ur5(q):
    #d,theta ,a, alfa
    T1 = dh(0.18,q[0],0,0)
    T2 = dh(0.0,0,0.18,q[1])
    T3 = dh(0.38,q[2],0.0,0)
    T4 = dh(0.0,0,0.28,q[3])
    T5 = dh(0.23,q[4],0,0)
    T6 = dh(0.0,0,0.18,q[5]) 
    T7 = dh(-0.140,q[6],0.0,0)
    #T8 = dh(-0.100,0,0,0)
    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6).dot(T7)#.dot(T8)

    return T
def jacobian_ur5(q, delta=0.0001):

#Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
#entrada el vector de configuracion articular q=[q1, q2, q3, q4,q5, q6, q7]

    J = np.zeros((3,7))
    # Transformacion homogenea inicial (usando q)
    Tq = fkine_ur5(q)
    # Iteracion para la derivada de cada columna
    for i in xrange(7):
    # Copiar la configuracion articular inicial
        dq = copy(q)
        vdelta = np.zeros(7)
        vdelta[i] = delta
    # Incrementar la articulacion i-esima usando un delta
        qd = dq + vdelta
    # Transformacion homogenea luego del incremento (q+delta)
        Tqd = fkine_ur5(qd)
    # Aproximacion del Jacobiano de posicion usando diferencias finitas
        xyzq = Tq[0:3,3]
        xyzqd = Tqd[0:3,3]
        J[:,i] = (xyzqd - xyzq)/delta
    return J
def ikine_ur5(xdes, q0):

# Calcular la cinematica inversa de UR5 numericamente a partir de
# la configuracion articular inicial de q0.
# Emplear el metodo de newton

    epsilon = 0.0001
    max_iter = 200
    delta = 0.00001
    q = copy(q0)
    itr = 0
    for i in range(max_iter):
    # Main loop
        J = jacobian_ur5 (q, delta)
        f = fkine_ur5(q)
        f = f[0:3,3]
        e = xdes - f
        q = q + np.dot(np.linalg.pinv(J), e)
        itr = itr + 1
        # Condicion de termino
        if (np.linalg.norm(e) < epsilon):
            print('Iteraciones totales: ')
            print( itr )
            print('Error en la iteracion {}:{}'.format(itr,np.round(np.linalg.norm(e),4)))
            break
        print('Valores articuladores: ')
        print(str(np.round(q,3)) + '[rads]')
    return q

