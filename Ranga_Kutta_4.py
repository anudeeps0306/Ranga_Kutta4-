import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Example function, taking a pendulum as an example
def pend(y, t, b, c):
    return np.array([y[1], -b*y[1] - c*np.sin(y[0])])

def rungekutta1(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        y[i+1] = y[i] + (t[i+1] - t[i]) * f(y[i], t[i], *args)
    return y

def rungekutta2(f, y0, t, args=()):
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        y[i+1] = y[i] + h * f(y[i] + f(y[i], t[i], *args) * h / 2., t[i] + h / 2., *args)
    return y

def runge_kutta_4(f, y0, t, args=()):
    """
    Fourth-order Runge-Kutta numerical solver.

    Parameters:
        f (callable): Function defining the differential equation.
        y0 (array-like): Initial state vector.
        t (array-like): Array of time points.
        args (tuple, optional): Additional arguments to pass to the function f.

    Returns:
        array-like: Solution array containing the state vectors at each time point.
    """
    n = len(t)
    y = np.zeros((n, len(y0)))
    y[0] = y0
    for i in range(n - 1):
        h = t[i+1] - t[i]
        k1 = f(y[i], t[i], *args)
        k2 = f(y[i] + k1 * h / 2., t[i] + h / 2., *args)
        k3 = f(y[i] + k2 * h / 2., t[i] + h / 2., *args)
        k4 = f(y[i] + k3 * h, t[i] + h, *args)
        y[i+1] = y[i] + (h / 6.) * (k1 + 2*k2 + 2*k3 + k4)
    return y

def ploting_wth_different_interval(b, c, y0):
    """
    Plot the motion of a pendulum with different time intervals.

    Parameters:
        b (float): Damping factor.
        c (float): Coefficient of friction.
        y0 (array-like): Initial state vector.
    """
    t4_21 = np.linspace(0, 10, 21)
    ranga_21 = runge_kutta_4(pend, y0, t4_21, args=(b, c))
    t4_101 = np.linspace(0, 10, 101)
    ranga_101 = runge_kutta_4(pend, y0, t4_101, args=(b, c))
    t4_1001 = np.linspace(0, 10, 1001)
    ranga_1001 = runge_kutta_4(pend, y0, t4_1001, args=(b, c))
    plt.plot(t4_21, ranga_21[:, 0], label='with 21 points')
    plt.plot(t4_101, ranga_101[:, 0], label='with 101 points')
    plt.plot(t4_1001, ranga_1001[:, 0], label='with 1001 points')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()

b = 0.25
c = 5.0
y0 = np.array([np.pi - 0.1, 0.0])


def test_1(n=20):
    t = np.linspace(0, 10, n)
    methods = [odeint, rungekutta1, rungekutta2, runge_kutta_4]
    markers = ['+', 'o', 's', '>']
    for method, m in zip(methods, markers):
        sol = method(pend, y0, t, args=(b, c))
        plt.plot(t, sol[:, 0], label=method.__name__, marker=m)
    plt.legend(loc='best')
    plt.title("Comparison of different ODE integration methods for $n={}$ points".format(n))
    plt.xlabel("$t = [0, 10]$")
    plt.grid()
    plt.show()

def test_2():
    ploting_wth_different_interval(b, c, y0)

def test_3():
    t = np.linspace(0, 10, 101)
    for method in [rungekutta1, rungekutta2, runge_kutta_4]:
        sol = method(pend, y0, t, args=(b, c))
        plt.plot(t, sol[:, 0], label=method.__name__)
    plt.legend(loc='best')
    plt.title("Comparison of different Runge-Kutta methods")
    plt.xlabel("t=[0,10]")
    plt.ylabel("y")
    plt.grid()
    plt.show()

while True:
    print("Choose an option:")
    print("1. Compare ODE integration methods")
    print("2. Plot motion of a pendulum with different time intervals")
    print("3. Compare different Runge-Kutta methods")
    print("4. Exit")
    option = input("Enter your choice: ")

    if option == "1":
        test_1()
    elif option == "2":
        test_2()
    elif option == "3":
        test_3()
    elif option == "4":
        print("Exiting...")
        break
    else:
        print("Invalid option. Please choose again.")
