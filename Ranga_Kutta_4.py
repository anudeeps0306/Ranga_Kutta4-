import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time

def pend(y, t, b, c):
    """
    Defines the differential equation for a simple pendulum.

    Parameters:
        y (array-like): The state vector.
        t (float): The time.
        b (float): Damping factor.
        c (float): Coefficient of friction.

    Returns:
        array-like: The derivative of the state vector.
    """
    return np.array([y[1], -b * y[1] - c * np.sin(y[0])])

def rungekutta1(f, y0, t, args=()):
    """
    Implement the first-order Runge-Kutta numerical solver.

    Parameters:
        f (callable): The function defining the differential equation.
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
        y[i+1] = y[i] + (t[i+1] - t[i]) * f(y[i], t[i], *args)
    return y

def rungekutta2(f, y0, t, args=()):
    """
    Implement the second-order Runge-Kutta numerical solver.

    Parameters:
        f (callable): The function defining the differential equation.
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
        s1 = h*f(y[i], t[i], *args)
        s2 = h*f(y[i] + s1 * 1 / 2., t[i] + h / 2., *args)
        s3 = h*f(y[i] + s2 * 1 / 2., t[i] + h / 2., *args)
        s4 = h*f(y[i] + s3 * 1, t[i] + h, *args)
        y[i+1] = y[i] + (1 / 6.) * (s1 + 2*s2 + 2*s3 + s4)
    return y

def ploting_wth_different_interval(b, c, y0):
    """
    Plot the motion of a pendulum with different time intervals.

    Parameters:
        b (float): Damping factor.
        c (float): Coefficient of friction.
        y0 (array-like): Initial state vector.
    """
    test_21 = np.linspace(0, 10, 21)
    runge_21 = runge_kutta_4(pend, y0, test_21, args=(b, c))
    test_101 = np.linspace(0, 10, 101)
    runge_101 = runge_kutta_4(pend, y0, test_101, args=(b, c))
    test_1001 = np.linspace(0, 10, 1001)
    runge_1001 = runge_kutta_4(pend, y0, test_1001, args=(b, c))
    plt.plot(test_21, runge_21[:, 0], label='with 21 points')
    plt.plot(test_101, runge_101[:, 0], label='with 101 points')
    plt.plot(test_1001, runge_1001[:, 0], label='with 1001 points')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.grid()
    plt.show()

b = 0.25
c = 5.0
y0 = np.array([np.pi - 0.1, 0.0])

def benchmark_methods():
    methods = [odeint,rungekutta1, rungekutta2, runge_kutta_4]

    for n in [20, 100, 1000, 10000, 100000]:
        print("\n")
        t = np.linspace(0, 3, n)
        for method in methods:
            print("Time of solving this ODE for {} points with {} method...".format(n, method.__name__))
            start_time = time.time()
            sol = method(pend, y0, t, args=(b, c))
            end_time = time.time()
            print("Execution time:", end_time - start_time, "seconds")

def test_1(n=20):
    """
    Compare ODE integration method with scipy library and normal function.

    Parameters:
        n (int, optional): Number of time points to use in the comparison.
    """
    t = np.linspace(0, 10, n)

    # Using odeint from scipy
    sol_odeint = odeint(pend, y0, t, args=(b, c))

    # Using the normal function (in this case, using the fourth-order Runge-Kutta method)
    sol_rk4 = runge_kutta_4(pend, y0, t, args=(b, c))

    # Plotting the results
    plt.plot(t, sol_odeint[:, 0], label='ODE Integration (odeint)')
    plt.plot(t, sol_rk4[:, 0], label='Normal Function (Runge-Kutta)')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('y')
    plt.title('Comparison between odeint and normal function')
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
    print("1. Compare ODE integration method with scipy library")
    print("2. Plot motion of a pendulum with different time intervals")
    print("3. Exit")
    option = input("Enter your choice: ")

    if option == "1":
        test_1()
    elif option == "2":
        test_2()
    elif option == "3":
        print("Exiting...")
        break
    else:
        print("Invalid option. Please choose again.")
