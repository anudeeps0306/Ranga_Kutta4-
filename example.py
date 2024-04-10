import numpy as np
import matplotlib.pyplot as plt

# Initialise step-size variables
h = 0.5
t = np.arange(0, 2.5, h)

N = len(t)

# Initialise vectors
x = np.zeros(N)
y = np.zeros(N)

# Starting conditions
x[0] = 4
y[0] = 6

# Define derivative functions
def dx(t, x, y, z):
    return (-0.5)*(x)

def dy(t, x, y, z):
    return 4-(0.3*y)-(0.1*x)

def dz(t, x, y, z):
    return -x - np.exp(t) + 1

# Initialise K vectors
kx = np.zeros(4)
ky = np.zeros(4)
b = np.array([1, 2, 2, 1])

for i in range(N-1):
    kx[0] = dx(t[i], x[i], y[i], z[i])
    ky[0] = dy(t[i], x[i], y[i], z[i])
    kz[0] = dz(t[i], x[i], y[i], z[i])

    kx[1] = dx(t[i] + (h/2), x[i] + (h/2)*kx[0], y[i] + (h/2)*ky[0], z[i] + (h/2)*kz[0])
    ky[1] = dy(t[i] + (h/2), x[i] + (h/2)*kx[0], y[i] + (h/2)*ky[0], z[i] + (h/2)*kz[0])
    kz[1] = dz(t[i] + (h/2), x[i] + (h/2)*kx[0], y[i] + (h/2)*ky[0], z[i] + (h/2)*kz[0])

    kx[2] = dx(t[i] + (h/2), x[i] + (h/2)*kx[1], y[i] + (h/2)*ky[1], z[i] + (h/2)*kz[1])
    ky[2] = dy(t[i] + (h/2), x[i] + (h/2)*kx[1], y[i] + (h/2)*ky[1], z[i] + (h/2)*kz[1])
    kz[2] = dz(t[i] + (h/2), x[i] + (h/2)*kx[1], y[i] + (h/2)*ky[1], z[i] + (h/2)*kz[1])

    kx[3] = dx(t[i] + h, x[i] + h*kx[2], y[i] + h*ky[2], z[i] + h*kz[2])
    ky[3] = dy(t[i] + h, x[i] + h*kx[2], y[i] + h*ky[2], z[i] + h*kz[2])
    kz[3] = dz(t[i] + h, x[i] + h*kx[2], y[i] + h*ky[2], z[i] + h*kz[2])


    x[i+1] = x[i] + (h/6)*np.sum(b*kx)
    y[i+1] = y[i] + (h/6)*np.sum(b*ky)
    z[i+1] = z[i] + (h/6)*np.sum(b*kz)

# Group together in one solution matrix
txyz = np.column_stack((t, x, y))
print(txyz)



# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Solution of Differential Equations using RK4')
plt.legend()
plt.grid(True)
plt.show()