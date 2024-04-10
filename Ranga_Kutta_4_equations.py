import numpy as np
import matplotlib.pyplot as plt

# Initialise step-size variables
h = 0.1
t = np.arange(0, 1 + h, h)

N = len(t)

# Initialise vectors
x = np.zeros(N)
y = np.zeros(N)
z = np.zeros(N)

# Starting conditions
x[0] = 1
y[0] = 0
z[0] = 1

# Define derivative functions
def dx(t, x, y, z):
    return y

def dy(t, x, y, z):
    return -x - 2*np.exp(t) + 1

def dz(t, x, y, z):
    return -x - np.exp(t) + 1

# Initialise K vectors
kx = np.zeros(4)
ky = np.zeros(4)
kz = np.zeros(4)
b = np.array([1, 2, 2, 1])

# Iterate, computing each K value in turn, then the i+1 step values
for i in range(N-1):
    kx[0] = h*dx(t[i], x[i], y[i], z[i])
    ky[0] = h*dy(t[i], x[i], y[i], z[i])
    kz[0] = h*dz(t[i], x[i], y[i], z[i])

    kx[1] = h*dx(t[i] + (h/2.), x[i] + (1/2.)*kx[0], y[i] + (1/2.)*ky[0], z[i] + (1/2.)*kz[0])
    ky[1] = h*dy(t[i] + (h/2.), x[i] + (1/2.)*kx[0], y[i] + (1/2.)*ky[0], z[i] + (1/2.)*kz[0])
    kz[1] = h*dz(t[i] + (h/2.), x[i] + (1/2.)*kx[0], y[i] + (1/2.)*ky[0], z[i] + (1/2.)*kz[0])

    kx[2] = h*dx(t[i] + (h/2.), x[i] + (1/2.)*kx[1], y[i] + (1/2.)*ky[1], z[i] + (1/2.)*kz[1])
    ky[2] = h*dy(t[i] + (h/2.), x[i] + (1/2.)*kx[1], y[i] + (1/2.)*ky[1], z[i] + (1/2.)*kz[1])
    kz[2] = h*dz(t[i] + (h/2.), x[i] + (1/2.)*kx[1], y[i] + (1/2.)*ky[1], z[i] + (1/2.)*kz[1])

    kx[3] = h*dx(t[i] + h, x[i] + kx[2], y[i] + ky[2], z[i] + kz[2])
    ky[3] = h*dy(t[i] + h, x[i] + kx[2], y[i] + ky[2], z[i] + kz[2])
    kz[3] = h*dz(t[i] + h, x[i] + kx[2], y[i] + ky[2], z[i] + kz[2])

    x[i+1] = x[i] + (1/6.)*np.sum(b*kx)
    y[i+1] = y[i] + (1/6.)*np.sum(b*ky)
    z[i+1] = z[i] + (1/6.)*np.sum(b*kz)

# Group together in one solution matrix
txyz = np.column_stack((t, x, y, z))
print(txyz)


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.plot(t, z, label='y(t)')
plt.xlabel('Steps')
plt.ylabel('Values')
plt.title('Solution of Differential Equations using RK4')
plt.legend()
plt.grid(True)
plt.show()