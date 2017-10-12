# Old Faithful Dataset
# Read in the data

# Imports and preliminaries
import numpy as np
import matplotlib.pyplot as pl
# https://stackoverflow.com/questions/3518778/how-to-read-csv-into-record-array-in-numpy

# This just sets the default plot size to be bigger.
pl.rcParams['figure.figsize'] = (16.0, 8.0)

# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.genfromtxt.html
eruptions = np.genfromtxt('old-faithful.csv', delimiter=',', skip_header=1, usecols=(1))
waiting = np.genfromtxt('old-faithful.csv', delimiter=',', skip_header=1, usecols=(2))

# Investigate the data

pl.plot(eruptions, waiting, 'k.', label='Original data')
pl.xlabel('Eruption Time (minutes)')
pl.ylabel('Time between Eruptions (minutes)')
pl.legend()
pl.show()

# Find the best fit line using np.polyfit()
m, c = np.polyfit(eruptions, waiting, 1)
print("Best fit is m = %f and c = %f" % (m, c))

pl.plot(eruptions, waiting, 'k.', label='Original data')
pl.plot(eruptions, m * eruptions + c, 'b-', label='Best fit (np.polyfit): $%0.1f x + %0.1f$' % (m,c))
pl.xlabel('Eruption Time (minutes)')
pl.ylabel('Time between Eruptions (minutes)')
pl.legend()
pl.show()

# Find best fit line using Gradient Descent

# Gradient Descent

def grad_m(x, y, m, c):
  return -2.0 * np.sum(x * (y - m * x - c))

def grad_c(x, y, m , c):
  return -2.0 * np.sum(y - m * x - c)

# define eta (the learning rate) - keep as a low number
# define m, c as our initial guesses
eta = 0.0001
m, c = 1.0, 1.0
change = True

while change:
  mnew = m - eta * grad_m(eruptions, waiting, m, c)
  cnew = c - eta * grad_c(eruptions, waiting, m, c)
  if m == mnew and c == cnew:
    change = False
  else:
    m, c = mnew, cnew
    print("m: %20.16f  c: %20.16f" % (m, c))

pl.plot(eruptions, waiting, 'k.', label='Original data')
pl.plot(eruptions, m * eruptions + c, 'b-', label='Best fit (Gradient Descent): $%0.1f x + %0.1f$' % (m,c))
pl.xlabel('Eruption Time (minutes)')
pl.ylabel('Time between Eruptions (minutes)')
pl.legend()
pl.show()

