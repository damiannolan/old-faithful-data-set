
# coding: utf-8

# # Old Faithful Dataset

# ## Read in the data

# In[1]:


# Imports and preliminaries
import numpy as np
import matplotlib.pyplot as pl
# https://stackoverflow.com/questions/3518778/how-to-read-csv-into-record-array-in-numpy

# This just sets the default plot size to be bigger.
pl.rcParams['figure.figsize'] = (16.0, 8.0)


# In[9]:


# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.genfromtxt.html
eruptions = np.genfromtxt('old-faithful.csv', delimiter=',', skip_header=1, usecols=(1))
waiting = np.genfromtxt('old-faithful.csv', delimiter=',', skip_header=1, usecols=(2))

#could be used to filter nan
#old_faithful = old_faithful[np.logical_not(np.isnan(old_faithful))]

eruptions


# In[10]:


waiting


# ## Investigate the data

# In[13]:


pl.plot(eruptions, waiting, 'k.', label='Original data')
pl.xlabel('Eruption Time (minutes)')
pl.ylabel('Time between Eruptions (minutes)')
pl.legend()
pl.show()


# ## Finding the best fit line

# ### Find the best fit line using np.polyfit()

# In[28]:


# Find best fit line
m, c = np.polyfit(eruptions, waiting, 1)
print("Best fit is m = %f and c = %f" % (m, c))


# In[29]:


pl.plot(eruptions, waiting, 'k.', label='Original data')
pl.plot(eruptions, m * eruptions + c, 'b-', label='Best fit (np.polyfit): $%0.1f x + %0.1f$' % (m,c))
pl.xlabel('Eruption Time (minutes)')
pl.ylabel('Time between Eruptions (minutes)')
pl.legend()
pl.show()


# ### Find best fit line using Gradient Descent

# In[30]:


# Gradient Descent

def grad_m(x, y, m, c):
  return -2.0 * np.sum(x * (y - m * x - c))

def grad_c(x, y, m , c):
  return -2.0 * np.sum(y - m * x - c)


# In[19]:


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


# In[26]:


pl.plot(eruptions, waiting, 'k.', label='Original data')
#pl.plot(m, 'b-', label='Slope: %f' % c)
pl.plot(eruptions, m * eruptions + c, 'b-', label='Best fit (Gradient Descent): $%0.1f x + %0.1f$' % (m,c))
pl.xlabel('Eruption Time (minutes)')
pl.ylabel('Time between Eruptions (minutes)')
pl.legend()
pl.show()

