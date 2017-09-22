import numpy.random as nr
from math import log, pi, exp
import lnn 

def lnn_samplor(length):
    tmpList = np.array(ITM_samplor(length)[0]).reshape(length,1)
    return tmpList
    
fig,ax = plt.subplots()
length = [100]
true = 0.5*(1-4*np.log(2))/np.log(2)
m = 100
errorSet = []

for _ in range(m):
    for l in length:
        data = lnn_samplor(l)
        H = lnn.entropy(data)
        err = np.sqrt(np.mean(np.square(H-true)))
        errorSet.append(err)
        #print ('data size =', l)
        #print ("Ground Truth = ", true)
        #print ("LNN: H(X) =  ", H)
        #print ("RMSE = ", err)
        
ax.plot(range(m), errorSet, 's-', linewidth=1.5)
ax.set_title('n = 10000, m = 100, lnn.entropy')

mc_times = 50 # Total number of Monte-Carlo trials for each alphabet size
#record_n = np.array([100000, 10000, 1000, 100]) # sample size 
record_n = np.array([100])
record_h = [1/3]
num = len(record_h)


JVHW_err = np.zeros(num)
total_time = []
err = 0
errset1 = []
#fig,ax = plt.subplots()
for order in range(m):
    for n in record_n:
        start = time.clock()
        #tmpSamplor = ARM_samplor(n, m = mc_times) # ARM sampling is much slower than ITM
        tmpSamplor = np.array(ITM_samplor(n, m = mc_times)).reshape(n,1)
        elapsed = (time.clock() - start)
        #print('Sampling of n='+str(n)+' has finished. Time use:',elapsed)
        total_time.append(elapsed)
        start = time.clock()
        for i, h in enumerate(record_h):
            S = int(1/h)
            edges = np.linspace(0,1,S+1)
            samp = np.digitize(tmpSamplor, edges)
            record_JVHW = est_entro_JVHW(samp) - np.log(S)
            JVHW_err[i] = np.sqrt(np.mean(np.square(record_JVHW - true)))
            errset1.append(np.mean(np.square(record_JVHW - true)))
            if (h == 1/3):
                err = JVHW_err[i]

        elapsed = (time.clock() - start)
        #print('Calculation of n='+str(n)+' has finished. Time use:',elapsed)
        total_time.append(elapsed)
        
        
        start = time.clock()
        #print(n,' ',JVHW_err)
        #ax.plot(record_h, JVHW_err, 's-', linewidth=1.5)
        #ax.plot(order, JVHW_err, 's-', linewidth=1.5)
        elapsed = (time.clock() - start)
        #print('Plotting of n='+str(n)+' has finished. Time use:',elapsed)
        total_time.append(elapsed)
errset1 = np.array(errset1).reshape(100,1)
#ax.plot(range(100), errset1, 's-', linewidth=1.5)
#ax.set_title('n = 1000, m = 100, SHVW.entropy')

fig,ax = plt.subplots()
ax.plot(range(m), errorSet, 's-', linewidth=1.5)
ax.plot(range(m), errset1, 's-', linewidth=1.5)
ax.legend(['lnn','SHVW'])
ax.set_title('n = 100, m = 100')
ax.set_ylabel('RMSE')
