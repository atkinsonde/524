# -*- coding: utf-8 -*-
"""
Testing and monte carlo class
October 18, 2019
"""
import numpy as np
'''
Binomial test Example 5.1
'''
Claim = 6/7 # cloud-free day proportion
Obs = 15/25 # actual observations

'''
Null hypothesis: assertion correct and proportion = 6/7
alternative: assertion incorrect and proportion is less than 6/7
not interested in propotion > 6/7, so one-tailed test

What is the sampling distribution to test against?

Test statistic can be thought of as a series of 0 and 1 for no-cloud/cloud
This makes it a bionmial distribution with p=6/7 and N=25

Now need to compute the probability that 15 or fewer cloudless days are observed in 25, given a "true" p=6/7

Need to run a binomial calculation for all values 25 choose 0-15 and sum these individual probabilities
'''

n=25
p=6/7
k=np.arange(0,16) # this is what we want

# play with it for a bit to get a feel
k=22
ss.binom.pmf(k, n, p)

# We can use a list comprehension to create binomial probabilities for
# each of values 0-15. 
[ss.binom.pmf(k, n, p) for k in np.arange(0,16)]

# ...and simply place all of this inside a np.sum() function
val = np.sum([ss.binom.pmf(k, n, p) for k in np.arange(0,16)])

# Or using the k=np.arange(0,16) we can vectorize the result
np.sum(ss.binom.pmf(k,n,p))

''''''''''''''''''''''''


import pandas as pd
import numpy as np
import scipy.stats as ss

NB_FRD = pd.read_csv(r"C:\Users\Windows\Documents\courses\GEOG_524\Monte_carlo\en_climate_daily_NB_8101505_2019_P1D.csv",
                  usecols=["Mean Temp (°C)","Date/Time"], parse_dates=['Date/Time']).set_index('Date/Time')

NB_STJ = pd.read_csv(r"C:\Users\Windows\Documents\courses\GEOG_524\Monte_carlo\en_climate_daily_NB_8104901_2019_P1D.csv",
                  usecols=["Mean Temp (°C)","Date/Time"], parse_dates=['Date/Time']).set_index('Date/Time')

# subset summers
NB_FRD_sum = NB_FRD['2019-05-01':'2019-08-31']
NB_STJ_sum = NB_STJ['2019-05-01':'2019-08-31']

# prep for concatenate
NB_FRD_sum = NB_FRD_sum.rename(columns={"Mean Temp (°C)": "FRD_T"})
NB_STJ_sum = NB_STJ_sum.rename(columns={"Mean Temp (°C)": "STJ_T"})

# concatenate
NB_join = pd.concat([NB_FRD_sum,NB_STJ_sum],axis=1).dropna()  #linreg won't run with nans

NB_join.plot()
NB_join.plot.scatter('FRD_T','STJ_T',title="NB temp")


'''
Wilcoxan rank sum
> are the means different?
'''
# Null is they are the same

np.mean(NB_join)
np.std(NB_join)

# two-sample t-test 
ss.ttest_ind(NB_join['FRD_T'],NB_join['STJ_T'])

# The difference between this and the t-test is the Wilcoxon test is paired.
# Looking at the Fredricton/Saint John datasets in aggregate, it appears
# that their means may be very similar. However, at almost every time point,
# Fredricton is a little warmer than Saint John. Thus, if we constrain a 
# comparison of means to be built each pair at a time, we can see that
# Fredricton is definitely warmer that Saint John. THe results of a 
# t-test (not so bad) and Wilcoxon (very strong result) reflect this.
ss.wilcoxon(NB_join['FRD_T'],NB_join['STJ_T'])


'''
Monte carlo 
Use Saint John to predict Fredricton summer daily mean T
> what is the minimum # days needed to do this?

'''
# function to extract X random pairs and perform regression. 
import statsmodels.api as sm

def reg(pairs,dataset):
    ds = dataset.sample(pairs)  # pandas has a dataframe random sample method!
    model = sm.OLS(ds['FRD_T'], ds['STJ_T']).fit() #goes as (y,x), not (x,y)
    slope = model.params #run dir(model) - there are many parameters returned, but all we want is slope
    return slope

# run for many experiments
exptaccum=[]
# remember to reinitialize fullset between runs
fullset = np.empty((0,3), float) # create an empty array of appropriate dimensions (3 columns)

for expt in np.arange(2,50):
    for samp in np.arange(1,500):
        oneexpt = reg(expt,NB_join)
        exptaccum=np.append(exptaccum,oneexpt)
        zz=[[expt,np.mean(exptaccum),np.std(exptaccum)]]
        #print(zz)
    np.vstack((fullset,zz))    
    fullset = np.concatenate((fullset,[[expt,np.mean(exptaccum),np.std(exptaccum)]]),axis=0)
    #print(np.mean(exptaccum),np.std(exptaccum))
    exptaccum=[]
    
fullset_500 = fullset    

pd.DataFrame(fullset).plot(0,[1])    

#1000 samples, 2-100 = 5:20 min to run on my computer
# I have not attempted to optimize this routine.

f1=pd.DataFrame(fullset_10)
f2=pd.DataFrame(fullset_50)
f3=pd.DataFrame(fullset_500)
   
# You would *really* want to properly set these datasets up, so that 
# you have proper column names. Remember use of "axis=1" when you want 
# to make the concat across columns, (i.e. axis=0 makes one big column) 
all_mn =pd.concat([f1[1],f2[1],f3[1]],axis=1) 
all_std=pd.concat([f1[2],f2[2],f3[2]],axis=1)

all_mn.plot()
NB_join.plot()




















# Note the difference in argument order
model = sm.OLS(NB_join['FRD_T'], NB_join['STJ_T']).fit() #goes as (y,x)
predictions = model.predict(NB_join['STJ_T']) # make the predictions by the model
pd.DataFrame(predictions).plot()





dir(model)
slope = model.params

# Print out the statistics
modzz = model.summary()
