
import pandas as pd
df = pandas.read_csv('1871433.csv',
                         usecols=["DATE","PRCP","SNOW","SNWD","TMAX","TMIN"],
                         parse_dates=["DATE"]).set_index("DATE")

# determine if a given year has any occurrences that satisfy the condition
mask = pd.DataFrame(df["TMIN"]<-25).resample("Y").max()

# isolate years that have some minimum number of observations per year
full_years_TMIN = pd.DataFrame(df["TMIN"].resample("Y").count()>350)

# pull out only the year identifiers that meet the threshold
fyt = full_years_TMIN[full_years_TMIN["TMIN"]==True].drop("TMIN")

# merge the occurrence data with the reduced year set
year_counts = fyt.drop("TMIN",axis=1).join(mask,how="left")

# Determine probability that an event occurs in a given year
P = year_counts["TMIN"].sum()/len(year_counts)

# Want to know probability of getting x cold winters in a N year period

import scipy.special as sm

N = 10
x = 2

P=.2

sum([sm.comb(N,x)*P**x*(1-P)**(N-x) for x in range(1,11)])



# range of values, plot, title
xr = range(0,10)

import scipy.stats as ss
pd.DataFrame(ss.binom.pmf(xr, N, P)).plot(title='10 years, P=.17') 

