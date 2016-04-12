import pandas as pd
import numpy as np

data = pd.read_csv("anclean3.csv", skiprows = 1, header=None)
data = data[[0, 1, 8]]
data = data.iloc[8:47]
data.dropna(inplace=True)
data['val'] = data.apply(lambda x: 1 if x[1] > x[8] else 0, axis = 1)

switchCost = .02;

def succ(x,p):
    return p * x[1] + (1-p) * x[8]

def fail(x,p):
    return p * x[8] + (1-p) * x[1]

def calcp(p):
    val = data.apply(lambda x: 1 if x[1] > x[8] else 0, axis = 1)
    suc = succ(data,p)
    fai = fail(data,p)
    tr = val*suc + (1 - val) * fai

    pinvst = val * p + (1 - val)*(1-p)
    pinvst2= pinvst.shift()
    pinvst2.fillna(0, inplace = True)
    pswitch = pinvst2 + pinvst - 2 * pinvst * pinvst2
    fingain = tr - switchCost * pswitch
    return pd.Series({'mean': fingain.mean(), 'zmeanraw': tr.mean()})

def stdev(p):
    q = 1 - p
    val = data['val']
    pgood = (val.sum() + 0.0) / val.size
    dg = data[[1,8]].groupby(val)
    means = dg.aggregate([np.mean, np.std])

    stock = means[1]
    tbill = means[8]

    pstock = p * pgood + q * (1 - pgood)
    pswitch = 2 * pstock * (1 - pstock)

    mean = p * pgood * stock['mean'].iloc[1] + \
           q * pgood * tbill['mean'].iloc[1] + \
           p * (1 - pgood) * tbill['mean'].iloc[0] + \
           q * (1 - pgood) * stock['mean'].iloc[0]

    std1 = p * pgood * stock['std'].iloc[1]**2 + \
           q * pgood * tbill['std'].iloc[1]**2 +\
           p * (1 - pgood) * tbill['std'].iloc[0]**2 + \
           q * (1 - pgood) * stock['std'].iloc[0]**2

    std2 = p * pgood * (stock['mean'].iloc[1] - mean)**2 + \
           q * pgood * (tbill['mean'].iloc[1] - mean)**2 + \
           p * (1 - pgood) * (tbill['mean'].iloc[0] - mean)**2 + \
           q * (1 - pgood) * (stock['mean'].iloc[0] - mean)**2

    std = (std1 + std2)**(.5)

    adjMean = mean - pswitch * switchCost
    return pd.Series({'std': std, 'madjMean': adjMean, 'munadjMean':mean})

#mean is calculated differently from standard deviation:
ans = pd.DataFrame({'p' : range(50, 101)}) * .01
ans[['Mean', 'Raw Mean']] = ans['p'].apply(calcp) * 100
ans[['Mean2','Raw Mean 2', 'Stdev']] = ans['p'].apply(stdev) * 100


ans.to_csv("answer.csv")
print ans
# print data







