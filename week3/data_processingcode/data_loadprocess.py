# Python code to Rescale data (between 0 and 1) 
import pandas 
import scipy 
import numpy 
from sklearn.preprocessing import MinMaxScaler 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['preg', 'plas', 'pres', 'skin',  'class'] 
dataframe = pandas.read_csv(url, names=names) 
array = dataframe.values 
print(array)

# separate array into input and output components 
X = array[:,0:4] 
Y = array[:,4] 
scaler = MinMaxScaler(feature_range=(0, 1)) 
rescaledX = scaler.fit_transform(X) 

# summarize transformed data 
numpy.set_printoptions(precision=3) 
print(rescaledX[0:4,:]) 

