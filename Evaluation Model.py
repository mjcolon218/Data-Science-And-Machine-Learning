from sklearn.datasets import load_boston#from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error#from sklearn.metrics import mean_squared_error 
import pandas as pd#import pandas as pd to convert into dataframes which are matrices
import numpy as np # handles numerical data for calculations now these and know them well maurice 


boston_dataset = load_boston()
data = pd.DataFrame(data = boston_dataset.data, columns = boston_dataset.feature_names)
dir(boston_dataset)
print(boston_dataset.DESCR)

features = data.drop(['INDUS','AGE'],axis = 1)#Dropping two Rows Axis1 is the column 0 would be the whole row 
log_prices = np.log(boston_dataset.target)#Using log method and passing variable with a chain to target which are the values
#we want converted, this is how we get DOLLAR PRICES BY TAKING THE LOG
features.shapeCRIME_IDX = 0 #assigning numbers to features so i can make it easier when im using the indices and accessing them 
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8
target = pd.DataFrame(log_prices,columns = ['PRICE'])# we convert the prices to log so we can read them and we use this at our target
log_prices.shape
target.shape


property_stats = features.mean().values.reshape(1,11)

#using sklearns linear Regression
regr =  LinearRegression().fit(features,target)#fitting regression line on our data model/calculates all theta values
fitted_vals=regr.predict(features)#calculating all the predicted values using theta values from up above 
#CHALLENE CALCULATE MSE  RMSE USING THE MODULE
MSE = mean_squared_error(target,fitted_vals)

RMSE =np.sqrt(MSE)#use Np to access the sqare root method on your MSE object u created 

def get_log_estimate(nr_rooms=3,students_per_classroom=6,next_to_river=False,high_confidence=True):
    #log_estimate = regr.predict(property_stats)
    #configure property
    property_stats[0][RM_IDX]=nr_rooms
    property_stats[0][PTRATIO_IDX]=students_per_classroom
    if next_to_river:
        property_stats[0][CHAS_IDX]=1
    else:
        property_stats[0][CHAS_IDX]=0
    
    #make Prediction
    log_estimate= regr.predict(property_stats)[0][0]
    #Calculate Range
    if high_confidence:
        #do X
        upper_bound = log_estimate + 2 *RMSE
        lower_bound = log_estimate - 2 *RMSE
        interval = 95
        
    else:#DO Y
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    return log_estimate,upper_bound,lower_bound,interval
zillow_median_price = 583.3
scale_factor = zillow_median_price / np.median(boston_dataset.target)#this what you multiply it by 
log_est,upper,lower,conf = get_log_estimate(9,students_per_classroom=15,next_to_river=False,high_confidence =False)


#Convert to todays dollar
dollar_est = np.e**log_est*1000*scale_factor
dollar_hi = np.e**upper*1000*scale_factor
dollar_low = np.e**lower*1000*scale_factor
#Round DOllar to nearest 1,000
round_est = np.around(dollar_est,-3)
round_hi = np.around(dollar_hi,-3)
round_low = np.around(dollar_low,-3)
print(f'The estimate of the property is ${round_est}')
print(f'at {conf}% confidence the valuation range is' )
print(f'USD {round_low} at the lower end to USD ${round_hi} is  at the high end')


def get_dollar_estimate(rm,ptratio,chas=False,large_range=True):
    """Estimate of a Price of a property in Boston
    rm--number of rooms in the property
    ptRatio--number of students to professors
    chas -- True if property is next to water false if not
    large_range--True for 95% prediction interval False for a 68% interval
    """
    log_est,upper,lower,conf = get_log_estimate(rm,students_per_classroom=ptratio,next_to_river=chas,high_confidence=large_range)
    dollar_est = np.e**log_est*1000*scale_factor
    dollar_hi = np.e**upper*1000*scale_factor
    dollar_low = np.e**lower*1000*scale_factor
    #Round DOllar to nearest 1,000
    round_est = np.around(dollar_est,-3)
    round_hi = np.around(dollar_hi,-3)
    round_low = np.around(dollar_low,-3)
    print(f'The estimate of the property is ${round_est}')
    print(f'at {conf}% confidence the valuation range is' )
    print(f'USD {round_low} at the lower end to USD ${round_hi} is  at the high end')
