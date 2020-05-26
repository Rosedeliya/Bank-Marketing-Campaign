using CSV

#LOADING THE (original) CSV DATA FILE


#By default only 7 columns from the data set will be displayed.


data=CSV.read("bank-additional-full.csv")

using DataFrames

#DESCRIBING THE DATA SET

#We going to learn our data set

# shape of the dataset, 41188 rows, 21 columns

size(data)

#print all the columns 

names(data)

#showing the first 10 rows

first(data, 10)

#showing the last 10 rows

last(data, 10)

#reports the summary of descriptive statistics for any numeric column and count statistics on each categorical column
#shows min, max, quartiles, median and mean for the numeric
# Shows the count number of values for each categories

describe(data)

#checking Different Data Types in the data set


#The outcome shows that there are 5 integers, 5 floats and 11 strings

eltype.(eachcol(data))

#DATA CLEANING

#Checking for missing values


#This shows that there are no values missing in the data set

ismissing(data)

#Checking for outliers

#The following plot graghs shows outliers for some numeric attributes and float attributes


#The results shows (age and loan) are considered as outliers because they are more extreme than the observation of the outer fences

#This outliers do have affect the analysis therefore it will not be removed.

 #The boxplot detects the outliers on age over 70 years


Plots.boxplot((data[:age]), ylabel="age")

 Plots.boxplot((data[:previous]), ylabel="previous")

 Plots.boxplot((data[:duration]), ylabel="duration")

 #No outlier detected euribor3m


Plots.boxplot((data[:euribor3m]), ylabel="euribor3m")

# No outlier detected on nr.employed


Plots.boxplot((data[:20]), ylabel=20)

# No outlier detected on emp.var.rate

Plots.boxplot((data[:16]), ylabel=16)

# No outlier detected on cons.price.idx

Plots.boxplot((data[:17]), ylabel=17)

#Visualization on age

#The results shows that people between the age 30 and 40, are likely to place a term deposit


 Plots.histogram((data[:age]),bins=50,xlabel="age",labels="Frequency")

#Converting the data type

#Converting all categories to numeric

 
categories = [2 3 4 5 6 7 8 9 10 15 21]


 for col in categories 
     data[col] = fit_transform!(labelencoder, data[col]) 
 end

#This shows that we have converted all our categories to numeric

last(data, 10)

first(data, 10)

#Training and testing on the dataset

 using ScikitLearn: fit!, predict, @sk_import, fit_transform! 
 @sk_import preprocessing: LabelEncoder 
 @sk_import model_selection: cross_val_score  
 @sk_import metrics: accuracy_score 
 @sk_import linear_model: LogisticRegression 
 @sk_import ensemble: RandomForestClassifier 
 @sk_import tree: DecisionTreeClassifier 


#Training and testing on the dataset

#In order to see if the bank get its clients to place a term deposit, 10 critical columns are selected.

#Namely age, duration, previous, emp.var.rate, housing, default, loan, poutcome, job and marital

