# Used Car Price Prediction
Approximately 40 million used vehicles are sold each year.In the automotive sector, pricing analytics play an essential role for both companies and individuals to assess the market price of a vehicle before putting it on sale or buying it.
So,In this project we tried to solve this problem by analysing the data of used car and creating the machine learning model that would predict the fair price of your car.

## Understanding the dataset
The dataset we were working on is the data of used car taken from 'Truecar.com'. It contain details of used car from USA. The dataset contain records of over 1529216 used cars with the following features of car.
- **Price**: This is the list price of car given by the company 'Truecars'.
- **Year**: This is the year in which carwas registered or manufactured.
- **Mileage**: Mileage refer to the number of miles or kilometer the carhas been driven.
- **City**: The caris registered in which city.
- **State**: The car is registered in which State.
- **Vin**: The Vin that stands for vehicle identification number uniquely identifies the car (chasis number).
- **Make**: The company name which manufactured the car.
- **Model**: The model name of the car.

## More about the dataset and initial preprocessing
- The datset do not contain any missing values.
- The datset contain some duplicate records that were taken care.
- In total 58 different brands car are listed on the truecar site(present in dataset).
- The cars from 51 different state are listed on truecar site (present in the dataset).
- The cars from 2446 different city are listed on truecar site (prsent in the dataset).
- The cars with 22 different manufacturing year are listed in the truecar site (present in the dataset).
- VIN number is assigned to each car that is unique and different forn every car registered.
- The dataset categorical features contain data inconsistency so to overcome that each categorical feature except 'make' were converted into lowercase and spaces were removed were as 'make' was converted into uppercase and spaces were removed.

## Exploratory Data Analysis
In statistics, exploratory data analysis is an approach of analyzing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods.

### Univariate Analysis
**1. Numerical features:**

For numerical features like 'Price' and 'Mileage' we check the distribution of data using 'distplot' and check fro outliers using 'boxplot'. It was found that both price and mileage feature contain outliers but we didn't removed it as the outliers in price column were the price of branded luxirious cars like ferrari,aston etc. Year was treated as categorical feature.
||Price|Year|Mileage|
| :-- |:---------------:| -----:|-------:|
|count|1.529216e+06	|1.529216e+06|1.529216e+06|
|mean|2.132126e+04|2.013396e+03|5.203105e+04|
|std|1.325448e+04|3.374338e+00|7.519171e+04|
|min|1.500000e+03|1.997000e+00|5.000000e+00|
|25%|1.315000e+04|2.012000e+03|2.367700e+04|
|50%|1.841900e+04|2.014000e+03|3.995900e+04|
|75%|2.697800e+04|2.016000e+03|7.104400e+04|
|max|4.995000e+05|2.018000e+03|7.758776e+07|

**2. Categorical features:**

For categorical features like 'city','state','make','model'. We used countplot to check how many car falls in different labels of various categories. We created a new feature 'mileage_category' in which we convert continuous mileage feature into categorical feature creating five different labels low, medium, high, very high, extra high and use countplot for the same.
- Most of the car mentioned in the dataset is of ford company that is 209787 and least car mentioned is of GEO 'General Motors' company 4
- Most of car mentioned in the dataset are from Texas that is 172154 and least number of car are from state District of Columbia that is 31
- Most of the car mentioned in the dataset are from houston 24645 and least number car mentioned are from royston and eminence	2
- Most of the car mentioned had mileage less than 25000.

### Bivariate Analysis
To check coorelation between different numerical features we used heatmap and it was observed that 
- There is positive correlation between the year and price of car (But price of the car drops with the as the year is old).
- There is negative correlation between the mileage anf pric of car. As mileage indicates the miles travelled by car as it also indicates the wear and tear the car has under gone.

**Hypothesis**

For an example as we check the car with make 'Ford' and model 'GTCoupe' are mentioned listed above all of them full under low mileage category where as there is difference in the mielage travelled by each car listed above and all of them either registered in year 2006 or 2005 but there is price difference among them.
- We saw the least price of car is one which is registered in city west palm beach and State florida which do not impose any tax while buying the car.
- We saw three cars from New York are registered and price are different as here comes in play different features regarding car which are , exterior and interior color, additional feature of car, fuel type and drive type.

'Price' featured was grouped according to 'make' and average price was calculated for each make brand of car similarly for year feature was conducted. From conducting EDA we comclude following things,
For car price prediction following feature should be considered
1. Make and Model of the car.
2. Year
3. Mileage of the car.
4. Location that is city and state.
Apart from this other additional feature of the car should also be included while predicting the car price.

## Feature Enigneering
1. Dealing with outliers
2. Dealing with Missing Values
3. Creating features
4. Mutual Information
5. Encoding
6. Feature Scaling
 
- The dataset does not contain any missing values. 
- the outliers are kept as it is because we will be using tree based model and XGBoost which have it's own meachanism to deal with outliers

**Creating new features:**

We created several new feature that would help us in creating machine learning model that are,
1. **Age**: The age feature was created by subtracting year feature from current year.
2. **Is_Expensive**: This feature contain '1' if the car make is among the following ["mcLaren", "rolls-royce", "ferrari", "lamborghini", "bentley", "aston", "maybach", "tesla"] for other car brand it will contain '0'.
3. **Mileage*Year**: This feature is created by multiplying year and mileage.
4. **Condition**: This feature contain '1' that means bad condition when mileage is greater than 100000 or age is greater than 10 else '0'.
5. **mean_wrt_model**: Grouping according to model and calculating mean price for each model.
6. **mean_wrt_make**: Grouping according to make and calculating mean price for each make.
7. **mean_wrt_city**: Grouping according to city and calculating mean price for each city.

**Encoding:**

For encoding categorical features we calculated first top 14 most appeared labels in datset and categorise remaining labels in other category and then applied 'One-Hot Encoding' on the same. This was done for two features city and make.

There is good correlation between target varibale and newly created fetures that is more than 0.2. We use heatmap for the same. The mutual information technique was used to check non-linear
relation between dependent and independent variables. The scaling of the data was done using 'StandardScaler'.

## Model Building
We build various model using 'Lasso Regression' and 'XGBoost REgressor' after checking the accuracy and RMSE we like to go ahead with XGBoost Regressor. We conducted hyperparameter
tunning so get best parameter for our model.
Gradient boosting is a method that goes through cycles to iteratively add models into an ensemble.It begins by initializing the ensemble with a single model, whose predictions can be pretty naive. (Even if its predictions are wildly inaccurate, subsequent additions to the ensemble will address those errors.)
Then, we start the cycle:
- First, we use the current ensemble to generate predictions for each observation in the dataset. To make a prediction, we add the predictions from all models in the ensemble.
- These predictions are used to calculate a loss function (like mean squared error, for instance).
- Then, we use the loss function to fit a new model that will be added to the ensemble. Specifically, we determine model parameters so that adding this new model to the ensemble will reduce the loss. (Side note: The "gradient" in "gradient boosting" refers to the fact that we'll use gradient descent on the loss function to determine the parameters in this new model.)
- Finally, we add the new model to ensemble, and ...
- ... repeat!

## Deployment
The deployment was done using Flask and Amazon EC2. We created web application using Flask u=in the backend and forntend was build using HTML and CSS. The deployment was done on Amazon EC2.For the deployment phase, we used the EC2 instance provided by Amazon. First, we had to make an account on Amazon Web Services. After which we created an instance for our project. It was a 7steo procedure for which we choose specific option that best suited our project. For example the OS we went for Ubuntu. We also used putty and winscp which helped us set up a virtual machine. After successful setting our task on Amazon EC2, we were able to generate a link that would give access to everyone/anyone to be able to use the Used Car Prices prediction model we created
