setwd('C:/Users/olivi/OneDrive/Documents/School2023/RestaurantRevenue')

# libraries
library(vroom)
library(tidyverse)
library(tidymodels)
library(embed)
library(discrim)

# data
train <- vroom("./train.csv")
test <- vroom("./test.csv")

# alter column names to make them easier to use
make_names <- function(x) {
  gsub("\\.", "_", make.names(names(x),unique = TRUE))
}
names(test) <- make_names(test)
names(train) <- make_names(train)



# columns
# Id : Restaurant id. 
# Open Date : opening date for a restaurant
# City : City that the restaurant is in. Note that there are unicode in the names. 
# City Group: Type of the city. Big cities, or Other. 
# Type: Type of the restaurant. FC: Food Court, IL: Inline, DT: Drive Thru, MB: 
#  Mobile
# P1, P2 - P37: There are three categories of these obfuscated data. Demographic
#  data are gathered from third party providers with GIS systems. These include 
#  population in any given area, age and gender distribution, development 
#  scales. Real estate data mainly relate to the m2 of the location, front 
#  facade of the location, car park availability. Commercial data mainly include
#  the existence of points of interest including schools, banks, other 
#  QSR operators.
# Revenue: The revenue column indicates a (transformed) revenue of the 
#  restaurant in a given year and is the target of predictive analysis. Please 
#  note that the values are transformed so they don't mean real dollar values. 


#############################
### EDA
#############################

## looking to see what kind of information we can get from slide in class
dplyr::glimpse(train) #lists the variable type of each column
skimr::skim(train) # nice overview of the train data
DataExplorer::plot_intro(test) # visualization of glimpse()
DataExplorer::plot_correlation(test)
DataExplorer::plot_bar(train) # bar charts of all discrete variables
DataExplorer::plot_histogram(train) # histograms of all numerical variables
DataExplorer::plot_missing(test) # percent missing in each column

#############################
### Setup
#############################

Years <- unique(format(as.Date(test$Open_Date, format="%m/%d/%Y"),"%Y"))

my_recipe <- recipe(revenue ~ ., data=train) %>%
  step_mutate(Open_Date = as.Date(Open_Date, '%m/%d/%Y')) %>%
  step_date(Open_Date, features=c('dow', 'month', 'year', 'quarter')) %>%
  step_mutate(Open_Date_year = factor(Open_Date_year, levels = Years),
              Open_Date_quarter = factor(Open_Date_quarter),
              Type=factor(Type, levels=c("FC", "IL", "DT", "MB"))) %>%
  step_novel(all_nominal_predictors()) %>%
  step_rm(City, Open_Date, Id)

# make sure prep/bake works

prepped_recipe <- prep(my_recipe) 
new_data <- bake(prepped_recipe, new_data = train)
DataExplorer::plot_missing(test)
#sapply(new_data, class)

# folds for cv
folds <- vfold_cv(train, v = 5, repeats=1)

#############################
### Models to Try
#############################




### Random Forest ###

RF_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% #Type of model (500 or 1000) # more is better
  set_engine("ranger") %>% # What R function to use
  set_mode("regression")

## Create a workflow with model & recipe
# use my_recipe_tree from tree model
RF_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(RF_mod)

## Set up grid of tuning values
RF_tuning_grid <- grid_regular(mtry(range = c(1, 9)),
                               min_n())


## Run the CV
CV_results_RF <- RF_wf %>%
  tune_grid(resamples=folds,
            grid=RF_tuning_grid,
            metrics=metric_set(rmse)) #Or leave metrics NULL

## Find best tuning parameters
bestTune <- CV_results_RF %>%
select_best("rmse")

## Finalize workflow and predict
final_wf <-
  RF_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

## Predict
test_preds <- final_wf %>%
  predict(new_data = test) %>%
  bind_cols(., test) %>% #Bind predictions with test data
  select(Id, .pred) %>% #Just keep datetime and predictions
  rename(Prediction=.pred) #rename pred to Prediction (for submission to Kaggle)

## Write prediction file to CSV
vroom_write(x=test_preds, file="./RFSubmission21.csv", delim=",")

#############################
### Future work with imputation
#############################

# train$P3[train$P3 == 0] <- NA
# train$P29[train$P29 == 0] <- NA

#test$P3[test$P3 == 0] <- NA
#test$P29[test$P29 == 0] <- NA

# num of missing values
# sum(test$P3==0) #318
# sum(test$P14==0) #65734
# sum(test$P15==0) #65772
# sum(test$P16==0) #66094
# sum(test$P17==0) #65792
# sum(test$P18==0) #65980
# sum(test$P24==0) #65766
# sum(test$P25==0) #65738
# sum(test$P26==0) #65784
# sum(test$P27==0) #66193
# sum(test$P29==0) #3083
# sum(test$P30==0) #65596
# sum(test$P31==0) #65566
# sum(test$P32==0) #65787
# sum(test$P33==0) #65791
# sum(test$P34==0) #65832
# sum(test$P35==0) #65776
# sum(test$P36==0) #65662
# sum(test$P37==0) #66029

# mice walk through
#https://www.r-bloggers.com/2016/06/handling-missing-data-with-mice-package-a-simple-approach/

# library(dplyr) 
# imp_test <- train %>%
#     mutate(Smoking = as.factor(Smoking)) %>% 
#     mutate(Education = as.factor(Education)) %>% 
#     mutate(Cholesterol = as.numeric(Cholesterol))
# 
# 
# library(mice)
# init = mice(dat, maxit=0) 
# meth = init$method
# predM = init$predictorMatrix


# my_recipe <- recipe(revenue ~ ., data=train) %>%
#   step_mutate(Open_Date = as.Date(Open_Date, '%m/%d/%Y')) %>%
#   step_date(Open_Date, features=c('dow', 'month', 'year', 'quarter')) %>%
#   step_mutate(Open_Date_year = factor(Open_Date_year),
#               Open_Date_quarter = factor(Open_Date_quarter),
#               Type=factor(Type)) %>%
#   step_novel(Open_Date_year, Type) %>%
#   step_rm(City, Open_Date, Id) %>%
#   step_rm(P14,P15,P16,P17,P18,P24,P25,P26,P27,P30,P31,P32,P33,P34,P35,P36,P37) %>%
#   step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   step_impute_knn(all_predictors(), neighbors = 3)
# 
# 
# #%>%
#   #step_mutate_at(all_numeric_predictors(), fn = factor) %>%
#   #step_mutate(P29= as.numeric(P29),
#   #            P3 = as.numeric(P3))
#   step_impute_linear(P29, impute_with = imp_vars("City_Group","Type","P1","P2"
#                                                  ,"P4","P5","P6","P7","P8","P9"
#                                                  ,"P10","P11","P12","P13","P19"
#                                                  ,"P20","P21","P22","P23","P28"
#                                                  ,"Open_Date_dow","Open_Date_month"
#                                                  ,"Open_Date_year","Open_Date_quarter")) %>%
#   step_impute_linear(P3, impute_with = imp_vars("City_Group","Type","P1","P2"
#                                                 ,"P4","P5","P6","P7","P8","P9"
#                                                 ,"P10","P11","P12","P13","P19"
#                                                 ,"P20","P21","P22","P23","P28"
#                                                 ,"P29","Open_Date_dow"
#                                                 ,"Open_Date_month","Open_Date_year"
#                                                 ,"Open_Date_quarter"))
