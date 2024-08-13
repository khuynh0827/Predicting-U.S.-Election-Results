library(tidyverse)
library(tidymodels)
library(yardstick)
library(prodlim)
library(pec)
library(dials)
library(themis)
library(dplyr)
library(stacks)
library(bonsai)

set.seed(2024)

train <- read_csv("train_class.csv")
train_removed <- train %>% 
  dplyr::select(!x0033e) %>% 
  dplyr::select(!name) %>% 
  dplyr::select(!id) %>% 
  dplyr::select(!x0036e) %>% 
  dplyr::select(!x0058e) 
test <- read_csv("test_class.csv")

train_removed_downsample <- recipe(winner ~., data = train_removed) %>% 
  step_downsample(winner) %>% 
  prep() %>% 
  juice()

k_folds_data <- vfold_cv(train_removed_downsample, v = 10, strata = winner)
k_folds_original_data <- vfold_cv(train, v = 10, strata = winner)
model_control <- control_grid(save_pred = TRUE, save_workflow = TRUE)
model_metrics <- metric_set(accuracy, sens, mn_log_loss, roc_auc)

base_recipe <- recipe(winner ~., data = train_removed_downsample) %>%
  step_mutate(x2013_code = factor(x2013_code)) %>% 
  step_dummy(x2013_code) %>% 
  step_impute_median(all_numeric()) %>% 
  step_mutate(prop_white = x0064e / x0001e, 
              # black + american Indian + asian + native Hawaiian + some other race + hispanic 
              prop_minor = (x0064e + x0065e + x0066e + x0067e + x0068e + x0069e + x0071e)/ x0001e, 
              prop_women = x0003e / x0001e,
              prop_men = x0002e / x0001e,
              prop_minor = x0019e / x0001e,
              prop_college = (c01_005e + c01_013e + c01_018e + c01_021e + c01_024e + c01_027e) / x0001e,
              prop_old = x0024e / x0001e) %>% 
  step_rm(x0064e) %>% 
  step_rm(x0065e) %>% 
  step_rm(x0066e) %>% 
  step_rm(x0067e) %>% 
  step_rm(x0068e) %>% 
  step_rm(x0069e) %>% 
  step_rm(x0071e) %>% 
  step_rm(x0003e) %>% 
  step_rm(x0002e) %>%
  step_rm(x0019e) %>% 
  step_rm(x0020e) %>% 
  step_rm(x0021e) %>% 
  step_rm(x0022e) %>% 
  step_rm(x0023e) %>% 
  step_rm(x0024e) %>% 
  step_rm(x0005e) %>% 
  step_rm(x0006e) %>% 
  step_rm(x0007e)

# logistic reg model 
logistic_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

# create the decision tree workflow 
log_reg_wf <- workflow() %>% 
  add_model(logistic_spec) %>% 
  add_recipe(base_recipe)

# parameters for the tuning grid 
log_reg_grid <- grid_latin_hypercube(parameters(logistic_spec), 
                                     size = 20)

# use the tune_grid function to test out different values for the tuning grid 
log_reg_res <- tune_grid(
  log_reg_wf, 
  resamples = k_folds_original_data, 
  grid = log_reg_grid,
  metrics = model_metrics, 
  control = model_control
)

# random forest model 
randomForest_spec <- rand_forest(min_n = tune(), 
                                 trees = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("randomForest")

# random forest workflow 
random_forest_wf <- workflow() %>% 
  add_model(randomForest_spec) %>% 
  add_recipe(base_recipe)

# parameters for random forest grid 
rf_grid <- grid_latin_hypercube(parameters(randomForest_spec), 
                                size = 20)

# use tune_grid() to try out different values for the tuning function 
rf_res <- tune_grid(
  random_forest_wf, 
  resamples = k_folds_original_data, 
  grid = rf_grid, 
  metrics = model_metrics, 
  control = model_control
)

xgboost_spec <- boost_tree(learn_rate = tune(), 
                           trees = tune(), 
                           tree_depth = tune(), 
                           mtry = tune(), 
                           min_n = tune(), 
                           loss_reduction = tune(), 
                           sample_size = tune(), 
                           stop_iter = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("xgboost")

# boosted tree workflow
xgboost_wf <- workflow() %>% 
  add_model(xgboost_spec) %>% 
  add_recipe(base_recipe)

recipes_param <- extract_parameter_set_dials(xgboost_spec)

# parameters for boosted forest grid 
recipes_param_xgboost <- extract_parameter_set_dials(xgboost_spec) %>% 
  update("mtry" = mtry(c(0, 500))) 

# define grid for tuning for the xgboost model 
xgboost_grid <- grid_latin_hypercube(recipes_param_xgboost, 
                                     size = 20)

# use tune_grid() to auto tune the xgboost model 
xgboost_res <- tune_grid(
  xgboost_wf, 
  resamples = k_folds_original_data, 
  grid = xgboost_grid, 
  metrics = model_metrics, 
  control = model_control
)

# create a stacks model of the random forest, xgboost model, and decision tree model 
stack <- stacks()

# Add candidates to the stack
stack <- stack %>%
  add_candidates(rf_res) %>%
  add_candidates(xgboost_res) %>%
  add_candidates(log_reg_res)

# add the specified penalty to increase accuracy 
data_stack <- stack %>% blend_predictions(penalty = 10^-2) %>% fit_members()

# get parameters for each model 
select_best(rf_res, metric = "accuracy") 
select_best(xgboost_res, metric = "accuracy") 
select_best(log_reg_res, metric = "accuracy") 

# plot that helps visualize metrics 
autoplot(data_stack)
autoplot(data_stack, type = "members")
autoplot(data_stack, type = "weights")

# use the stacks model on the test data to create predictions 
predictions <- predict(data_stack, test)  %>% cbind(test %>% dplyr::select(id))
predictions <- predictions[, c(2, 1)] # swap the columns id and prediction 
predictions <- predictions %>% dplyr::rename(id = id, winner = .pred_class) # rename the columns to id and winner 
write_csv(predictions, "stacks_base_recipe_long2024.csv") # write the file 

## Note: This had an accuracy score of 0.92791 on the test data from https://www.kaggle.com/competitions/ucla-stats-101-c-2024-summer-classification/submissions#