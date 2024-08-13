library(tidyverse)
library(tidymodels)
library(yardstick)
library(prodlim)
library(pec)
library(dials)
library(themis)
library(dplyr)
library(stacks)

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
  step_impute_median(all_numeric()) %>% 
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

# create the logistic regression workflow 
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
  add_candidates(xgboost_res) %>%
  add_candidates(log_reg_res) 

data_stack <- stack %>% blend_predictions(penalty = 10^-6) 

stack_results <- data_stack %>% fit_members()

autoplot(stack_results)
autoplot(stack_results, type = "members")
autoplot(stack_results, type = "weights")

# use the stacks model on the test data to create predictions 
predictions <- predict(stack_results, test)  %>% cbind(test %>% dplyr::select(id))
predictions <- predictions[, c(2, 1)] # swap the columns id and prediction 
predictions <- predictions %>% dplyr::rename(id = id, winner = .pred_class) # rename the columns to id and winner 
write_csv(predictions, "stacks_xgboost_logreg2024.csv") # write the file 

## Note: this produced an accuracy of 0.90757 on the full test data from the https://www.kaggle.com/competitions/ucla-stats-101-c-2024-summer-classification/submissions#