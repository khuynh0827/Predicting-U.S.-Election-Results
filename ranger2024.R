library(tidyverse)
library(tidymodels)
library(yardstick)
library(dials)
library(dplyr)
library(ranger)

set.seed(2024)

train <- read_csv("train_class.csv")
train_removed <- train %>% dplyr::select(!x0033e) %>% dplyr::select(!name) %>% dplyr::select(!id)
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
  step_impute_median(all_numeric()) 

# random forest model 
randomForest_spec <- rand_forest(min_n = tune(), 
                                 trees = tune(), 
                                 mtry = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("ranger")

# random forest workflow 
random_forest_wf <- workflow() %>% 
  add_model(randomForest_spec) %>% 
  add_recipe(base_recipe)

recipes_param <- extract_parameter_set_dials(randomForest_spec)

# parameters for boosted forest grid 
recipes_param_rf <- extract_parameter_set_dials(randomForest_spec) %>% 
  update("mtry" = mtry(c(1, 20))) 

# parameters for random forest grid 
rf_grid <- grid_latin_hypercube(recipes_param_rf, 
                                size = 20)

# decided to get rid of one tree because it seemed that that would be over fitting 

# use tune_grid() to try out different values for the tuning function 
rf_res <- tune_grid(
  random_forest_wf, 
  resamples = k_folds_original_data, 
  grid = rf_grid, 
  metrics = model_metrics, 
  control = model_control
)

results <- rf_res %>% collect_metrics() %>% filter(.metric == "accuracy")
results %>% arrange(desc(mean)) # highest said = 0.909

final_param <- select_best(rf_res, metric = "accuracy") 

rf_res %>% 
  collect_predictions() %>% 
  inner_join(final_param) %>% 
  conf_mat(truth = winner, estimate = .pred_class)

final_rf_wf <- random_forest_wf %>%
  finalize_workflow(final_param)

# Fit the final model on the full training data
final_rf_fit <- final_rf_wf %>%
  fit(data = train)

predictions <- predict(final_rf_fit, test)  %>% cbind(test %>% dplyr::select(id))
predictions <- predictions[, c(2, 1)] # swap the columns id and prediction 
predictions <- predictions %>% dplyr::rename(id = id, winner = .pred_class) # rename the columns to id and winner 
write_csv(predictions, "ranger2024.csv") # write the file 

## Note: This had an accuracy score of 0.90203 on the full test data from https://www.kaggle.com/competitions/ucla-stats-101-c-2024-summer-classification/submissions#