library(tidyverse)
library(tidymodels)
library(yardstick)
library(prodlim)
library(pec)
library(dials)
library(themis)
library(dplyr)

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

k_folds_original_data <- vfold_cv(train, v = 10, strata = winner)
model_metrics <- metric_set(accuracy, sens, mn_log_loss, roc_auc)

# Define recipe
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

mlp_spec <- mlp(
  hidden_units = 7, 
  penalty = 0,
  epochs = 600) %>% 
  set_engine("nnet") %>% 
  set_mode("classification") 

mlp_wf <- workflow() %>% 
  add_model(mlp_spec) %>% 
  add_recipe(base_recipe)

mlp_res <- mlp_wf %>% fit_resamples(resamples = k_folds_original_data, metrics = model_metrics)

results <- mlp_res %>% collect_metrics() %>% filter(.metric == "accuracy")
results %>% arrange(desc(mean))

final_param <- select_best(mlp_res, metric = "accuracy") 

mlp_res %>% 
  collect_predictions() %>% 
  inner_join(final_param) %>% 
  conf_mat(truth = winner, estimate = .pred_class)

final_mlp_wf <- mlp_wf %>%
  finalize_workflow(final_param)

# Fit the final model on the full training data
final_mlp_fit <- final_mlp_wf %>%
  fit(data = train)

predictions <- predict(final_mlp_fit, test)  %>% cbind(test %>% dplyr::select(id))
predictions <- predictions[, c(2, 1)] # swap the columns id and prediction 
predictions <- predictions %>% dplyr::rename(id = id, winner = .pred_class) # rename the columns to id and log_total 
write_csv(predictions, "mlp2024.csv") # write the file 

## Note: This had a slightly lower predicted accuracy score of 0.834 on our cross fold validation set, so we decided not to submit this on Kaggle officially