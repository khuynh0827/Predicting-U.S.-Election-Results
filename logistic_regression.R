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
train_removed <- train %>% dplyr::select(!x0033e) %>% dplyr::select(!name) %>% dplyr::select(!id)
test <- read_csv("test_class.csv")

train_removed_downsample <- recipe(winner ~., data = train_removed) %>% 
  step_downsample(winner) %>% 
  prep() %>% 
  juice()

k_folds_original_data <- vfold_cv(train, v = 10, strata = winner)
model_control <- control_grid(save_pred = TRUE, save_workflow = TRUE)
model_metrics <- metric_set(accuracy, sens, mn_log_loss, roc_auc)

base_recipe <- recipe(winner ~., data = train_removed_downsample) %>%
  step_mutate(x2013_code = factor(x2013_code)) %>% 
  step_dummy(x2013_code) %>% 
  step_impute_median(all_numeric()) %>% 
  step_mutate(prop_white = x0064e / x0001e, 
              prop_minor = (x0064e + x0065e + x0066e + x0067e + x0068e + x0069e + x0071e)/ x0001e, 
              prop_women = x0003e / x0001e,
              prop_men = x0002e / x0001e,
              prop_minor = x0019e / x0001e,
              prop_college = (c01_005e + c01_013e + c01_018e + c01_021e + c01_024e + c01_027e) / x0001e,
              prop_old = x0024e / x0001e) %>% 
  step_rm(x0064e, x0065e, x0066e, x0067e, x0068e, x0069e, x0071e, 
          x0003e, x0002e, x0019e, x0020e, x0021e, x0022e, x0023e, x0024e, x0005e, x0006e, x0007e)

regularized_spec <- logistic_reg(penalty = tune(), mixture = tune()) %>% 
  set_engine("glmnet") %>% 
  set_mode("classification")

log_reg_wf <- workflow() %>% 
  add_model(regularized_spec) %>% 
  add_recipe(base_recipe)

# parameters for the tuning grid 
log_reg_grid <- grid_latin_hypercube(parameters(regularized_spec), 
                                     size = 20)

# use the tune_grid function to test out different values for the tuning grid 
log_reg_res <- tune_grid(
  log_reg_wf, 
  resamples = k_folds_original_data, 
  grid = log_reg_grid,
  metrics = model_metrics, 
  control = model_control
)

results <- log_reg_res %>% collect_metrics() %>% filter(.metric == "accuracy")
results %>% arrange(desc(mean)) # highest said = 0.916

final_param <- select_best(log_reg_res, metric = "accuracy") 

log_reg_res %>% 
  collect_predictions() %>% 
  inner_join(final_param) %>% 
  conf_mat(truth = winner, estimate = .pred_class)

final_log_reg_wf <- log_reg_wf %>%
  finalize_workflow(final_param)

# Fit the final model on the full training data
final_log_reg_fit <- final_log_reg_wf %>%
  fit(data = train)

predictions <- predict(final_log_reg_fit, test)  %>% cbind(test %>% dplyr::select(id))
predictions <- predictions[, c(2, 1)] # swap the columns id and prediction 
predictions <- predictions %>% dplyr::rename(id = id, winner = .pred_class) # rename the columns to id and winner 
write_csv(predictions, "log_reg_round2.csv") # write the file 
