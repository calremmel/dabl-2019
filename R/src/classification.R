library(tidymodels)

df <- read.csv("data/raw/harini_cleaned.csv")
str(df)

df <- df %>% 
  select(-Sample) %>% 
  mutate(Cohort = factor(if_else(Cohort %in% c("PP", "NP"), "Primary", "Latent")))

data_split <- initial_split(df, strata = "Cohort", p = 0.75)

df_train <- training(data_split)
df_test <- testing(data_split)

clf <-
  logistic_reg() %>% 
  set_engine("glm") %>% 
  fit(Cohort ~ ., data = df_train)

test_results <-
  df_test %>% 
  select(Cohort) %>% 
  as_tibble() %>% 
  mutate(
    class = predict(clf, df_test) %>% 
      pull(.pred_class)
  )
  
test_results %>% conf_mat(truth = Cohort, class)

thing <- glm(Cohort ~ ., data = df_train, family = binomial)
summary(thing)
lah <- predict(thing, newdata = df_test)

library(glmnet)

df_train <- df_train %>% 
  drop_na()

x <-
  df_train %>% 
  select(-Cohort) %>% 
  drop_na() %>% 
  as.matrix()
y <- factor(df_train$Cohort)

x_test <-
  df_test %>% 
  select(-Cohort) %>% 
  as.matrix()

y_test <-
  df_test %>% 
  select(Cohort) %>% 
  factor()

model <- glmnet(x, y, family = "binomial")

test_results <-
  df_test %>% 
  select(Cohort) %>% 
  as_tibble() %>% 
  mutate(
    class = predict(model, x_test) %>% 
      pull(.pred_class)
  )
