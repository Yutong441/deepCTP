setwd ('..')
source ('ML/utils.R')
library (tidyverse)
label_root <- 'data/CTP/labels_186/'

# --------------------linear regression--------------------
out_df <- load_all_data (label_root)
x_df <- rbind (out_df$train$x, out_df$test$x)
y_df <- rbind (out_df$train$y, out_df$test$y)
out_df <- cbind (x_df, y_df)
xcols <- colnames (x_df)

lin_df <- lm_sum_all (out_df, xcols, 'mRS_3m')
plot_lm_val (lin_df)
write.csv (lin_df, 'report/1lm_non_imaging.csv')

# --------------------THRIVE--------------------
test_df <- read.csv ('data/CTP/labels/test.csv') %>% filter (!is.na (X24hNIH))
NIHSS_score <- function (x){
        return (0*(x<=10) + 2*(x>10 & x<20) + 4*(x>=20)) }

Age_score <- function (x){
        return (0*(x<=59) + 1*(x > 60 & x <= 79) + 2*(x>=80)) }

x <- test_df$BslNIH
THRIVE <- NIHSS_score (x) + Age_score (test_df$Age)+ as.numeric (test_df$HTN)+
        as.numeric (test_df$Diabetes) + as.numeric (test_df$AF)
caret::confusionMatrix (as.factor (THRIVE>5), as.factor (test_df$mRS_3m >2))

pROC::auc(as.numeric (THRIVE>5), as.numeric (test_df$mRS_3m >2))
# Area under the curve: 0.5814

# --------------------Bivard 2017--------------------
test_df <- read.csv ('data/CTP/labels/test.csv')
ypred <- test_df$Core <=25 & test_df$Penumbra <=20 & test_df$TotalDT6 <=30
caret::confusionMatrix (as.factor (!ypred), as.factor (test_df$mRS_3m >2))
pROC::auc (as.numeric(!ypred), as.numeric(test_df$mRS_3m >2))
# Area under the curve: 0.55

# --------------------demographic features--------------------
train_df <- get_img_data (paste (label_root, 'train.rds', sep=''))
test_df <- get_img_data (paste (label_root, 'test.rds', sep=''))
xcols <- c("Sex", "Age", "HTN", "Diabetes", "HC", "Smoke", "AF", "HF",
           "MI.Angina", "Side", "BslmRS", "BslNIH")
train_df <- list (x=train_df$x[,xcols], y=train_df$y)
test_df <- list (x=test_df$x[,xcols], y=test_df$y)
all_models (train_df, test_df, save_dir='results/ML/demo.csv')

read.csv ('results/ML/demo.csv')%>% 
        filter (mode=='test' & ycol == 'mRS_3m') %>%
        select (all_of (c('method', 'AUC', 'accuracy', 'sensitivity', 'specificity'))) %>%
        write.csv('report/2demo_acc.csv')

model <- train_model (train_df, 'mRS_3m', method='rf')
test_model (model, train_df, test_df, 'mRS_3m')
plot_var_imp (model)

# --------------------ML results--------------------
read.csv ('results/CTP_gru2_reg/clas_nimg_only.csv')%>% 
        filter (mode=='test' & ycol == 'mRS_3m') %>%
        select (all_of (c('method', 'AUC', 'accuracy', 'sensitivity', 'specificity'))) %>%
        write.csv('report/2ML_acc.csv')

# lasso
out_df <- load_all_data (label_root)
model <- train_model (out_df$train, 'mRS_3m', method='lasso')
test_model (model, out_df$train, out_df$test, 'mRS_3m')
plot_var_imp (model)
