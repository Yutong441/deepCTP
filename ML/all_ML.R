setwd ('..')
library (tidyverse)
source ('ML/utils.R')

img_root<- 'results/CTP_gru2_reg/'
label_root <- 'data/CTP/labels_186/'
save_dir <- 'results/ML/clas'

# --------------------test all models--------------------
load_train_test (label_root, paste (save_dir, 'img_nimg.csv', sep='_'), img_root)
load_train_test (label_root, paste (save_dir, 'nimg_only.csv', sep='_'), NULL)
load_train_test (label_root, paste (save_dir, 'img_only.csv', sep='_'), img_root, joining=F)
plot_results ('results/ML/CTP', 'mRS_new')

# --------------------test individual models--------------------
out_df <- load_all_data (label_root, img_root, joining=T)
model <- train_model (out_df$train, 'mRS_new', method='rf')
acc <- test_model (model, out_df$train, out_df$test, 'mRS_new')
imp_fea <- caret::varImp (model)
imp_fea$importance %>% filter (Overall >= 5) %>% rownames () -> new_fea
imp_fea$importance %>% arrange (desc (Overall)) %>% head (30)

new_train <- list (x=out_df$train$x [,new_fea], y=out_df$train$y)
new_test <- list (x=out_df$test$x [,new_fea], y=out_df$test$y) 
new_model <- train_model (new_train, 'mRS_new', method='gaussprRadial')
test_model (new_model, new_train, new_test, 'mRS_new')

out_df <- load_all_data (label_root, img_root, joining=T)
model <- train_model (out_df$train, 'mRS_new', method='gaussprRadial')
acc <- test_model (model, out_df$train, out_df$test, 'mRS_new')
expla_mod <- DALEX::explain (model, data=out_df$train$x, y=out_df$train$y$mRS_new)
varimps <- DALEX::variable_importance (expla_mod)
plot (varimps)

# --------------------Mult-modality--------------------
img_root<- c('results/CTP_gru2_reg', 'results/NCCT_gru2', 'results/CTA_gru2')
save_dir <- 'results/ML/CTP_NCCT'
load_train_test (label_root, paste (save_dir, 'img_only.csv', sep='_'), img_root[1:2], joining=F)

save_dir <- 'results/ML/CTP_CTA'
load_train_test (label_root, paste (save_dir, 'img_only.csv', sep='_'), img_root[c(1,3)], joining=F)

save_dir <- 'results/ML/CTA_NCCT'
load_train_test (label_root, paste (save_dir, 'img_only.csv', sep='_'), img_root[2:3], joining=F)

save_dirs <- c( 'results/ML/CTP_NCCT',  'results/ML/CTP_CTA',  'results/ML/CTA_NCCT')
multimod <- load_multimod (save_dirs)
method_order <- c('knn', 'svmRadial', 'svmLinear', 'lasso', 'rf', 'gaussprRadial')
multimod %>% filter (mode=='test' & ycol=='mRS_new' & method != 'nnet') %>%
        mutate (method = factor (method, levels=method_order)) %>%
        ggplot (aes (x=method, y=AUC, fill=modality)) +
        geom_text (aes (label=AUC), vjust=0., position=position_dodge(width=.9)) +
        geom_bar (stat='identity', position='dodge2')+ 
        coord_cartesian(ylim=c(0.5, 0.9)) +theme_ctp ()
