setwd ('..')
source ('ML/utils.R')
library (tidyverse)
img_root<- 'results/CTP_gru2_reg'
label_root <- 'data/CTP/labels_186/'
DL_fea <- TRUE

if (DL_fea){out_df <- load_all_data (label_root, img_root, joining=F)
}else{out_df <- load_all_data (label_root, NULL, joining=F)}

model <- train_model (out_df$train, 'mRS_new', method='gaussprRadial')
train_pred <- predict(model, out_df$train$x)
test_pred <- predict(model, out_df$test$x)

out_df$train$y$mRS_pred <- rescale_mRS (train_pred)
out_df$test$y$mRS_pred <- rescale_mRS (test_pred)
count_mat <- rbind (out_df$train$x, out_df$test$x)

if (DL_fea){
        rm_col <- c(colnames (out_df$train$y), c('BslmRS', 'BslNIH', 'tPAdelay'))
        xtrain <- read.csv (paste (label_root, 'train.csv', sep='/'), row.names=1) %>% 
                select (!one_of (rm_col))
        xtest <- read.csv (paste (label_root, 'test.csv', sep='/'), row.names=1) %>% 
                select (!one_of (rm_col))
        nonDL <- rbind (xtrain, xtest)
        count_mat <- cbind (nonDL, count_mat)
}

# --------------------perform PCA--------------------
meta <- rbind (out_df$train$y, out_df$test$y)
dataset <- Seurat::CreateSeuratObject(counts = t(count_mat), meta.data=meta)
dataset <- TBdev::run_dim_red (dataset, run_diff_map=F, var_scale=T,
                               normalize=F, find_var_features=T, run_umap=F)
TBdev::plot_dim_red (dataset, group.by= c('mRS_3m', 'mRS_pred'), DR='pca', return_sep=T,
                    nudge_ratio=0.2, plot_type='dim_red_sim', nudge_ortho=0.7)-> DR_plots

DL_str <- ifelse (DL_fea, 'DL', 'no_DL')
cairo_pdf (paste ('report/5DR_truth_', DL_str, '.pdf', sep=''), width=7, height=7)
DR_plots [[1]] +scale_fill_viridis_c (limits=c(0,6), breaks=c(0,3,6))
dev.off ()

cairo_pdf (paste ('report/5DR_pred_', DL_str, '.pdf', sep=''), width=7, height=7)
DR_plots [[2]] +scale_fill_viridis_c (limits=c(0,6), breaks=c(0,3,6))
dev.off ()

# --------------------PC weightings--------------------
xcols <- colnames (count_mat)
all_df <- cbind (count_mat, meta)
lin_df <- lm_sum_all (all_df, xcols, 'mRS_3m')
PC_weight <- dataset[['pca']]@feature.loadings
pc_row <- gsub ('-', '_', rownames (PC_weight))
lin_df$PC1 <- PC_weight [match (lin_df$variable, pc_row),'PC_1']

ggplot (lin_df, aes(y=R2, x=PC1))+
        ggrepel::geom_text_repel (aes (label=variable)) +
        geom_point (size=4, shape=21, fill='black', color='white') + theme_ctp ()
