preprocess_img <- function (img_path, modality='train'){
        img_list <- list()
        for (i in 1:length(img_path)){
                prefix <- gsub ('/$', '', img_path[i])
                prefix <- gsub ('^.*/', '', prefix)
                prefix <- gsub ('_.*$', '', prefix)
                img_dat_path <- paste (img_path[i], '/CTP_act_',
                                       modality, '.csv', sep='')
                img_fea <- read.csv (img_dat_path, row.names=1) 
                colnames (img_fea) <- paste (prefix, 1:ncol(img_fea), sep='')
                img_list [[i]] <- img_fea
        }
        if (length (img_list)==1){return (img_list[[1]])
        }else{do.call ('cbind', img_list) %>% return ()
        }
}

#' Obtain dataframe for machine learning
#'
#' @param label_path path to the rds file which is a list of 2 elements: x
#' variables and y variables
#' @param img_path path to the csv file containing features extracted by deep
#' neural network
#' @examples
#' train_df <- get_img_data ('data/CTP/labels/train.rds', 
#'                  'results/CTP_re/CTP_act_train.csv')
get_img_data <- function (label_path, img_path=NULL, joining=TRUE, modality='train'){
        label_df <- readRDS (paste (label_path, '/', modality, '.rds', sep=''))
        if (!is.null (img_path)){
                img_fea <- preprocess_img (img_path, modality)
                if (joining){ label_df$x <- cbind (label_df$x, img_fea)
                }else{label_df$x <- img_fea}
        }
        for (i in colnames (label_df$y) ){
                if ( length (unique (label_df$y[,i])) == 2 ){
                        label_df$y [,i] <- as.factor (label_df$y[,i])
                }
        }
        if ('mRS_3m' %in% colnames (label_df$y)){
                label_df$y$mRS_clas <- as.factor (label_df$y$mRS_3m)
        }
        return (label_df)
}

train_model <- function (train_df, y_col, method='rf'){
        set.seed(42)
        seeds <- vector(mode = "list", length = 26)
        for(i in 1:25) {seeds[[i]] <- sample.int(1000, 50)}
        seeds[[26]] <- sample.int(1000,1)
        caret::trainControl(method="repeatedcv", number = 5, repeats =5, seeds=seeds,
                            preProcOptions=list(cutoff=0.75)) -> train_control 
        set.seed (42)
        caret::train(train_df$x, train_df$y[,y_col], method=method, 
                     trControl=train_control) -> model
        return (model)
}

nmse <- function (ytrue, ypred){
        mean((ytrue - ypred)^2) / mean(ytrue^2) %>% sqrt () %>% return ()
}

test_stat <- function (ytrue, ypred, pos_label){
        if (is.na (pos_label)){pos_label <- median (ytrue)}
        acc_df <- data.frame (NMSE= nmse (as.numeric (ytrue), as.numeric (ypred) ) )
        if (length (unique (ypred) ) >2 ){ 
                ypred <- ifelse (as.numeric (ypred) > pos_label, 1, 0)
                ytrue <- ifelse (as.numeric (ytrue) > pos_label, 1, 0)
        }
        if (length (unique (ypred)) >1  & length (unique (ytrue) ) >1){
                acc_df$AUC <- pROC::auc (as.numeric (ytrue), as.numeric (ypred))
                acc <- caret::confusionMatrix (as.factor (ypred), as.factor (ytrue)) 
                acc_df$accuracy <- acc$overall ['Accuracy']
                acc_df$sensitivity <- acc$byClass ['Sensitivity']
                acc_df$specificity <- acc$byClass ['Specificity']
        }else{
                acc_df$AUC <- 0.5
                acc_df$accuracy <- 0.5
                acc_df$sensitivity <- 0.5
                acc_df$specificity <- 0.5
        }
        return (round (acc_df,3))
}

test_model <- function (model, train_df, test_df, ycol, pos_label=2.){
        set.seed (42)
        train_pred <- predict(model, train_df$x)
        acc_train <- test_stat (train_df$y[,ycol], train_pred, pos_label)
        set.seed (42)
        test_pred <- predict(model, test_df$x)
        acc_test <- test_stat (test_df$y[,ycol], test_pred, pos_label)
        acc_df <- rbind (acc_train, acc_test)
        col_order <- c('method', 'mode', 'AUC', 'accuracy', 'sensitivity',
                       'specificity', 'NMSE', 'ycol', 'sample')
        acc_df$method <- model$method
        acc_df$ycol <- ycol
        acc_df$mode <- c('train', 'test')
        acc_df$sample <- c(dim(train_df$x)[1], dim (test_df$x)[1])
        return (acc_df [, col_order])
}

remove_na <- function (df_list, ycol){
        remove_rows <- is.na (df_list$y [,ycol])
        return (list (x=df_list$x[!remove_rows,], y=df_list$y[!remove_rows,]))
}

train_test <- function (train_df, test_df, ycol, method, pos_label) {
        classification <- class (train_df$y[,ycol]) == 'factor'
        proceed <- TRUE
        if (classification & method %in% c('lasso', 'blasso')){proceed <- FALSE}
        if (!classification & method %in% c('vglmAdjCat', 'plr')){proceed <- FALSE}
        train_df <- remove_na (train_df, ycol)
        test_df <- remove_na (test_df, ycol)
        if (proceed){
                model <- train_model (train_df, ycol, method=method)
                return (test_model (model, train_df, test_df, ycol, pos_label))
        }
}

all_models <- function (train_df, test_df, ycols=NULL, model_names=NULL, pos_label=2.,
                        save_dir=NULL){
        if (is.null (model_names)){
                model_names <- c('knn', 'svmRadial', 'svmLinear', 'lasso',
                                  'rf', 'nnet', 'plr', 'gaussprRadial')
        }
        if (is.null (ycols)){
                ycols <- c('mRS_3m', 'mRS_new', 'mRS_clas', 'mRS_bool', 'ICH',
                           'X24hNIH')
        }
        acc_list <- list ()
        if (length (pos_label)==1){pos_label <- rep (pos_label, length (ycols))}
        for (i in 1:length (model_names)){
                all_acc <- lapply (as.list (1:length (ycols)), function (j){
                        train_test (train_df, test_df, ycols[j], model_names[i], pos_label[j])
                })
                acc_list[[i]] <- do.call('rbind', all_acc)
                print (acc_list[[i]])
        }
        final_df <- do.call('rbind', acc_list)
        if (is.null (save_dir)){return (final_df)
        }else{write.csv (final_df, save_dir, quote=F, row.names=F)}
}

load_all_data <- function (label_root, img_root=NULL, joining=T){
        if (!is.null (img_root) ){
                train_df <- get_img_data (label_root, img_root,
                                          joining=joining, modality='train')
                test_df <- get_img_data (label_root, img_root, joining=joining,
                                         modality='test')
        }else{
                train_df <- get_img_data (label_root, modality='train')
                test_df <- get_img_data (label_root, modality='test')
        }
        caret::preProcess(train_df$x, method=c("center", "scale", "corr", "nzv"),
                          cutoff=0.6) -> transformations
        train_df$x <- predict(transformations, train_df$x)
        test_df$x <- predict(transformations, test_df$x)
        return (list ('train'=train_df, 'test'=test_df))
}

load_train_test <- function (label_root, save_dir, img_root=NULL, joining=T,
                             model_names=NULL, ycols=c('mRS_3m', 'mRS_new',
                             'mRS_clas', 'mRS_bool', 'ICH', 'X24hNIH'), pos_label= 
                             c(2., 2., 0.5, 0.5, 6) ){
        out_df <- load_all_data (label_root, img_root, joining)
        all_models (out_df$train, out_df$test, ycols, model_names=model_names,
                    save_dir=save_dir)
}
#if (method %in% c('rf', 'vglmAdjCat')){ print (caret::varImp (model)) }

plot_results <- function (save_dir, y_col='mRS_3m'){
        img_nimg <- read.csv (paste (save_dir, 'img_nimg.csv', sep='_'))
        img_nimg$inp <- 'combined'
        nimg_only <- read.csv ('results/ML/nimg_only.csv')
        nimg_only$inp <- 'non_DL'
        img_only <- read.csv (paste (save_dir, 'img_only.csv', sep='_'))
        img_only$inp <- 'DL_only'
        plot_df <- do.call ('rbind', list(img_nimg, nimg_only, img_only))
        method_order <- c('knn', 'svmRadial', 'svmLinear', 'lasso', 'rf', 'gaussprRadial')
        plot_df %>% filter (ycol == y_col & mode == 'test' & method != 'nnet') %>%
                mutate (inp = factor (inp, levels = c('non_DL', 'DL_only', 'combined') )) %>%
                mutate (method= factor (method, levels=method_order)) -> plot_df
        ggplot (plot_df, aes (x=method, y=AUC, fill=inp)) + 
        geom_bar ( stat='identity', position='dodge2')+ 
        geom_text (aes (label=AUC), vjust=0., position=position_dodge(width=.9)) +
        coord_cartesian(ylim=c(0.5, 0.9)) + theme_ctp () -> p
        print (p)
}

lm_sum <- function (xx, xcol, ycol){
        lm(as.formula (paste (xcol, '~', ycol)), xx) %>% summary () -> lin_model
        data.frame (
                R2=lin_model$r.squared,
                Tval=lin_model$coefficients [2, 't value'],
                pval=lin_model$coefficients [2, 'Pr(>|t|)']
        ) %>% return ()
}

lm_sum_all <- function (xx, xcols, ycol){
        xcols %>% as.list () %>% lapply (function (ii){
                lm_sum (xx, ii, ycol)
        }) -> df_list
        do.call ('rbind', df_list) %>% mutate (variable=xcols) %>% 
                mutate (padj = p.adjust (pval)) %>% 
        select (all_of (c('variable', 'R2', 'padj', 'Tval'))) %>% 
        mutate_if (is.numeric, function (xx){round (xx, 3)}) %>% 
        arrange (desc (R2) ) %>% return ()
}

theme_ctp <- function (fontsize=15, font_fam='arial'){
        list (ggplot2::theme(
              panel.grid.major = element_line(color='grey92'), 
              panel.grid.minor = element_line(color='grey92'), 
              panel.background = element_blank(),
              panel.border = element_blank(),

              axis.ticks.x = element_blank(),
              axis.ticks.y = element_blank(),
              axis.text.x = element_text(family=font_fam, hjust=0.5,
                                         size=fontsize, color='black'),
              axis.text.y = element_text(family=font_fam, size=fontsize,
                                         color='black'),
              axis.title.x = element_text(family=font_fam, size=fontsize),
              axis.title.y = element_text(family=font_fam, size=fontsize),
              
              legend.background= element_blank(),
              legend.key = element_blank(),
              strip.background = element_blank (),
              text=element_text (size=fontsize, family=font_fam),
              plot.title = element_text (hjust=0.5, size=fontsize*1.5,
                                         family=font_fam, face='bold'),
              aspect.ratio=1,
              strip.text = element_text (size=fontsize, family=font_fam,
                                         face='bold'),
              legend.text = element_text (size=fontsize, family=font_fam),
              legend.title = element_text (size=fontsize, family=font_fam, 
                                           face='bold') )
        )
}

plot_lm_val <- function (lin_df, yval='R2'){
        lin_df %>% arrange (!!as.symbol (yval)) %>%
                mutate (variable= factor (variable, levels=variable)) %>%
                mutate (logP = -log (padj) ) %>%
                ggplot (aes_string (x='variable', y=yval, fill='logP')) +
                        geom_bar (stat='identity') + 
                        scale_fill_viridis_c ()+
                        coord_flip () + theme_ctp ()
}

plot_var_imp <- function (model){
        var_imp <- caret::varImp (model)
        var_imp$importance %>% rownames_to_column ('variable') %>%
                arrange (Overall) %>%
                mutate (variable= factor (variable, levels=variable)) %>%
                ggplot (aes (x=Overall, y=variable)) +
                        geom_bar (stat='identity') + 
                        xlab ('importance') +theme_ctp ()
}

load_multimod <- function (save_dirs){
        acc_list <- list()
        for (i in 1:length (save_dirs)){
                acc_list [[i]] <- read.csv (paste (save_dirs[i], 'img_only.csv', sep='_'))
                acc_list [[i]]$modality <- gsub ('^.*/', '', save_dirs[i])
        }
        return (do.call ('rbind', acc_list))
}

rescale_g2 <- function (xx){
        xx [xx >= 3.75] <- 6
        xx [xx >= 3.25 & xx < 3.75] <- 5
        xx [xx >= 2.75 & xx < 3.25] <- 4
        xx [xx >= 2.25 & xx < 2.75] <- 3
        xx [xx < 2.25] <- 2
        return (xx)
}

rescale_mRS <- function (xx){
        xx [xx < 2] <- round (xx [xx <2], 0)
        xx [xx > 2] <- rescale_g2 (xx [xx >2])
        return (xx)
}
