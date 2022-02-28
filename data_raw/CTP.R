setwd ('..')
library (tidyverse)

# --------------------Select subjects--------------------
CTP_shape <- read.csv ('data/CTP/tmp/CTP_shape.csv', header=F)
NCCT_shape <- read.csv ('data/CTP/tmp/NCCT_skull_shape.csv', header=F)
CTA_shape <- read.csv ('data/CTP/tmp/CTA_skull_shape.csv', header=F)
subjects <- sort (reduce (list (CTP_shape$V1, NCCT_shape$V1, CTA_shape$V1), intersect))
#subjects <- sort (reduce (list (CTP_shape$V1, NCCT_shape$V1), intersect))

# --------------------Clean columns--------------------
adata <- read.csv ('data/CTP/original_labels/data_final_name.csv')
rownames (adata) <- adata$Anonymised
features <- c("Sex", "Age", "HTN", "Diabetes", "HC", "Smoke", "AF", "HF",
              "MI.Angina", "Side", "mRS_3m", "TotalDT2", "TotalDT3",
              "TotalDT4", "TotalDT6", "TotalDT8", "TotalDT10", "CoreCBF30",
              "MismCBF30", "CoreCBF35", "MismCBF35", "CoreCBF40", "MismCBF40",
              "CoreCBF45", "MismCBF45", "CoreCBV50", "MismCBV50", "CoreCBV55",
              "MismCBV55", "CoreCBV60", "MismCBV60", "CoreCBV65", "MismCBV65",
              "CoreAbsCBV", "MismAbsCBV", "Total", "Core", "Core_no_zero",
              "Penumbra", "Mismatch", "DT8", "BslmRS", "BslNIH", "tPAdelay",
              "X24hNIH", 'ICH')

adata %>% mutate (Sex = ifelse (Sex== 'M', 0, 1)) %>%
        mutate (Side = ifelse (Side== 'L', 0, 1)) %>%
        mutate (ICH = ifelse (ICH== 'No', 1, 0)) %>%
        mutate (Mismatch = ifelse (Mismatch == '>1000', 1000, Mismatch)) %>%
        filter (!is.na (HTN)) %>%
        mutate (X24hNIH = ifelse (X24hNIH == '?', NA, X24hNIH)) %>%
        mutate (tPAdelay = ifelse (tPAdelay== '?', NA, tPAdelay)) %>%
        mutate (BslmRS= ifelse (BslmRS== '?', NA, BslmRS)) %>%
        mutate (BslNIH= ifelse (BslNIH== '?', NA, BslNIH)) %>%
        select (all_of (features)) -> pdata

pdata %>% mutate (mRS_new= ifelse (mRS_3m == 3, 2.5, mRS_3m)) %>%
        mutate (mRS_new= ifelse (mRS_new == 4, 3, mRS_new)) %>%
        mutate (mRS_new= ifelse (mRS_new == 5, 3.5, mRS_new)) %>%
        mutate (mRS_new= ifelse (mRS_new == 6, 4, mRS_new)) %>%
        mutate (mRS_bool = ifelse (mRS_new >=2, 1, 0)) -> pdata

# save original values
rownames (pdata) <- rownames (adata)
pdata [rownames (pdata) %in% subjects,] -> nonimputed
set.seed (100)
ind <- caret::createDataPartition (nonimputed$mRS_bool, p=0.7, list=F)
write.csv (nonimputed[ind,], 'data/CTP/labels_186/train.csv')
write.csv (nonimputed[-ind,], 'data/CTP/labels_186/test.csv')

# impute missing data
output_col <-c('mRS_3m', 'ICH', 'mRS_new', 'mRS_bool', 'X24hNIH')
all_df<- pdata[, !colnames (pdata) %in% output_col]
caret::preProcess(all_df, method=c('knnImpute')) -> transformations
imputed <- predict(transformations, all_df)

# split to train vs test
patients <- rownames (pdata) %in% subjects
train_df <- list (x=imputed[patients,][ind, ], 
                  y= pdata[patients,][ind, output_col])
test_df <- list (x=imputed[patients,][-ind, ], 
                 y= pdata[patients,][-ind, output_col])

# save dataframe for machine learning
saveRDS (train_df, paste ('data/CTP/labels_186/train.rds'))
saveRDS (test_df, paste ('data/CTP/labels_186/test.rds'))
write.csv (train_df$x, paste ('data/CTP/labels_186/train_x.csv'))
write.csv (train_df$y, paste ('data/CTP/labels_186/train_y.csv'))
write.csv (test_df$x, paste ('data/CTP/labels_186/test_x.csv'))
write.csv (test_df$y, paste ('data/CTP/labels_186/test_y.csv'))

