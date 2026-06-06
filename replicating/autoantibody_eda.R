library(edgeR)
library(data.table)

y    <- readRDS('../datasets/protogen_mat.rds')
clin <- fread('../datasets/clinical_03.03.21.csv')

clin[, s := ifelse(TIME == 'Baseline', paste0(Tacera_ID, '_BL'),
                   paste0(Tacera_ID, '_M6'))]
clin[, DAS28 := ifelse(TIME == 'Baseline', DAS28.0M, DAS28.6M)]
clin <- unique(clin[, .(s, TIME, SEX, AGE, BMI, ALCOHOL, DAS28)])

keep <- intersect(clin$s, colnames(y))
clin <- clin[s %in% keep]
y    <- y[, clin$s]

# Transformation — exakt wie im EDA
y_log <- cpm(y, log = TRUE, prior.count = 1)

# Export Matrix mit anonymen IDs aber gleicher Reihenfolge
write.csv(as.data.frame(y_log), '../datasets/protogen_log_cpm.csv')
write.csv(as.data.frame(clin),  '../datasets/protogen_clin.csv', row.names = FALSE)

print(paste("Samples:", ncol(y_log)))
print(paste("Antigene:", nrow(y_log)))
print(head(clin$s, 5))