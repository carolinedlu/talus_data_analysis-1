################################
### MSSTATS
setwd("~/Desktop/talus/talus_data_analysis/R")

# install dependency packages for MSstats from CRAN
install.packages(c("gplots","lme4","ggplot2","ggrepel","reshape","reshape2",
                   "data.table","Rcpp","survival","minpack.lm", "stringi"))

# install dependency packages from BioConductor
BiocManager::install(c("limma","marray","preprocessCore","MSnbase", "MSstats"))

library(MSstats)

### WE ONLY NEED TO CHANGE THIS LINE FOR EACH NEW PROJECT
data_folder = "../data/210525_Gryder/"
###

peptide_protein_input_file = "peptide_proteins_results.csv"
peptide_protein_norm_output = "peptide_proteins_normalized.csv"
msstats_groupcompare_output = "msstats_groupcompare.csv"
comparison_matrix_input_file = "comparison_matrix.csv"

# read in the Skyline export data
raw = read.csv(paste(c(data_folder, peptide_protein_input_file), collapse=""))
head(raw)

# format the dataframe so that it plays nicely with MSstats
raw_msstats = SkylinetoMSstatsFormat(raw, filter_with_Qvalue=FALSE)

## 2. Data processing with the `dataProcess` function
quant_tmp = dataProcess(raw=raw_msstats, censoredInt='0')
write.csv(quant_tmp$ProcessedData, file=paste(c(data_folder, peptide_protein_norm_output), collapse=""), row.names=FALSE)

# check unique conditions and check order of condition information
levels(quant_tmp$ProcessedData$GROUP_ORIGINAL)

# load contrast matrix for Diseased vs Healthy
comparison <- data.matrix(read.csv(paste(c(data_folder, comparison_matrix_input_file), collapse=""), row.names=1))

# perform the group comparison
gpcomp_tmp <- groupComparison(contrast.matrix=comparison, data=quant_tmp)

# pull just the results out of the whole group comparison output
gpcomp_res <- gpcomp_tmp$ComparisonResult
write.csv(gpcomp_res, file=paste(c(data_folder, msstats_groupcompare_output), collapse=""), row.names=FALSE)

# subset only proteins with adjusted p-value < 0.05 and a FC > 2^2
list_sig <- gpcomp_res[gpcomp_res$adj.pvalue < 0.05 & gpcomp_res$log2FC > 2^2, ]
head(list_sig)
nrow(list_sig)

# Remove log files created by msstats
file.remove("msstats.log")
file.remove("msstats-1.log")
file.remove("sessionInfo.txt")
