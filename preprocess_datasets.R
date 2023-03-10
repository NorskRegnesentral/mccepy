library(data.table)
data <- fread("Dropbox/pkg/MCCE_paper/mccepy/Data/german_credit_data.csv")
data <- data[,2:ncol(data)]
data[, .N, by = "Risk"][order(N)]

data <- data[complete.cases(data),]
data[, Sex := ifelse(Sex == "female", 1, 0)]
data[, Housing := ifelse(Housing == "own", 1, 0)]
# data[, `Saving accounts` := ifelse(`Saving accounts` == "moderate", 2, ifelse(`Saving accounts` == "quite rich", 3, 
#                                                                               ifelse(`Saving accounts` == 'rich', 4, 1)))]

data[, `Saving accounts` := ifelse(`Saving accounts` == "little", 1, 0)]
data[, `Checking account` := ifelse(`Checking account` == "little", 1, 0)]
# data[, `Checking account` := ifelse(`Checking account` == "moderate", 2, ifelse(`Checking accounts` == "quite rich", 3, 
#                                                                               ifelse(`Checking accounts` == 'rich', 4, 1)))]
data[, Purpose := ifelse(Purpose == 'car', 1, 0)]
data[, Risk := ifelse(Risk == 'bad', 1, 0)]
data[, Job := ifelse(Job == '2', 1, 0)]

data[, .N, by = "Risk"][order(N)]

# fwrite(data, "Dropbox/pkg/MCCE_paper/mccepy/Data/german_credit_data_complete.csv")

fico <- fread("Dropbox/pkg/MCCE_paper/mccepy/Data/heloc_dataset_v1.csv")
fico <- fico[complete.cases(fico), ]
setnames(fico, "RiskPerformance", "Risk")
fico[, Risk := ifelse(Risk == 'Bad', 1, 0)]
fico[, .N, by = Risk]

cols <- names(fico)
for(i in cols){
  fico <- fico[get(i) != -9]
}

fwrite(fico, "Dropbox/pkg/MCCE_paper/mccepy/Data/fico_data_complete.csv")
