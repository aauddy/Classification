getwd()
setwd("C:/Users/USER/Desktop/MS in Business Analytics/Data Mining/Project")
# Load the caret and related libraries
library(plyr)
library(Amelia)
library(caret)
library(caretEnsemble)
library(kernlab)
library(mlbench)
library(foreign)
library(ggplot2)
library(dplyr)
library(scales)
library(reshape)
library(e1071)
library(klaR)
library(MASS)
library(party)
library(lattice)
library(C50)
library(randomForest)
library(pROC)
# reading the training credit data
credit <- read.csv("C:/Users/USER/Desktop/MS in Business Analytics/Data Mining/Project/credit card clients.csv")
str(credit) # structure of the data
head(credit) # first 6 rows of the data


# Converting the features namely age,sex,education, marital status to factors

credit$AGE <- cut(credit$AGE,c(0,30,50,100),labels=c("young","middle","old"))
credit$SEX <- cut(credit$SEX,c(0,1,2),labels=c("Male","Female"))
credit$EDUCATION <- cut(credit$EDUCATION,c(0,1,2,3,4),
                        labels=c("Graduate School","University","High School","Others"))
credit$MARRIAGE <- cut(credit$MARRIAGE,c(-1,0,1,2,3),
                        labels=c("Unknown","Married","Single","Others"))
credit$PAY_0 <-as.factor(credit$PAY_0)
credit$PAY_2 <-as.factor(credit$PAY_2)
credit$PAY_3 <-as.factor(credit$PAY_3)
credit$PAY_4 <-as.factor(credit$PAY_4)
credit$PAY_5 <-as.factor(credit$PAY_5)
credit$PAY_6 <-as.factor(credit$PAY_6)

# changing the dependent variable i.e. payment.next.month column to factor
credit$default.payment.next.month <- as.factor(credit$default.payment.next.month)
head(credit)
# Checking for missing values
sapply(credit,function(x) sum(is.na(x)))
missmap(credit,main= "Missing Values Observed")
credit <- na.omit(credit)

# deleting the first column ID
credit$ID <- NULL

# plotting the data 

pairs(~LIMIT_BAL+ PAY_0+ PAY_2+ PAY_3 +
        PAY_4 + PAY_5 + PAY_6,data=credit, main="Scatterplot Matrix")

# number of defaulter vs non-defaulter
table(credit$default.payment.next.month)
# Plot the distribution using ggplot, see chaining in ggplot with + theme
qplot(default.payment.next.month, data=credit, geom = "bar") + theme(axis.text.x = element_text(angle = 90, hjust = 1))

# set the seed
set.seed(12345)

# Lets do stratified sampling. Select rows to based on default.payment.next.month variable as strata
TrainingDataIndex <- createDataPartition(credit$default.payment.next.month, p=0.45, list = FALSE)
TrainingDataIndex
# Create Training Data as subset of  dataset with row index numbers as identified above and all columns
trainData <- credit[TrainingDataIndex,]
table(trainData$default.payment.next.month)

# See percentages across classes
prop.table(table(trainData$default.payment.next.month))

# number of rows in training data
nrow(trainData)

# Compare percentages across classes between training and orginal data
DistributionCompare <- cbind(prop.table(table(trainData$default.payment.next.month)), prop.table(table(credit$default.payment.next.month)))
colnames(DistributionCompare) <- c("Training", "Orig")
DistributionCompare

# Melt Data - Convert from columns to rows
meltedDComp <- melt(DistributionCompare)
meltedDComp
# Plot to see distribution of training vs original - is it representative or is there over/under sampling?
ggplot(meltedDComp, aes(x= X1, y = value)) + geom_bar( aes(fill = X2), stat = "identity", position = "dodge") + theme(axis.text.x = element_text(angle = 90, hjust = 1))


# Everything else not in training is test data. Note the - (minus)sign
testData <- credit[-TrainingDataIndex,]

# We will use 10 fold cross validation to train and evaluate model
TrainingParameters <- trainControl(method = "cv", number = 10)
########################################################################
####################Decision Classification Model######################
#######################################################################

# Train a model with above parameters. We will use C5.0 algorithm
DecTreeModel <- train(default.payment.next.month ~ ., data = trainData, 
                      method = "C5.0",
                      trControl= TrainingParameters,
                      na.action = na.omit
)

#Lets take a look at results
DecTreeModel

# Plot performance
plot.train(DecTreeModel)
ggplot(DecTreeModel)

# Now make predictions on test set
DTPredictions <-predict(DecTreeModel, testData, na.action = na.pass)
DTPredictions
head(testData)

# Print confusion matrix and results the test data
cm <-confusionMatrix(DTPredictions, testData$default.payment.next.month,positive="1")
cm$overall
cm$byClass
cm

#################################################################################
####################Decision Tree Cost Classification Model######################
#################################################################################

statGrid <-  expand.grid(trials = 1,
                         model = "tree",
                         winnow = FALSE,
                         cost = matrix(c(
                           0, 1,
                           4, 0
                         ), 2, 2, byrow=TRUE))
# Train a model with above parameters. We will use C5.0Cost algorithm
DecTreeModelnew <- train(default.payment.next.month ~ ., data = trainData, 
                      method = "C5.0Cost",
                      trControl= TrainingParameters,
                      na.action = na.omit,
                      tuneGrid = statGrid, metric = "Accuracy"
)


DecTreeModelnew

# Plot performance
plot.train(DecTreeModelnew)
ggplot(DecTreeModelnew)

# Now make predictions on test set
DTPredictionsnew <-predict(DecTreeModelnew, testData, na.action = na.pass)
DTPredictionsnew


# Print confusion matrix and results the test data
cmnew <-confusionMatrix(DTPredictionsnew, testData$default.payment.next.month,positive="1")
cmnew$overall
cmnew$byClass
cmnew

#################################################################################
####################Decision Tree RPART Classification Model######################
#################################################################################

# Train a model with above parameters. We will use rpart algorithm
DecTreeModelrpart <- train(default.payment.next.month ~ ., data = trainData, 
                         method = "rpart",
                         trControl= TrainingParameters,
                         na.action = na.omit
)
                  
DecTreeModelrpart

# Plot performance
plot.train(DecTreeModelrpart)
ggplot(DecTreeModelrpart)

# Now make predictions on test set
DTPredictionsrpart <-predict(DecTreeModelrpart, testData, na.action = na.pass)
DTPredictionsrpart


# Print confusion matrix and results the test data
cmrpart <-confusionMatrix(DTPredictionsrpart, testData$default.payment.next.month,positive="1")
cmrpart$overall
cmrpart$byClass
cmrpart

############################################################################
##################### Random Forest Classification Model####################
############################################################################


# Build the model
rf_model <- randomForest(default.payment.next.month ~ .,data = trainData)
# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

varImpPlot(rf_model)
#Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance), 
                            Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
                           y = Importance, fill = Importance)) +
  geom_bar(stat='identity') + 
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust=0, vjust=0.55, size = 4, colour = 'red') +
  labs(x = 'Variables') +
  coord_flip() 

predictionrf <- predict(rf_model, testData)

cmrf <- confusionMatrix(predictionrf,testData$default.payment.next.month,positive = "1")
cmrf$overall
cmrf$byClass
cmrf
