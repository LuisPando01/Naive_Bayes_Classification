#EXTRACT DATA

sms_raw <- read.csv("sms_spam.csv")

#Review data
str(sms_raw)
str(sms_raw$type)

#Transform the column "type" to Factors
sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)

#Package
install.packages("tm")
library(tm) #For text mining

#Save the data
sms_corpus <- VCorpus(VectorSource(sms_raw$text))

#Verify the data
print(sms_corpus)
inspect(sms_corpus[1:2])
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:5], as.character)

###DATA CLEANING###

#standardise text

#Change texts to lower case
sms_corpus_clean <- tm_map(sms_corpus,content_transformer(tolower))
#Text comparison
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])

#Remove numbers
sms_corpus_clean <- tm_map(sms_corpus_clean,removeNumbers)

#Remove useless words
str(stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean,removeWords, stopwords())

#Remove punctuation and other characters
sms_corpus_clean <- tm_map(sms_corpus_clean,removePunctuation)

#Package
install.packages("SnowballC")
library(SnowballC) #For the wordStem function, works together with "tm"

#Stemming, transforming words so that they are not conjugated (transformed)
wordStem(c("learn", "learned", "learning", "learns"))
sms_corpus_clean <- tm_map(sms_corpus_clean,stemDocument)

#Removing blanks
sms_corpus_clean <- tm_map(sms_corpus_clean,stripWhitespace)

#Verify
lapply(sms_corpus[1:3],as.character)
lapply(sms_corpus_clean[1:3],as.character)

#CREATE THE TEXT MATRIX
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm

#Separate training and test data
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]

sms_train_labels <- sms_raw[1:4169, ]$type
sms_test_labels <- sms_raw[4170:5559, ]$type

prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))

#WORDCLOUD
install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50,random.order = FALSE)

###PREPARATION###
#Find words that appear in more than 5 messages
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)
#We keep the columns of words that are most frequently used
sms_dtm_freq_train <- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

#We convert the numbers in the matrix into Yes/No
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

#Transform into Yes/No
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2, convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2, convert_counts)

#Proportion
prop.table(table(sms_train))
prop.table(table(sms_test))

###TRAINING THE MODEL###

#Package
install.packages("e1071")
library(e1071) #For the naiveBayes function

#Create the model (use Bayes' theorem)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)
#Evaluate the model
sms_test_pred <- predict(sms_classifier, sms_test)

#Package
install.packages("gmodels")
library(gmodels) #For the CrossTable function

#Model results
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE,
           prop.r = FALSE,dnn = c('predicted', 'actual'))

#Model building using Laplace estimator
sms_classifier2 <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
#Model evaluation using the Laplace estimator
sms_test_pred2 <- predict(sms_classifier2, sms_test)
#Model results using the Laplace estimator
CrossTable(sms_test_pred2, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE,
           prop.r = FALSE, dnn = c('predicted', 'actual'))

###BUILD THE TABLE WITH RESULTS###
sms_test_prob <- predict(sms_classifier, sms_test, type = "raw")
#Verify
head(sms_test_prob)

###IMPORT THE RESULTS ALREADY WORKED ON###
sms_results <- read.csv("sms_results.csv", stringsAsFactors=TRUE)

#Verify
head(sms_results)
head(subset(sms_results, prob_spam > 0.40 & prob_spam < 0.60))
#Bad predictions
head(subset(sms_results, actual_type != predict_type))

###CONFUSION MATRICES###
#Create the matrix
CrossTable(sms_results$actual_type, sms_results$predict_type)

#Package to assign to values: positive/negative
install.packages("caret")
library(caret)

#Obtain measurements
#Sensitivity
sensitivity(sms_results$predict_type, sms_results$actual_type, positive = "spam")
#Specificity
specificity(sms_results$predict_type, sms_results$actual_type, negative = "ham")
#Precision
posPredValue(sms_results$predict_type, sms_results$actual_type, positive = "spam")
#Recall
sensitivity(sms_results$predict_type, sms_results$actual_type, positive = "spam")

#Function that returns all results
confusionMatrix(sms_results$predict_type, sms_results$actual_type, positive = "spam")

###CREATE THE ROC CURVE###
install.packages("ROCR")
library(ROCR)

#Create the prediction
pred <- prediction(predictions = sms_results$prob_spam,
                   labels = sms_results$actual_type)

#Build a performance object
perf <- performance(pred, measure = "tpr",
                    x.measure = "fpr")

#Plot the ROC curve
plot(perf, main = "ROC curve for SMS spam filter", col = "blue",
     lwd = 2)
abline(a = 0, b = 1, lwd = 2, lty = 2)

#Review the result
perf.auc <- performance(pred, measure = "auc")
str(perf.auc)
unlist(perf.auc@y.values)
