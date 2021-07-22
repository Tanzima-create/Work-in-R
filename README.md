# Work-in-R
# News Popularity 
rm(list=ls())
#FACEBOOK
library(dplyr)
library(data.table)
data=fread("/Users/tanz/Desktop/News_Final.csv", header = T)

names(data)<-tolower(names(data))
colnames(data)
str(data)
news_pop<-subset(data, select = -c(idlink,title,headline,source,topic,publishdate,sentimenttitle,sentimentheadline))

#Cleaning data
news_pop=data%>% filter(facebook>=0)%>% select(c("headline","facebook"))

#Creating popular/unpopular parameters
news_pop$facebook<-ifelse(news_pop$facebook>=10,'Popular','Unpopular')

# Sampling the data
news_sample = news_pop[sample(1:nrow(news_pop),nrow(news_pop)/3,
                   replace = FALSE)]

# selecting columns
str(news_sample)
summary(news_sample)

news_sample$facebook=factor(news_sample$facebook)

library(tm)
news_corpus=VCorpus(VectorSource(news_sample$headline))

inspect(news_corpus[1:2])
# converted to Character
lapply(news_corpus[1:3], as.character)
# making all to lower cases
news_corpus_ready=tm_map(news_corpus, content_transformer(tolower))

as.character(news_corpus_ready[[1]])
news_corpus_ready=tm_map(news_corpus_ready,removeNumbers)
news_corpus_ready=tm_map(news_corpus_ready,removeWords,stopwords())
news_corpus_ready=tm_map(news_corpus_ready,removePunctuation)

###############
library(SnowballC)
news_corpus_ready=tm_map(news_corpus_ready,stemDocument)
news_corpus_ready=tm_map(news_corpus_ready,stripWhitespace)
news_corpus_ready=tm_map(news_corpus_ready,removeNumbers)
##############
#Creates a DTM
news_dtm=DocumentTermMatrix(news_corpus_ready)

#converting to a matrix and to see
news_mat=as.matrix(news_dtm)

#Create a train and test set
dim(news_dtm)
#Create a train and test set with round the prediction number(train 80% and test 20%)
n = round(nrow(news_dtm)*.80,digits=0)
news_dtm_train=news_dtm[1:n,]
news_dtm_test=news_dtm[(n+1):nrow(news_dtm),]

# Creating labels (6530 train and 1633 test)
news_train_labels=news_sample[1:n,]$facebook
news_test_labels=news_sample[(n+1):nrow(news_dtm),]$facebook
length(news_test_labels)

# Naive bayes
news_freq=findFreqTerms(news_dtm_train,2)
library(dplyr)
news_dtm_freq_train=news_dtm_train[,news_freq]
news_dtm_freq_test=news_dtm_test[,news_freq]

convert_counts= function(x){
  x=ifelse(x>0, "Yes", "No")
}
news_train=apply(news_dtm_freq_train,MARGIN = 2, convert_counts)

news_test=apply(news_dtm_freq_test,MARGIN = 2, convert_counts)

#Phase- 2: Implement Bayes Algo and eval performance

library(e1071)
news_algo=naiveBayes(news_train, news_train_labels)
news_test_pred=predict(news_algo, news_test)

length(news_train)
library(gmodels)

CrossTable(news_test_pred,news_test_labels,
           prop.chisq = FALSE, prop.c = FALSE,
           prop.r = FALSE, dnn = c('classified','actual'))

