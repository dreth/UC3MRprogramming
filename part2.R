library(VIM)
library(caret)
library(fastDummies)
library(dplyr)
library(RANN)
library(MLeval)
library(caretEnsemble)
df=read.csv("C:/Users/ander/OneDrive/Escritorio/Master/Programming/titanic.csv",na.strings = c("",NA))
g=read.csv("C:/Users/ander/OneDrive/Escritorio/Master/Programming/gender_submission.csv",na.strings = c("",NA))
for (i in g$PassengerId[1]:g$PassengerId[length(g$PassengerId)]){
  df$Survived[i]=g$Survived[i-891]
}



#Remove columns that we are not going to use.
df[df$Embarked==" "]<-NA
df=df[,-(4)]
df=df[,-(6:8)]
df=df[,-(7)]

#Define classes as character:
df$Pclass=as.character(df$Pclass)
df$Sex<-replace(df$Sex,df$Sex=='male',0)
df$Sex<-replace(df$Sex,df$Sex=='female',1)
df$Survived<-replace(df$Survived,df$Survived==1,'YES')
df$Survived<-replace(df$Survived,df$Survived==0,'NO')
df$Survived=as.factor(df$Survived)
df$Sex=as.factor(df$Sex)
str(df)

#Preprocess: Remove NAs using K-nearest neighbours
df=kNN(df,variable=c("Embarked","Age","Fare"))
df=df[,-(8:10)]
anyNA(df)

#One hot encoding (Dummy variables) of character variables:
df=dummy_cols(df,select_columns = c('Pclass','Embarked'))
df=df[,-c(3,7)]

#Create partition:
n=createDataPartition(df$PassengerId,p=0.8,list=FALSE)
train=df[n,]
train=df[,-1]
test=df[-n,]

#Store x and y for using them:
x_train=train[,(2:10)]
y_train=train$Survived

#Analysis of variables:
analy = x_train
apply(analy[,c(2,3)], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})

featurePlot(x = train[,(2:10)], 
            y = train$Survived, 
            plot = "box",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
#RFE:

set.seed(613)

options(warn = -1)
subsets = 1:9
ctrl = rfeControl(functions = rfFuncs, method = "repeatedcv", repeats = 5, verbose = FALSE)
lmProfile = rfe(x = x_train, y = y_train, sizes = subsets, rfeControl = ctrl,metric="Accuracy")
lmProfile


#Model fitting and predicting:
ctrl= trainControl(method='repeatedcv', number=5, savePredictions='final', classProbs=T, summaryFunction=twoClassSummary)
train_mars=train(Survived~.,data=train,method="earth",metric='accuracy',trControl=ctrl)
train_mars
p_mars=predict(train_mars,test)
confusionMatrix(reference=test$Survived,data=p_mars,mode="everything",positive = "YES")

roc_mars = evalm(train_mars)$roc

train_rf=train(Survived~.,data=train,method="rf",metric='ROC',trControl=ctrl)
train_rf
p_rf=predict(train_rf,test)
confusionMatrix(reference=test$Survived,data=p_rf,mode="everything",positive = "YES")

train_svmRadial=train(Survived~.,data=train,method="svmRadial",metric='ROC',trControl=ctrl)
train_svmRadial
p_svmRadial=predict(train_svmRadial,test)
confusionMatrix(reference=test$Survived,data=p_svmRadial,mode="everything",positive = "YES")

train_ada=train(Survived~.,data=train,method='adaboost',metric='ROC',trControl=ctrl)
train_ada
p_ada=predict(train_ada,test)
confusionMatrix(reference=test$Survived,data=p_ada,mode="everything",positive = "YES")

train_xgb=train(Survived~.,data=train,method='xgbDART',metric='ROC',trControl=ctrl)
train_xgb
p_xgb=predict(train_xgb,test)
confusionMatrix(reference=test$Survived,data=p_xgb,mode="everything",positive = "YES")

#Comparing models and ROC curves:

models_compare <- resamples(list(ADABOOST=train_ada, RF=train_rf, XGBDART=train_xgb, MARS=train_mars, SVM=train_svmRadial))
summary(models_compare)

res_mars = evalm(train_mars)

res_rf = evalm(train_rf)

res_svmRadial = evalm(train_svmRadial)

res_ada = evalm(train_ada)

res_xgb = evalm(train_xgb)

#Ensembling:
ctrlE <- trainControl(method="repeatedcv", number=10, repeats=5, savePredictions=TRUE, classProbs=TRUE)
algorithmList <- c('rf', 'adaboost', 'earth', 'xgbDART', 'svmRadial')
models <- caretList(Survived~.,data=train, trControl=ctrlE, methodList=algorithmList)
results <- resamples(models)
summary(results)

#Combining Models:
stackControl <- trainControl(method="repeatedcv", number=10, repeats=5, savePredictions=TRUE, classProbs=TRUE)
stack.glm <- caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)
p_com=predict(stack.glm,test)
confusionMatrix(reference=test$Survived,data=p_com,mode="everything",positive = "YES")
