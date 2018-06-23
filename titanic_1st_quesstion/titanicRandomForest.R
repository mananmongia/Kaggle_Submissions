# Titanic
# Importing the dataset
# setwd("~/Desktop/ann kaggle")
titanic.train = read.csv('train.csv',header = TRUE, stringsAsFactors = FALSE)
titanic.test = read.csv('test.csv',header = TRUE, stringsAsFactors = FALSE)

titanic.train$IsTrainSet = TRUE
titanic.test$IsTrainSet = FALSE
titanic.test$Survived = NA
titanic.full = rbind(titanic.train, titanic.test)
titanic.full[titanic.full$Embarked == '',"Embarked"] = 'S'
# titanic.full[is.na(titanic.full$Age),"Age"] = mean(titanic.full$Age, na.rm = TRUE)
# find missing fare ---- educated guess
upper.whisker = boxplot.stats(titanic.full$Fare)$stats[5]
outlier.filter = titanic.full$Fare < upper.whisker
fare.model = lm(formula = Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked,
                data = titanic.full[outlier.filter,])
# guess fare 
fare.row = titanic.full[is.na(titanic.full$Fare),
                        c("Pclass", "Sex", "Age", "SibSp", "Parch", "Embarked")]
fare.prediction = predict(fare.model,newdata = fare.row)
titanic.full[is.na(titanic.full$Fare),'Fare'] = fare.prediction
# titanic.full[is.na(titanic.full$Fare),"Fare"] = median(titanic.full$Fare, na.rm = TRUE)

# find missing Age ---- educated guess
ageupper.whisker = boxplot.stats(titanic.full$Age)$stats[5]
ageoutlier.filter = titanic.full$Age < ageupper.whisker
age.model = lm(formula = Age ~ Pclass + Sex + Fare + SibSp + Parch + Embarked,
                data = titanic.full[ageoutlier.filter,])
# guess fare 
age.row = titanic.full[is.na(titanic.full$Age),
                        c("Pclass", "Sex", "Fare", "SibSp", "Parch", "Embarked")]
age.prediction = predict(age.model,newdata = age.row)
titanic.full[is.na(titanic.full$Age),'Age'] = age.prediction

# Categorical Casting
titanic.full$Pclass = as.factor(titanic.full$Pclass)
titanic.full$Sex = as.factor(titanic.full$Sex)
titanic.full$Embarked = as.factor(titanic.full$Embarked)

# Split Dataset back out
titanic.train = titanic.full[titanic.full$IsTrainSet == TRUE,]
titanic.test = titanic.full[titanic.full$IsTrainSet == FALSE,]
titanic.train$Survived = as.factor(titanic.train$Survived)
survived.formula = as.formula("Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked")

library('randomForest')
titanic.model = randomForest(formula = survived.formula,
                             data = titanic.train,
                             ntree = 1000,
                             mtry = 3)


survived = predict(titanic.model, newdata = titanic.test) 
PassengerId = titanic.test$PassengerId
output.df = as.data.frame(PassengerId)
output.df$Survived = survived
write.csv(output.df,file = "submission_4.csv",row.names = FALSE)