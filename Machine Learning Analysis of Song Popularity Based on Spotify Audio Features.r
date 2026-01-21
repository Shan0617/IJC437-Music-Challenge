# 1. Basic setup

library(tidyverse)

install.packages("caret")
library(caret)

library(corrplot)
library(randomForest)
library(dplyr)

#2. Load dataset
data <- read.csv("acoustic_features.csv")

#3. Data pre-processing
#3.1 Create a binary categorical variable
data$is_hit <- ifelse(data$popularity >= 50, "Yes", "No") 
data$is_hit <- as.factor(data$is_hit)

#3.2 Remove irrelated data
model_data <- data %>% select(-song_id, -popularity, -year) 

#3.3 Convert categorical musical features to factors
model_data$key <- as.factor(model_data$key)
model_data$mode <- as.factor(model_data$mode)
model_data$time_signature <- as.factor(model_data$time_signature)

#4. Exploratior data analysis (EDA)
#4.1 Generate a correlation matrix heat map to visualize relationships between variables
numeric_data <- data %>% select(duration_ms, acousticness, danceability, 
                                energy, instrumentalness, liveness, 
                                loudness, speechiness, valence, tempo, popularity)
cor_matrix <- cor(numeric_data)
corrplot(cor_matrix, method = "color", addCoef.col = "black",number.cex = 0.4, title = "Feature Correlation",mar = c(0, 0, 3, 0))

#5. Train-Test Split
#5.1 split the data into training (80%) and testing (20%) sets
set.seed(42)
trainIndex <- createDataPartition(model_data$is_hit, p = .8, list = FALSE)
train_set <- model_data[trainIndex,]
test_set  <- model_data[-trainIndex,]

#6. Random Forest Model
#6.1 Create random forest model
rf_model <- randomForest(is_hit ~ ., data = train_set, 
                         ntree = 100, importance = TRUE)

#6.2 Evaluate model
predictions <- predict(rf_model, test_set) #Use the trained Random Forest model to predict outcomes for the test set
conf_matrix <- confusionMatrix(predictions, test_set$is_hit) #Compare predictions against actual values using a confusion matrix
print(conf_matrix) #Model's accuracy

#6.3 Check important feature for popularity song
importance_plot <- varImpPlot(rf_model, 
                              cex = 0.8, 
                              cex.lab = 0.7, 
                              main = "Feature Importance for Hit Potential")
#7. Logistic regression model
#7.1 Create logistic regression model
logit_model <- glm(is_hit ~ ., data = train_set, family = "binomial")

#7.2 Evaluate model
#7.2.1 Use the trained Logistic regression model to predict outcomes for the test set
prob_predictions <- predict(logit_model, test_set, type = "response")

#7.2.2 Convert probability predictions to binary classes
class_predictions <- ifelse(prob_predictions > 0.5, "Yes", "No")

#7.2.3 Calculate the classification error rate
class_error<-mean(class_predictions != test_set$is_hit)

#7.2.3 Calculate the classification accuracy rate
print(paste('Accuracy',1-class_error))

#7.2.4 Display the summary statistics for the logistic regression model to evaluate coefficient significance
summary(logit_model)













