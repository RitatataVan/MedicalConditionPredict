# ICR - Identifying Age-Related Conditions
Use Machine Learning to detect conditions with measurements of anonymous characteristics

![ICR](https://github.com/user-attachments/assets/1a1329bf-5453-4d7a-8321-c3823117b0a8)

## Objective
The aim is to predict if a person has any of three medical conditions. Specifically, it's to predict if the person has one or more of any of the three medical conditions (Class 1), or none of the three medical conditions (Class 0). We'll create a model trained on measurements of health characteristics. With predictive models, we can shorten this process and keep patient details private by collecting key characteristics relative to the conditions, then encoding these characteristics.

## Methods
-	Built LightGBM and CatBoost models and implemented ensemble modeling through model averaging to predict the likelihood of individuals having one of three age-related medical conditions based on 55 health characteristics
-	Applied cross-validation and Bayesian optimization techniques to minimize generalization error
-	Fine-tuned class weights for positive samples during model training to align with the evaluation metric of balanced logarithmic loss, handling imbalanced data issue
