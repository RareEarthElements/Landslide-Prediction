# Landslide-Prediction
Portfolio Machine Learning Modeling

**Context**

A government and geodata-consultant tried to analyze the observed data with various features from the severe earthquake on 12th May 2008, with magnitude 8.0, that shocked the Wenchuan area, Northwestern Sichuan Province, China. This natural phenomena led to the landslide-dammed lakes (“earthquake lakes”), debris flow, and rock avalanches – landslide which caused the most fatalities during that hazard event. The earthquake occurred along the Northeast oriented Longmenshan fault zone due to the collision between the Indian Plate and Eurasian Plate in conjunction with the eastern part of the Tibetan block and western part of the Sichuan Basin (Cui et al., 2011). The Longmen mount fault zone, and its surrounding area, have a humid subtropical climate which is common to the heavy rainfall (160 mm/day in 145 locations) between June and September, which is evident to the external dynamic conditions for the loose accumulation failure (Gan and Zhang, 2019). Therefore, by the geological fact, the area is prone to topographic movement like slide, collapse, erosion, and debris flow, even without the driving force of tectonic activity.

Prediction of landslides can be done if supported by the existence of data with representative features. Predicting landslides can be useful for stakeholders either government or third party to mitigate landslides. Landslide mitigation helps to minimize casualties and damages, thus machine learning modeling for landslide prediction is the appropriate solution. From the machine learning modeling, stakeholders can also estimate the budget needed for the landslide mitigation program, including taking into account the occurrence of false alarms from the weakness of machine learning modeling. Budget calculation for landslide mitigation by considering the evaluation of machine learning modeling is important so that the preparation of funds that need to be provided can be sufficient, not more or not less. Stakeholders can utilize predictions from machine learning models as a tool for landslide mitigation. Therefore, this task is important to support the landslide mitigation process planned by the stakeholders.


**Problem Statement**

How can we predict the likelihood of a landslide to occur? This prediction is crucial to provide landslide mitigation effectively. The application of machine learning methods is essential to facilitate predictions, offer more quantifiable insights, and provide recommendations.


**Goals**

The goal is for the stakeholders to predict the likelihood of a landslide to occur. This approach aims to make the stakeholders formulate strategies to mitigate the occurrence of landslides, in order to reduce the possibility of casualties and damages.


**Analytical Approach**

Our approach involves constructing a classification model in predicting the probability that a landslide will occur and provides an evaluation of what are the factors that influence the occurrence of landslides.


**Metrics**

   1. Accuracy: in this context would measure the overall correctness of the model's predictions. It tells us how often the model correctly predicts both landslide will occur and will not occur. However, if the dataset has imbalanced classes (more of one class than the other), high accuracy might be misleading as the model could simply predict the majority class.

   2. Recall: would be crucial to capture as many landslide as possible, we also need to consider getting a model with minimum false negatives, because it will be dangerous if it has a relatively high false negative value because it indicates a landslide but the model predicts a landslide did not happen which will be detrimental and dangerous.

   3. Precision: it is important to ensure that when the model predicts a landslide event, it is likely to be correct. High precision indicates that landslide events predicted to occur are more likely to happen, leading to better landslide mitigation because the model can predict well.

   4. F1-Score: is a combination of Precision and Recall. It's useful when we need to strike a balance between identifying as many positive cases as possible (Recall) while ensuring that those predictions are accurate (Precision). It helps to assess the overall effectiveness of the model.

   5. ROC-AUC Curve: is a representation of the trade-off between Sensitivity (Recall) and Specificity (1 - False Positive Rate). A model with a high ROC-AUC value indicates that the model is effective in distinguishing between landslides that will and will not occur.

In summary, for this classification task, we want to maximize the precision to predict the occurrence of landslides and the ROC-AUC Curve provides insight into the overall strength of the model.


**Machine Learning Method**

The landslide dataset has a relatively imbalanced proportion of target classes. However, the oversampling technique to overcome the imbalance data has an impact on the decrease in the model's precision score due to the increase in the recall score as a result of increasing the positive class data to be equal to the amount of data from the negative class. This can lead to more landslides that occur but are not predicted to occur, which is very dangerous, causing the risk of accidents to be higher compared to model that is not oversampled. However, it should be noted that high precision still has the possibility of false alarms, where an landslide is predicted to occur when in reality it did not. Therefore, for the case of landslide prediction, considering precision is more important than recall, we can use the model without oversampling instead.

In the selection of models for optimization, the XGBoost model has the highest ROC-AUC value with a value of 0.9748194970889036 compared to other tree-based models, thus the XGBoost model was selected to proceed to the hyperparameter tuning stage. From the classification report after hyperparameter tuning was applied, the precision, recall, and F-1 score metrics for class 1 have decreased, meaning that the tuning model has decreased performance in predicting class 1. Therefore, we will use the non-tuned XGBoost model as our final model. The following is the classification report of the XGBoost model:

                precision    recall  f1-score   support

         0.0       0.95      0.93      0.94      2789
         1.0       0.87      0.89      0.88      1344

    accuracy                           0.92      4133
    macro avg      0.91      0.91      0.91      4133
    weighted avg   0.92      0.91      0.92      4133
              
Based on the classification report results of our model (Default XGBoost model), we can conclude that if we later use our model to predict the occurrences of landslide, then our model can identify 93% (recall class 0) of landslides which will not occur so that stakeholders would not have to worry about preparing funds for landslide predictions that will not occur, and our model can get 89% (recall class 1) of landslides that will occur from all landslides that happened. (all of this is based on recall)

Our model has a prediction of occurred landslide of 87% (precision class 1), so every time our model predicts that a landslide that occur, the probability of guessing correctly is 87% or so. However, there will still be landslides who are not actually happened but are predicted as landslide about 7% or 182 (false positive) landslides that are considered as false alarms, this will be economically detrimental because we allocate funds for mitigation as the prediction results say it will happen when in fact the landslide did not happen. Moreover, there are still 11% or 144 (false negative) landslides that were predicted not to occur but actually did occur, this needs to be considered as it is dangerous and having the risk of causing casualties or damage.

**Feature Importance**

The top five influential features in predicting landslides using the XGBoost classifier are scarpdist (distance to previous failure scarps), frictang (friction angle), specwt (specific weight of soil), woods (presence of vegetation), and scarps (existence of failure scarps). The proximity to previous failure scarps indicates most likely increased landslide risk, while a lower friction angle and lower specific weight of soil may suggest reduced stability. The presence of vegetation reinforces the soil, decreasing the likelihood of landslides. Additionally, the existence of scarps signifies potential soil instability. Collectively, these features contribute significantly to the model's ability to predict landslides accurately.

**Business Scenario:**

If the cost to mitigate per landslide is $ 500 and if the number of landslide that possibly happen is 1,000 (of which 500 are most likely to happen, and 500 are not most likely to happen), then the calculation would be something like this:

No Models (all landslides we anticipate):

    Total Cost => 1,000 x 500 USD = 500,000 USD, because we anticipated all of the landslides
    Total savings => 0 USD

With the Model (only landslides predicted by the model are occurring that we mitigate):

    Total Cost => ((0.89 x 500) x 500 USD) + ((0.07 x 500) x 500 USD) = 240,000 USD
    Total landslide that will occur => 0.89 x 500 = 445 landslide (because the recall of 1/occurrence of landslide is 89%)
    Total landslide that fail to be anticipated => 0.11 x 500 = 55 landslide (because recall 1/occurrence of landslide 89%)
    Wasted cost by false alarm => (0.07 x 500) x 500 USD = 17,500 USD (based on recall 0/not occurrences of landslide)
    Total savings => 0.93 x 500 x 500 USD = 232,500 USD (only those landslides not occur are counted, those landslides occur but not anticipated are not counted here)

In terms of business implementation, it can be seen that landslide prediction by using our model has advantages such as:

1. **The company will have 52 % cheaper operational costs.**

       Percentage Cheaper Cost Allocation = ((Cost without ML − Cost with ML) / Cost without ML) x 100 %
   
       Percentage Cheaper Cost Allocation = ((500,000 − 240,000) / 500,000) x 100 % = 52 %
   
2. **The company saves 53.5% of operational costs.**

       Percentage Saving Cost Allocation = ((Cost without ML − Saving Cost with ML) / Cost without ML) x 100 %
   
       Percentage Saving Cost Allocation = ((500,000 − 232,500) / 500,000) x 100 % = 53.5%

The weaknesses of the machine learning model that need to be anticipated include:

1. **There were 7% false alarms and the cost wasted due to false alarms was 17,500 USD.**

2. **There are 11% of landslides that are not detected by the model or 55 landslides that will be missed, putting the landslide at risks.**


  **Recommended action**
  
* It is recommended that stakeholders use machine learning models to predict the occurrence of landslides, because the model used has a precision of 87% where the probability of guessing correctly is 87% and the ability of the model to distinguish classes 1 and 0 in the data also has excellent performance with a ROC-AUC score of 97%.
  
* Referring to the machine learning model and scenario to anticipate the possibility of 1000 landslides with mitigation cost per landslide of 500 USD, it is suggested that stakeholders should prepare at least 240,000 USD as fund allocation for landslide mitigation program.
  
* Locations with characteristics such as close proximity to the scarp or on the scarp, low friction angle and specific weight of soil, and absence of vegetation in those locations need to be considered and prioritized by stakeholders in developing landslide mitigation strategies. Areas with these characteristics are prone to landslides, if referring to the results of feature importance evaluation of machine learning modeling.
  
* Stakeholders need to add spatial information such as longitude and latitude so that data analysis using a geospatial approach can be carried out. In addition, it is also possible to integrate geographic information system (GIS) with machine learning so that predictions of landslides that will occur, false alarms, and predictions of landslides that are missed can be tracked, which makes it easier to develop landslide mitigation strategies. Anticipating false alarms and missed landslide predictions with spatial data will reduce the risk of casualties, damages, and economic losses.
