# MP3-Machine-Learning-for-Prediction-by-Regression
## Created by Danyal, Pelle, Nicolai and Carsten

### question 1:

We have applied all 3 models (Linear, Multi linear and Polynomial) for future reference work.

### question 2:

It was difficult to create a model with a high enough prediction, to determend which model would be the best.
Multi Linear might have a better prediction score overall, because it took many more attributes into account, but it also needs more information, that might be harder to get for future use
Where as the polynomial gave a slight worse result, but it only took in one variable. Living sqft.

### question 3:

The result was: 
- 51% for the Linear model 
- 66% for the Multi-linear model
- 59% the Polynomial model

### question 4:
One team member tried to use the Ridge library on the polynomial model, and got a score at 68%, but since none of us could defend the work, or even properbly explain the difference,
we decided not to include it in the final work, and keep it as a comment in the documentation.

A mystery in the work process was, that when we removed the Outliers, the R-rating for our model actuelly went down. which we could not explain (if we remove the data which is not a part of the "mainstream" shouldn't the model prediction go up?)