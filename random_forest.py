"""
Tensorflow Random Forest for predicting house prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_decision_forests as tfdf
from sklearn.model_selection import train_test_split
import seaborn as sns


# %%codecell
df = pd.read_csv('train.csv').drop('Id', axis=1)


# %% codecell
sns.kdeplot(
    data=df,
    x='SalePrice'
)


# %% codecell
df_num = train.select_dtypes(include = ['float64', 'int64'])
df_num.hist(
    figsize=(16,20),
    bins=30,
    xlabelsize=8,
    ylabelsize=8
)


# %% codecell
# Prepare data for crossvalidation
X_train, X_test, y_train, y_test = train_test_split(
    df[[c for c in train.columns if c != 'SalePrice']],
    df[['SalePrice']],
    test_size=0.3,
    shuffle=True

)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)


# %% codecell
# Transform into tensorflow datasets
train_tf = tfdf.keras.pd_dataframe_to_tf_dataset(
    train,
    label='SalePrice',
    task=tfdf.keras.Task.REGRESSION
)
test_tf = tfdf.keras.pd_dataframe_to_tf_dataset(
    test,
    label='SalePrice',
    task=tfdf.keras.Task.REGRESSION
)


# %% codecell
# Create the model
rf = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
rf.compile(metrics=["mse"])


# %% codecell
# Train
rf.fit(x=train_tf)


# %% codecell
# Write model plot to html file
with open("plot.html", "w") as f:
    f.write(tfdf.model_plotter.plot_model(rf))


# %% codecell
# Evaluate model performance on training data
logs = rf.make_inspector().training_logs()
trees = [log.num_trees for log in logs]
rmse = [log.evaluation.rmse for log in logs]
plt.plot(trees, rmse)
plt.xlabel("Number of trees")
plt.ylabel("RMSE")
plt.show()


# %% codecell
# Crossvalidate
eval = rf.evaluate(x=test_tf,return_dict=True)
for k, v in eval.items():
  print(f"{k}: {v:.2f}")


# %% codecell
# Variable importance by SUM_SCORE
importance = rf.make_inspector().variable_importances()['SUM_SCORE']
featnames = [imp[0].name for imp in importance]
featimps = [imp[1] for imp in importance]

fig, ax = plt.subplots(1, 1)
ax.barh(featnames[:10], featimps[:10])


# %% codecell
# Predict competition dataset
competition = pd.read_csv('test.csv')
ids = competition.pop('Id')
competition_tf = tfdf.keras.pd_dataframe_to_tf_dataset(
    competition,
    task=tfdf.keras.Task.REGRESSION
)
pred = rf.predict(competition_tf)

# %% codecell
# Prepare submission
result = pd.DataFrame(
    {
        'Id': ids,
        'SalePrice': pred.squeeze()
    }
)


result.head(5)
result.to_csv('submission.csv', index=False)
