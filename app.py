import heapq
import json
from flask import Flask
import pandas as pd
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.shell import spark


app = Flask(__name__)

recommendations = []
recipes_rated = []
recipes_recom = []
recom_results = []
rated_results = []

user_index = 1

recipes = np.load('recipes.npy')

ratings = pd.read_csv('ratings.csv')

ratings_pivot = ratings.pivot_table(values='rating', index=['user_id'], columns=['recipe_id'], fill_value=0,
                                    dropna=False)

ratings_values = ratings_pivot.values

ratings = spark.createDataFrame(ratings)

string_indexer1 = StringIndexer(inputCol="user_id", outputCol="user_id_index")
string_indexer2 = StringIndexer(inputCol="recipe_id", outputCol="recipe_id_index")

indexers = [string_indexer1, string_indexer2]

pipeline = Pipeline(stages=indexers)

ratings_final = pipeline.fit(ratings).transform(ratings)

als = ALS(rank=20, maxIter=20, regParam=0.1, userCol="user_id_index", itemCol="recipe_id_index", ratingCol="rating",
          coldStartStrategy="drop")
model = als.fit(ratings_final)

users_recs = model.recommendForAllUsers(10)

recipes_recommended_list = users_recs.where(users_recs.user_id_index == user_index).select('recommendations')

recipes_recommended = [i.recommendations for i in recipes_recommended_list.collect()]

for rec in recipes_recommended[0]:
    result = ratings_final.where(ratings_final.recipe_id_index == rec.recipe_id_index).select('recipe_id')
    recommendations.append(result.rdd.flatMap(list).first())

recipes_rated_index = heapq.nlargest(20, range(len(ratings_values[user_index])), ratings_values[user_index].take)
recipes_rated_id = ratings_pivot.columns[[recipes_rated_index]]

for rec1, rec2 in zip(recipes_rated_id, recommendations):
    for recipe in recipes:
        if rec1[1] == recipe.get('amazon_id'):
            recipes_rated.append(recipe)
        if rec2 == recipe.get('amazon_id'):
            recipes_recom.append(recipe)

for i in range(0, len(recipes_rated)):
    rated_results.append(recipes_rated[i].get('title'))

for i in range(0, len(recipes_recom)):
    recom_results.append(recipes_recom[i].get('title'))


@app.route('/reco', methods=['GET'])
def get_recommendation():

    return json.dumps(recom_results)


if __name__ == "__main__":
    app.run()

