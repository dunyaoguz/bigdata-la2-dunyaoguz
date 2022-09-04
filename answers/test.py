import os
import sys
from pyspark.rdd import RDD
from pyspark.sql import Row
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit
from pyspark.sql.functions import desc
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS

def toCSVLineRDD(rdd):
    '''
    This function convert an RDD or a DataFrame into a CSV string
    '''
    a = rdd.map(lambda row: ",".join([str(elt) for elt in row])) \
        .reduce(lambda x,y: '\n'.join([x,y]))
    return a + '\n'

def toCSVLine(data):
    '''
    Convert an RDD or a DataFrame into a CSV string
    '''
    if isinstance(data, RDD):
        return toCSVLineRDD(data)
    elif isinstance(data, DataFrame):
        return toCSVLineRDD(data.rdd)
    return None

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

lines = spark.read.text("data/sample_movielens_ratings.txt").rdd
parts = lines.map(lambda row: row.value.split("::"))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                     rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD)
(training, test) = ratings.randomSplit([0.8, 0.2], seed=123)

# create the transformations
user_mean = training.groupBy("userId").agg({"rating": "avg"}).withColumnRenamed("avg(rating)",
                                                                                "user_mean")
training = training.join(user_mean, "userId")
item_mean = training.groupBy("movieId").agg({"rating": "avg"}).withColumnRenamed("avg(rating)",
                                                                                 "item_mean")
training = training.join(item_mean, "movieId")
global_mean = training.agg({"rating": "avg"}).first()[0]
training = training.withColumn("user_item_interaction",
                               training.rating - (training.user_mean + training.item_mean -
                                                  global_mean))

# train the model
als = ALS(maxIter=5, rank=70, userCol="userId", itemCol="movieId",
          ratingCol="user_item_interaction",
          regParam=0.01, coldStartStrategy="drop")
als.setSeed(123)
model = als.fit(training)
# calculate rmse
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction_modified")
test = model.transform(test)

# create the transformations on test
test = test.join(user_mean, "userId")
test = test.join(item_mean, "movieId")
test = test.withColumn("prediction_modified", test.prediction + test.user_mean +
                       test.item_mean - global_mean)
rmse = evaluator.evaluate(test)
print(rmse)