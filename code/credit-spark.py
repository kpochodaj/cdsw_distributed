# original code https://towardsdatascience.com/build-an-end-to-end-machine-learning-model-with-mllib-in-pyspark-4917bdf289c5
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.types import *
spark = SparkSession\
    .builder\
    .appName("Home-credit-spark")\
    .master("local[*]") \
    .getOrCreate()

# READ SOURCE DATA
df = spark.read.csv("file:///home/cdsw/data/application_train.csv", header=True, inferSchema=True)

df.printSchema()

# PRE-PROCESS DATA
# Sk_ID_Curr is the id column which we dont need it in the process #so we get rid of it. and we rename the name of our
# target variable to "label"
drop_col = ['SK_ID_CURR']
df = df.select([column for column in df.columns if column not in drop_col])

# Spark requires target variable to be called label
df = df.withColumnRenamed('TARGET', 'label')

# EAD
# display first 10 rows
df.take(10)
print(df.limit(10).toPandas())

# display counts for labels
print(df.groupby('label').count().toPandas())

# the same with SQL
df.createOrReplaceTempView("application")
spark.sql("SELECT label, count(*) FROM application GROUP BY label LIMIT 10").show()

# data wrangling - clean data
# display counts of missing values
# we are going to fill the numerical missing values with the average of each column and the categorical missing values with the most frequent category of each column
# now let's see how many categorical and numerical features we have:

from collections import defaultdict

data_types = defaultdict(list)
for entry in df.schema.fields:
  data_types[str(entry.dataType)].append(entry.name)

# ONE-HOT ENCODING

# impute categorical
strings_used = [var for var in data_types["StringType"]]

missing_data_fill = {}
for var in strings_used:
  missing_data_fill[var] = "missing"

df = df.fillna(missing_data_fill)

# convert to numeric - 
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer

stage_string = [StringIndexer(inputCol= c, outputCol= c+"_string_encoded") for c in strings_used]
stage_one_hot = [OneHotEncoder(inputCol= c+"_string_encoded", outputCol= c+ "_one_hot") for c in strings_used]

ppl = Pipeline(stages= stage_string + stage_one_hot)
df = ppl.fit(df).transform(df)

# NUMERIC
numericals = data_types["DoubleType"]
numericals = [var for var in numericals]
numericals_imputed = [var + "_imputed" for var in numericals]

from pyspark.ml.feature import Imputer

imputer = Imputer(inputCols = numericals, outputCols = numericals_imputed)
df = imputer.fit(df).transform(df)

# cast integer to double and impute
for c in data_types["IntegerType"]:
  df = df.withColumn(c+ "_cast_to_double", df[c].cast("double"))

cast_vars = [var for var in  df.columns if var.endswith("_cast_to_double")]
cast_vars_imputed  = [var+ "imputed" for var in cast_vars]

imputer_for_cast_vars = Imputer(inputCols = cast_vars, outputCols = cast_vars_imputed)
df = imputer_for_cast_vars.fit(df).transform(df)

# vector assembly
from pyspark.ml.feature import VectorAssembler

features = cast_vars_imputed + numericals_imputed \
  + [var + "_one_hot" for var in strings_used]

vector_assembler = VectorAssembler(inputCols = features, outputCol= "features")
data_training_and_test = vector_assembler.transform(df)

data_training_and_test.printSchema()

# DIMENSIONALITY REDUCTION IN SPARK
# from pyspark.ml.feature import PCA
# pca_model = PCA(k = 30,inputCol = "features", outputCol = "pca_features")
# model = pca_model.fit(data_training_and_test)
# data_training_and_test = model.transform(data_training_and_test)

# SPLIT DATA INTO TRAIN AND TEST
train, test = data_training_and_test.randomSplit([0.80, 0.20], seed = 42)

print(train.count())
print(test.count())

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator as BCE

very_small_sample = data_training_and_test.sample(False, 0.001).cache()

# RANDOM FOREST
rf = RandomForestClassifier(labelCol = "label", featuresCol = "features")

# grid search
paramGrid = ParamGridBuilder() \
  .addGrid(rf.numTrees, [20, 30]) \
  .build()

crossval = CrossValidator(
  estimator = rf, 
  estimatorParamMaps=paramGrid, 
  evaluator = BCE(labelCol = "label",\
                  rawPredictionCol = "probability",\
                  metricName = "areaUnderROC"),
  numFolds= 3, 
)

cv_model = crossval.fit(very_small_sample)

# EVALUATE PERFORMANCE WITH TEST DATA
predictions = cv_model.transform(test)
evaluator= BCE(labelCol = "label", rawPredictionCol="probability", metricName= "areaUnderROC")
accuracy = evaluator.evaluate(predictions)
print(accuracy)
