# 複数の回帰モデルをトレーニングして比較し、Azure Machine Learning デザイナーで自動車の価格を予測します
<!-- # Train & compare multiple regression models to predict car prices with Azure Machine Learning designer -->

<!-- 訳注: multiple regression models は「複数の回帰モデル」以外に「重回帰モデル」とも訳せますが、文意から前者で訳しています。 -->

**デザイナーサンプル2**


デザイナーを使用してコードを 1 行も書かずに機械学習パイプラインを構築する方法を学びます。 このサンプルでは、複数の回帰モデルをトレーニングして比較し、技術的特徴に基づいて自動車の価格を予測します。 独自の機械学習の問題に取り組むことができるように、このパイプラインで行われた選択の根拠を提供します。
<!-- Learn how to build a machine learning pipeline without writing a single line of code using the designer. This sample trains and compares multiple regression models to predict a car's price based on its technical features. We'll provide the rationale for the choices made in this pipeline so you can tackle your own machine learning problems. -->

機械学習を始めたばかりの場合は、このパイプラインの [基本バージョン](regression-automobile-price-prediction-basic.md) をご覧ください。
<!-- If you're just getting started with machine learning, take a look at the [basic version](regression-automobile-price-prediction-basic.md) of this pipeline. -->

このパイプラインの完成したグラフは次のとおりです。
<!-- Here's the completed graph for this pipeline: -->

[![パイプラインのグラフ](./media/regression-automobile-price-prediction-compare-algorithms/graph.png)](./media/regression-automobile-price-prediction-compare-algorithms/graph.png#lightbox)


## パイプラインの概要

次の手順を使用して、機械学習パイプラインを構築します。
<!-- Use following steps to build the machine learning pipeline: -->

1. データを取得します。<!-- 1. Get the data. -->
1. データを前処理します。<!-- 1. Pre-process the data. -->
1. モデルをトレーニングします。<!-- 1. Train the model. -->
1. モデルをテスト、評価、および比較します。<!-- 1. Test, evaluate, and compare the models. -->


## データを取得する

このサンプルでは、UCI Machine Learning Repository からの **Automobile price data (Raw)** データセットを使用します。 このデータセットには、メーカー、モデル、価格、車両の特徴 (シリンダー数など)、MPG (Miles per Gallon)、保険リスク スコアなど、自動車に関する情報を含む 26 列が含まれています。
<!-- This sample uses the **Automobile price data (Raw)** dataset, which is from the UCI Machine Learning Repository. This dataset contains 26 columns that contain information about automobiles, including make, model, price, vehicle features (like the number of cylinders), MPG, and an insurance risk score. -->


## データの前処理

主なデータ準備タスクには、データのクリーニング、統合、変換、削減、および離散化または量子化が含まれます。 デザイナーでは、左側のパネルの **Data Transformation** グループで、これらの操作やその他のデータ前処理タスクを実行するモジュールを見つけることができます。
<!-- The main data preparation tasks include data cleaning, integration, transformation, reduction, and discretization or quantization. In the designer, you can find modules to perform these operations and other data pre-processing tasks in the **Data Transformation** group in the left panel. -->

**Select Columns in Dataset** モジュールを使用して、多くの欠損値を持つ normalized-losses (ここでは自動車の年式に応じた相対的・平均的な売却価格下落を指す) を除外します。 次に **Clean Missing Data** を使用して、欠損値のある行を削除します。 これは、トレーニング データのクリーンなセットを作成するのに役立ちます。
<!-- Use the **Select Columns in Dataset** module to exclude normalized-losses that have many missing values. We then use **Clean Missing Data** to remove the rows that have missing values. This helps to create a clean set of training data. -->

![データ前処理](./media/regression-automobile-price-prediction-compare-algorithms/data-processing.png)


## モデルをトレーニングする

機械学習の問題はさまざまです。 一般的な機械学習タスクには、分類、クラスタリング、回帰、レコメンデーション システムが含まれ、それぞれに異なるアルゴリズムが必要になる場合があります。 アルゴリズムの選択は、多くの場合、ユース ケースの要件によって異なります。 アルゴリズムを選択したら、そのパラメーターを調整して、より正確なモデルをトレーニングする必要があります。 次に、精度、わかりやすさ、効率などの指標に基づいてすべてのモデルを評価する必要があります。
<!-- Machine learning problems vary. Common machine learning tasks include classification, clustering, regression, and recommender systems, each of which might require a different algorithm. Your choice of algorithm often depends on the requirements of the use case. After you pick an algorithm, you need to tune its parameters to train a more accurate model. You then need to evaluate all models based on metrics like accuracy, intelligibility, and efficiency. -->

このパイプラインの目的は自動車の価格を予測することであり、ラベル列 (価格) には実数が含まれているため、回帰モデルが適しています。
<!-- Because the goal of this pipeline is to predict automobile prices, and because the label column (price) contains real numbers, a regression model is a good choice. -->

さまざまなアルゴリズムのパフォーマンスを比較するために、**Boosted Decision Tree Regression** と **Decision Forest Regression** という 2 つの非線形アルゴリズムを使用してモデルを構築します。 どちらのアルゴリズムにも変更可能なパラメーターがありますが、このサンプルでは、このパイプラインの既定値を使用しています。
<!-- To compare the performance of different algorithms, we use two nonlinear algorithms, **Boosted Decision Tree Regression** and **Decision Forest Regression**, to build models. Both algorithms have parameters that you can change, but this sample uses the default values for this pipeline. -->

**Split Data** モジュールを使用して入力データをランダムに分割し、トレーニング データセットに元のデータの 70% が含まれ、テスト データセットに元のデータの 30% が含まれるようにします。
<!-- Use the **Split Data** module to randomly divide the input data so that the training dataset contains 70% of the original data and the testing dataset contains 30% of the original data. -->


## モデルをテスト、評価、比較する

前のセクションで説明したように、ランダムに選択された 2 つの異なるデータ セットを使用してモデルをトレーニングし、テストします。 データセットを分割し、異なるデータセットを使用してモデルのトレーニングとテストを行い、モデルの評価をより客観的にします。
<!-- You use two different sets of randomly chosen data to train and then test the model, as described in the previous section. Split the dataset and use different datasets to train and test the model to make the evaluation of the model more objective. -->

モデルのトレーニングが完了したら、**Score Model** モジュールと **Evaluate Model** モジュールを使用して予測結果を生成し、モデルを評価します。 **Score Model** は、トレーニング済みのモデルを使用して、テスト データセットの予測を生成します。 次に、スコアを **Evaluate Model** に渡して、評価指標を生成します。
<!-- After the model is trained, use the **Score Model** and **Evaluate Model** modules to generate predicted results and evaluate the models. **Score Model** generates predictions for the test dataset by using the trained model. Then pass the scores to **Evaluate Model** to generate evaluation metrics. -->


結果は次のとおりです。
<!-- Here are the results: -->

![結果を比較](./media/regression-automobile-price-prediction-compare-algorithms/result.png)

これらの結果は、**Boosted Decision Tree Regression** で構築されたモデルは、**Decision Forest Regression** で構築されたモデルよりも二乗平均平方根誤差が低いことを示しています。
<!-- These results show that the model built with **Boosted Decision Tree Regression** has a lower root mean squared error than the model built on **Decision Forest Regression**. -->


## 次のステップ

デザイナーが利用できるその他のサンプルを調べます。
<!-- Explore the other samples available for the designer: -->

- [サンプル 1 - 回帰: 自動車の価格を予測する](regression-automobile-price-prediction-basic.md)
- [サンプル 3 - 特徴選択による分類: 所得予測](binary-classification-feature-selection-income-prediction.md)
- [サンプル 4 - 分類: 信用リスクの予測 (コスト重視)](binary-classification-python-credit-prediction.md)
- [サンプル 5 - 分類: チャーンを予測する](binary-classification-customer-relationship-prediction.md)
- [サンプル 6 - 分類: フライトの遅延を予測する](r-script-flight-delay-prediction.md)
- [サンプル 7 - テキスト分類: ウィキペディア SP 500 データセット](text-classification-wiki.md)


---


Original: https://github.com/Azure/MachineLearningDesigner/blob/master/articles/samples/regression-automobile-price-prediction-compare-algorithms.md

<!-- # Train & compare multiple regression models to predict car prices with Azure Machine Learning designer

**Designer sample 2**


Learn how to build a  machine learning pipeline without writing a single line of code using the designer. This sample trains and compares multiple regression models to predict a car's price based on its technical features. We'll provide the rationale for the choices made in this pipeline so you can tackle your own machine learning problems.

If you're just getting started with machine learning, take a look at the [basic version](regression-automobile-price-prediction-basic.md) of this pipeline.

Here's the completed graph for this pipeline:

[![Graph of the pipeline](./media/regression-automobile-price-prediction-compare-algorithms/graph.png)](./media/regression-automobile-price-prediction-compare-algorithms/graph.png#lightbox)


## Pipeline summary

Use following steps to build the machine learning pipeline:

1. Get the data.
1. Pre-process the data.
1. Train the model.
1. Test, evaluate, and compare the models.

## Get the data

This sample uses the **Automobile price data (Raw)** dataset, which is from the UCI Machine Learning Repository. This dataset contains 26 columns that contain information about automobiles, including make, model, price, vehicle features (like the number of cylinders), MPG, and an insurance risk score.

## Pre-process the data

The main data preparation tasks include data cleaning, integration, transformation, reduction, and discretization or quantization. In the designer, you can find modules to perform these operations and other data pre-processing tasks in the **Data Transformation** group in the left panel.

Use the **Select Columns in Dataset** module to exclude normalized-losses that have many missing values. We then use **Clean Missing Data** to remove the rows that have missing values. This helps to create a clean set of training data.

![Data pre-processing](./media/regression-automobile-price-prediction-compare-algorithms/data-processing.png)

## Train the model

Machine learning problems vary. Common machine learning tasks include classification, clustering, regression, and recommender systems, each of which might require a different algorithm. Your choice of algorithm often depends on the requirements of the use case. After you pick an algorithm, you need to tune its parameters to train a more accurate model. You then need to evaluate all models based on metrics like accuracy, intelligibility, and efficiency.

Because the goal of this pipeline is to predict automobile prices, and because the label column (price) contains real numbers, a regression model is a good choice.

To compare the performance of different algorithms, we use two nonlinear algorithms, **Boosted Decision Tree Regression** and **Decision Forest Regression**, to build models. Both algorithms have parameters that you can change, but this sample uses the default values for this pipeline.

Use the **Split Data** module to randomly divide the input data so that the training dataset contains 70% of the original data and the testing dataset contains 30% of the original data.

## Test, evaluate, and compare the models

You use two different sets of randomly chosen data to train and then test the model, as described in the previous section. Split the dataset and use different datasets to train and test the model to make the evaluation of the model more objective.

After the model is trained, use the **Score Model** and **Evaluate Model** modules to generate predicted results and evaluate the models. **Score Model** generates predictions for the test dataset by using the trained model. Then pass the scores to **Evaluate Model** to generate evaluation metrics.



Here are the results:

![Compare the results](./media/regression-automobile-price-prediction-compare-algorithms/result.png)

These results show that the model built with **Boosted Decision Tree Regression** has a lower root mean squared error than the model built on **Decision Forest Regression**.


## Next steps

Explore the other samples available for the designer:

- [Sample 1 - Regression: Predict an automobile's price](regression-automobile-price-prediction-basic.md)
- [Sample 3 - Classification with feature selection: Income Prediction](binary-classification-feature-selection-income-prediction.md)
- [Sample 4 - Classification: Predict credit risk (cost sensitive)](binary-classification-python-credit-prediction.md)
- [Sample 5 - Classification: Predict churn](binary-classification-customer-relationship-prediction.md)
- [Sample 6 - Classification: Predict flight delays](r-script-flight-delay-prediction.md)
- [Sample 7 - Text Classification: Wikipedia SP 500 Dataset](text-classification-wiki.md) -->
