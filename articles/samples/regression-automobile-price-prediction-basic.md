# Azure Machine Learning デザイナーで回帰を使用して車の価格を予測する
<!-- # Use regression to predict car prices with Azure Machine Learning designer -->

**デザイナーサンプル1**

デザイナーを使用してコードを 1 行も書かずに機械学習回帰モデルを構築する方法を学びます。
<!-- Learn how to build a machine learning regression model without writing a single line of code using the designer. -->

このパイプラインは **linear regressor** (線形回帰学習器) をトレーニングして、メーカー、モデル、馬力、サイズなどの技術的特徴に基づいて自動車の価格を予測します。 「いくら？」という質問に答えようとしており、これは回帰問題と呼ばれます。 ただし、この例と同じ基本的な手順を適用して、回帰、分類、クラスタリングなど、あらゆるタイプの機械学習の問題に取り組むことができます。
<!-- This pipeline trains a **linear regressor** to predict a car's price based on technical features such as make, model, horsepower, and size. Because you're trying to answer the question "How much?" this is called a regression problem. However, you can apply the same fundamental steps in this example to tackle any type of machine learning problem whether it be regression, classification, clustering, and so on. -->

機械学習モデルのトレーニングの基本的な手順は次のとおりです。
<!-- The fundamental steps of a training machine learning model are: -->

1. データを取得する<!-- 1. Get the data -->
1. データの前処理<!-- 1. Pre-process the data -->
1. モデルをトレーニングする<!-- 1. Train the model -->
1. モデルを評価する<!-- 1. Evaluate the model -->

これは、パイプラインの最終的な完成グラフです。 この記事では、理論的根拠をすべてのモジュールに対して提供するため、あなたは同様の決定を自分自身で行うことができます。
<!-- Here's the final, completed graph of the pipeline. This article provides the rationale for all the modules so you can make similar decisions on your own. -->

![パイプラインのグラフ](./media/regression-automobile-price-prediction-basic/overall-graph.png)


## データを取得する

このサンプルでは、UCI Machine Learning Repository からの **Automobile price data (Raw)** データセットを使用します。 このデータセットには、メーカー、モデル、価格、車両の特徴 (シリンダー数など)、MPG (Miles per Gallon)、保険リスク スコアなど、自動車に関する情報を含む 26 列が含まれています。 このサンプルの目標は、車の価格を予測することです。
<!-- This sample uses the **Automobile price data (Raw)** dataset, which is from the UCI Machine Learning Repository. The dataset contains 26 columns that contain information about automobiles, including make, model, price, vehicle features (like the number of cylinders), MPG, and an insurance risk score. The goal of this sample is to predict the price of the car. -->


## データの前処理

主なデータ準備タスクには、データのクリーニング、統合、変換、削減、および離散化または量子化が含まれます。 デザイナーでは、左側のパネルの **Data Transformation** グループで、これらの操作やその他のデータ前処理タスクを実行するモジュールを見つけることができます。
<!-- The main data preparation tasks include data cleaning, integration, transformation, reduction, and discretization or quantization. In the designer, you can find modules to perform these operations and other data pre-processing tasks in the **Data Transformation** group in the left panel. -->

**Select Columns in Dataset** モジュールを使用して、多くの欠損値を持つ normalized-losses (ここでは自動車の年式に応じた相対的・平均的な売却価格下落を指す) を除外します。 次に、**Clean Missing Data** を使用して、欠損値のある行を削除します。 これは、トレーニング データのクリーンなセットを作成するのに役立ちます。
<!-- Use the **Select Columns in Dataset** module to exclude normalized-losses that have many missing values. Then use **Clean Missing Data** to remove the rows that have missing values. This helps to create a clean set of training data. -->

![データ前処理](./media/regression-automobile-price-prediction-basic/data-processing.png)


## モデルをトレーニングする

機械学習の問題はさまざまです。 一般的な機械学習タスクには、分類、クラスタリング、回帰、レコメンデーション システムが含まれ、それぞれに異なるアルゴリズムが必要になる場合があります。 アルゴリズムの選択は、多くの場合、ユース ケースの要件によって異なります。 アルゴリズムを選択したら、そのパラメーターを調整して、より正確なモデルをトレーニングする必要があります。 次に、精度、理解しやすさ、効率などの指標に基づいてすべてのモデルを評価する必要があります。
<!-- Machine learning problems vary. Common machine learning tasks include classification, clustering, regression, and recommender systems, each of which might require a different algorithm. Your choice of algorithm often depends on the requirements of the use case. After you pick an algorithm, you need to tune its parameters to train a more accurate model. You then need to evaluate all models based on metrics like accuracy, intelligibility, and efficiency. -->

このサンプルの目的は自動車の価格を予測することであり、ラベル列 (価格) は連続データであるため、回帰モデルが適しています。 このパイプラインには**線形回帰**を使用します。
<!-- Since the goal of this sample is to predict automobile prices, and because the label column (price) is continuous data, a regression model can be a good choice. We use **Linear Regression** for this pipeline. -->

**Split Data** モジュールを使用して入力データをランダムに分割し、トレーニング データセットに元のデータの 70% が含まれ、テスト データセットに元のデータの 30% が含まれるようにします。
<!-- Use the **Split Data** module to randomly divide the input data so that the training dataset contains 70% of the original data and the testing dataset contains 30% of the original data. -->


## テスト、評価、比較

データセットを分割し、異なるデータセットを使用してモデルのトレーニングとテストを行い、モデルの評価をより客観的にします。
<!-- Split the dataset and use different datasets to train and test the model to make the evaluation of the model more objective. -->

モデルのトレーニングが完了したら、**Score Model** モジュールと **Evaluate Model** モジュールを使用して、予測結果を生成し、モデルを評価できます。
<!-- After the model is trained, you can use the **Score Model** and **Evaluate Model** modules to generate predicted results and evaluate the models. -->

**Score Model** は、トレーニング済みのモデルを使用して、テスト データセットの予測を生成します。 結果を確認するには、**Score Model** の出力ポートを右クリックし、**Visualize** を選択します。
<!-- After the model is trained, you can use the **Score Model** and **Evaluate Model** modules to generate predicted results and evaluate the models. -->

![スコア結果](./media/regression-automobile-price-prediction-basic/scored-label.png)

スコアを **Evaluate Model** モジュールに渡して、評価指標を生成します。 結果を確認するには、**Evaluate Model** の出力ポートを右クリックし、**Visualize** を選択します。
<!-- Pass the scores to the **Evaluate Model** module to generate evaluation metrics. To check the result, right-click the output port of the **Evaluate Model** and then select **Visualize**. -->

![評価結果](./media/regression-automobile-price-prediction-basic/evaluate-model-output.png)


## 次のステップ

デザイナーが利用できるその他のサンプルを調べます。
<!-- Explore the other samples available for the designer: -->

- [サンプル 2 - 回帰: 自動車価格予測のアルゴリズムを比較](regression-automobile-price-prediction-compare-algorithms.md)
- [サンプル 3 - 特徴選択による分類: 所得予測](binary-classification-feature-selection-income-prediction.md)
- [サンプル 4 - 分類: 信用リスクの予測 (コスト重視)](binary-classification-python-credit-prediction.md)
- [サンプル 5 - 分類: チャーンを予測する](binary-classification-customer-relationship-prediction.md)
- [サンプル 6 - 分類: フライトの遅延を予測する](r-script-flight-delay-prediction.md)
- [サンプル 7 - テキスト分類: ウィキペディア SP 500 データセット](text-classification-wiki.md)


---


Original: https://github.com/Azure/MachineLearningDesigner/blob/master/articles/samples/regression-automobile-price-prediction-basic.md


<!-- # Use regression to predict car prices with Azure Machine Learning designer

**Designer sample 1**

Learn how to build a machine learning regression model without writing a single line of code using the designer.

This pipeline trains a **linear regressor** to predict a car's price based on technical features such as make, model, horsepower, and size. Because you're trying to answer the question "How much?" this is called a regression problem. However, you can apply the same fundamental steps in this example to tackle any type of machine learning problem whether it be regression, classification, clustering, and so on.

The fundamental steps of a training machine learning model are:

1. Get the data
1. Pre-process the data
1. Train the model
1. Evaluate the model

Here's the final, completed graph of the pipeline. This article provides the rationale for all the modules so you can make similar decisions on your own.

![Graph of the pipeline](./media/regression-automobile-price-prediction-basic/overall-graph.png)


## Get the data

This sample uses the **Automobile price data (Raw)** dataset, which is from the UCI Machine Learning Repository. The dataset contains 26 columns that contain information about automobiles, including make, model, price, vehicle features (like the number of cylinders), MPG, and an insurance risk score. The goal of this sample is to predict the price of the car.

## Pre-process the data

The main data preparation tasks include data cleaning, integration, transformation, reduction, and discretization or quantization. In the designer, you can find modules to perform these operations and other data pre-processing tasks in the **Data Transformation** group in the left panel.

Use the **Select Columns in Dataset** module to exclude normalized-losses that have many missing values. Then use **Clean Missing Data** to remove the rows that have missing values. This helps to create a clean set of training data.

![Data pre-processing](./media/regression-automobile-price-prediction-basic/data-processing.png)

## Train the model

Machine learning problems vary. Common machine learning tasks include classification, clustering, regression, and recommender systems, each of which might require a different algorithm. Your choice of algorithm often depends on the requirements of the use case. After you pick an algorithm, you need to tune its parameters to train a more accurate model. You then need to evaluate all models based on metrics like accuracy, intelligibility, and efficiency.

Since the goal of this sample is to predict automobile prices, and because the label column (price) is continuous data, a regression model can be a good choice. We use **Linear Regression** for this pipeline.

Use the **Split Data** module to randomly divide the input data so that the training dataset contains 70% of the original data and the testing dataset contains 30% of the original data.

## Test, evaluate, and compare

Split the dataset and use different datasets to train and test the model to make the evaluation of the model more objective.

After the model is trained, you can use the **Score Model** and **Evaluate Model** modules to generate predicted results and evaluate the models.

**Score Model** generates predictions for the test dataset by using the trained model. To check the result, right-click the output port of **Score Model** and then select **Visualize**.

![Score result](./media/regression-automobile-price-prediction-basic/scored-label.png)

Pass the scores to the **Evaluate Model** module to generate evaluation metrics. To check the result, right-click the output port of the **Evaluate Model** and then select **Visualize**.

![Evaluate result](./media/regression-automobile-price-prediction-basic/evaluate-model-output.png)


## Next steps

Explore the other samples available for the designer:

- [Sample 2 - Regression: Compare algorithms for automobile price prediction](regression-automobile-price-prediction-compare-algorithms.md)
- [Sample 3 - Classification with feature selection: Income Prediction](binary-classification-feature-selection-income-prediction.md)
- [Sample 4 - Classification: Predict credit risk (cost sensitive)](binary-classification-python-credit-prediction.md)
- [Sample 5 - Classification: Predict churn](binary-classification-customer-relationship-prediction.md)
- [Sample 6 - Classification: Predict flight delays](r-script-flight-delay-prediction.md)
- [Sample 7 - Text Classification: Wikipedia SP 500 Dataset](text-classification-wiki.md) -->
