# ブーステッド デシジョン ツリーを使用して、Azure Machine Learning デザイナーでチャーンを予測する
<!-- # Use boosted decision tree to predict churn with Azure Machine Learning designer -->

**デザイナーサンプル5**


デザイナーを使用してコードを 1 行も書かずに、複雑な機械学習パイプラインを構築する方法を学びます。
<!-- Learn how to build a complex machine learning pipeline without writing a single line of code using the designer. -->

デザイナーのホームページで **Binary Classification - Customer Relationship Prediction** サンプル パイプラインを開きます。 このパイプラインは、2 つの **2 クラス ブースト デシジョン ツリー** 分類器をトレーニングして、顧客関係管理 (CRM) システムの一般的なタスクである顧客離れを予測します。 データ値とラベルは複数のデータ ソースに分割され、顧客情報を匿名化するために使用されますが、デザイナーを使用してデータ セットを結合し、隠蔽された値を使用してモデルをトレーニングすることはできます。
<!-- Open the **Binary Classification - Customer Relationship Prediction** sample pipeline in the designer homepage. This pipeline trains 2 **two-class boosted decision tree** classifiers to predict common tasks for customer relationship management (CRM) systems - customer churn. The data values and labels are split across multiple data sources and scrambled to anonymize customer information, however, we can still use the designer to combine data sets and train a model using the obscured values. -->

「どれ？」という質問に答えようとしているため、これは分類問題と呼ばれますが、このサンプルに示されているのと同じロジックを適用して、回帰、分類、クラスタリングなど、あらゆる種類の機械学習の問題に取り組むことができます。
<!-- Because you're trying to answer the question "Which one?" this is called a classification problem, but you can apply the same logic shown in this sample to tackle any type of machine learning problem whether it be regression, classification, clustering, and so on. -->

このパイプラインの完成したグラフは次のとおりです。
<!-- Here's the completed graph for this pipeline: -->

![パイプライングラフ](./media/binary-classification-customer-relationship-prediction/pipeline-graph.png)


## データ

このパイプラインのデータは KDD Cup 2009 のものです。50,000 行と 230 の特徴列があります。 タスクは、これらの機能を使用する顧客のチャーン、欲求、アップセリングを予測することです。 データとタスクの詳細については、[KDD の Web サイト](https://www.kdd.org/kdd-cup/view/kdd-cup-2009) を参照してください。
<!-- The data for this pipeline is from KDD Cup 2009. It has 50,000 rows and 230 feature columns. The task is to predict churn, appetency, and up-selling for customers who use these features. For more information about the data and the task, see the [KDD website](https://www.kdd.org/kdd-cup/view/kdd-cup-2009). -->


## パイプラインの概要

デザイナーのこのサンプル パイプラインは、カスタマー リレーションシップ マネジメント (CRM) の一般的なタスクであるチャーン(churn)、欲求(appetency)、アップセリング(up-selling)のバイナリ分類器予測を示しています。
<!-- This sample pipeline in the designer shows binary classifier prediction of churn, appetency, and up-selling, a common task for customer relationship management (CRM). -->

まずは簡単なデータ処理。
<!-- First, some simple data processing. -->

- 生のデータセットに多くの欠損値があります。 **Clean Missing Data** モジュールを使用して、欠損値を 0 に置き換えます。<!-- - The raw dataset has many missing values. Use the **Clean Missing Data** module to replace the missing values with 0. -->
  ![データセットをきれいにする](media/binary-classification-customer-relationship-prediction/dataset.png)

- 特徴と対応するチャーンは異なるデータセットにあります。 **Add Columns** モジュールを使用して、ラベル列を特徴列に追加します。 最初の列 **Col1** はラベル列です。 視覚化の結果から、データセットが不均衡であることがわかります。 正の例 (+1) よりも負の (-1) 例の方がはるかに多くなります。 **SMOTE** モジュールを使用して、後で少数のケースを増やします。<!-- - The features and the corresponding churn are in different datasets. Use the **Add Columns** module to append the label columns to the feature columns. The first column, **Col1**, is the label column. From the visualization result we can see the dataset is unbalanced. There way more negative (-1) examples than positive examples (+1). We will use **SMOTE** module to increase underrepresented cases later. -->
  ![列データセットを追加](./media/binary-classification-customer-relationship-prediction/add-column.png)

- **Split Data** モジュールを使用して、データセットをトレーニング セットとテスト セットに分割します。
<!-- - Use the **Split Data** module to split the dataset into train and test sets. -->

- 次に、ブースト デシジョン ツリー バイナリ分類器を既定のパラメーターと共に使用して、予測モデルを構築します。 タスクごとに 1 つのモデルを構築します。つまり、それぞれ 1 つのモデルを構築して、アップセリング、欲求、チャーンを予測します。
<!-- - Then use the Boosted Decision Tree binary classifier with the default parameters to build the prediction models. Build one model per task, that is, one model each to predict up-selling, appetency, and churn. -->

- パイプラインの右側では、**SMOTE** モジュールを使用して、正例の割合を増やします。 SMOTE パーセンテージは 100 に設定され、正の例が 2 倍になります。 [SMOTE モジュール リファレンス 0](https://aka.ms/aml/smote) を使用して SMOTE モジュールがどのように機能するかの詳細をご覧ください。
<!-- - In the right part of the pipeline, we use **SMOTE** module to increase the percentage of positive examples. The SMOTE percentage is set to 100 to double the positive examples. Learn more on how SMOTE module works with [SMOTE module reference0](https://aka.ms/aml/smote). -->


## 結果

**Evaluate Model** モジュールの出力を視覚化して、テスト セットでのモデルのパフォーマンスを確認します。
<!-- Visualize the output of the **Evaluate Model** module to see the performance of the model on the test set.  -->

![結果を評価する](./media/binary-classification-customer-relationship-prediction/evaluate-result.png)

**しきい値** スライダーを動かして、二項分類タスクの指標の変化を確認できます。
<!-- You can move the **Threshold** slider and see the metrics change for the binary classification task.  -->


## 次のステップ

デザイナーが利用できるその他のサンプルを調べます。

- [サンプル 1 - 回帰: 自動車の価格を予測する](regression-automobile-price-prediction-basic.md)
- [サンプル 2 - 回帰: 自動車価格予測のアルゴリズムを比較](regression-automobile-price-prediction-compare-algorithms.md)
- [サンプル 3 - 特徴選択による分類: 所得予測](binary-classification-feature-selection-income-prediction.md)
- [サンプル 4 - 分類: 信用リスクの予測 (コスト重視)](binary-classification-python-credit-prediction.md)
- [サンプル 6 - 分類: フライトの遅延を予測する](r-script-flight-delay-prediction.md)
- [サンプル 7 - テキスト分類: ウィキペディア SP 500 データセット](text-classification-wiki.md)


---


Original: https://github.com/Azure/MachineLearningDesigner/blob/master/articles/samples/binary-classification-customer-relationship-prediction.md

<!-- # Use boosted decision tree to predict churn with Azure Machine Learning designer

**Designer sample 5**


Learn how to build a complex machine learning pipeline without writing a single line of code using the designer.

Open the **Binary Classification - Customer Relationship Prediction** sample pipeline in the designer homepage. This pipeline trains 2 **two-class boosted decision tree** classifiers to predict common tasks for customer relationship management (CRM) systems - customer churn. The data values and labels are split across multiple data sources and scrambled to anonymize customer information, however, we can still use the designer to combine data sets and train a model using the obscured values.

Because you're trying to answer the question "Which one?" this is called a classification problem, but you can apply the same logic shown in this sample to tackle any type of machine learning problem whether it be regression, classification, clustering, and so on.

Here's the completed graph for this pipeline:

![Pipeline graph](./media/binary-classification-customer-relationship-prediction/pipeline-graph.png)


## Data

The data for this pipeline is from KDD Cup 2009. It has 50,000 rows and 230 feature columns. The task is to predict churn, appetency, and up-selling for customers who use these features. For more information about the data and the task, see the [KDD website](https://www.kdd.org/kdd-cup/view/kdd-cup-2009).

## Pipeline summary

This sample pipeline in the designer shows binary classifier prediction of churn, appetency, and up-selling, a common task for customer relationship management (CRM).

First, some simple data processing.

- The raw dataset has many missing values. Use the **Clean Missing Data** module to replace the missing values with 0.

    ![Clean the dataset](media/binary-classification-customer-relationship-prediction/dataset.png)

- The features and the corresponding churn are in different datasets. Use the **Add Columns** module to append the label columns to the feature columns. The first column, **Col1**, is the label column. From the visualization result we can see the dataset is unbalanced. There way more negative (-1) examples than positive examples (+1). We will use **SMOTE** module to increase underrepresented cases later.

    ![Add the column dataset](./media/binary-classification-customer-relationship-prediction/add-column.png)

- Use the **Split Data** module to split the dataset into train and test sets.

- Then use the Boosted Decision Tree binary classifier with the default parameters to build the prediction models. Build one model per task, that is, one model each to predict up-selling, appetency, and churn.

- In the right part of the pipeline, we use **SMOTE** module to increase the percentage of positive examples. The SMOTE percentage is set to 100 to double the positive examples. Learn more on how SMOTE module works with [SMOTE module reference0](https://aka.ms/aml/smote).

## Results

Visualize the output of the **Evaluate Model** module to see the performance of the model on the test set. 

![Evaluate the results](./media/binary-classification-customer-relationship-prediction/evaluate-result.png)

 You can move the **Threshold** slider and see the metrics change for the binary classification task. 


## Next steps

Explore the other samples available for the designer:

- [Sample 1 - Regression: Predict an automobile's price](regression-automobile-price-prediction-basic.md)
- [Sample 2 - Regression: Compare algorithms for automobile price prediction](regression-automobile-price-prediction-compare-algorithms.md)
- [Sample 3 - Classification with feature selection: Income Prediction](binary-classification-feature-selection-income-prediction.md)
- [Sample 4 - Classification: Predict credit risk (cost sensitive)](binary-classification-python-credit-prediction.md)
- [Sample 6 - Classification: Predict flight delays](r-script-flight-delay-prediction.md)
- [Sample 7 - Text Classification: Wikipedia SP 500 Dataset](text-classification-wiki.md) -->
