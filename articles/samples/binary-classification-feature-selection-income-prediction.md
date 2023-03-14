# 分類器を構築し、特徴選択を使用して Azure Machine Learning デザイナーで収入を予測します
<!-- # Build a classifier & use feature selection to predict income with Azure Machine Learning designer -->

**デザイナーサンプル3**


デザイナーを使用してコードを 1 行も書かずに機械学習分類器を構築する方法を学びます。 このサンプルでは、**2 クラス ブースト デシジョン ツリー**をトレーニングして、成人の国勢調査収入 (>=50K または <=50K) を予測します。
<!-- Learn how to build a machine learning classifier without writing a single line of code using the designer. This sample trains a **two-class boosted decision tree** to predict adult census income (>=50K or <=50K). -->

質問は「どれ?」に答えるものであるため、これを分類問題と呼びます。 ただし、同じ基本的なプロセスを適用して、回帰、分類、クラスタリングなど、あらゆる種類の機械学習の問題に取り組むことができます。
<!-- Because the question is answering "Which one?", this is called a classification problem. However, you can apply the same fundamental process to tackle any type of machine learning problem - regression, classification, clustering, and so on. -->

このサンプルの最終的なパイプライン グラフは次のとおりです。
<!-- Here's the final pipeline graph for this sample: -->

![パイプラインのグラフ](./media/binary-classification-feature-selection-income-prediction/overall-graph.png)


## データ

データセットには、14 個の特徴と 1 つのラベル列が含まれています。 数値やカテゴリなど、複数のタイプの特徴があります。 次の図は、データセットからの抜粋を示しています。
<!-- The dataset contains 14 features and one label column. There are multiple types of features, including numerical and categorical. The following diagram shows an excerpt from the dataset: -->
![データ](media/binary-classification-feature-selection-income-prediction/sample3-dataset-1225.png)


## パイプラインの概要

次の手順に従って、パイプラインを作成します。
<!-- Follow these steps to create the pipeline: -->

1. Adult Census Income Binary データセット モジュールをパイプライン キャンバスにドラッグします。
<!-- 1. Drag the Adult Census Income Binary dataset module into the pipeline canvas. -->
2. **Split Data** モジュールを追加して、トレーニング セットとテスト セットを作成します。 最初の出力データセットの行の割合を 0.7 に設定します。 この設定は、データの 70% がモジュールの左側のポートに出力され、残りが右側のポートに出力されることを指定します。 左のデータセットをトレーニングに使用し、右のデータセットをテストに使用します。
<!-- 1. Add a **Split Data** module to create the training and test sets. Set the fraction of rows in the first output dataset to 0.7. This setting specifies that 70% of the data will be output to the left port of the module and the rest to the right port. We use the left dataset for training and the right one for testing. -->
3. **Filter Based Feature Selection** モジュールを追加して、ChiSquared によって 5 つの特徴を選択します。
<!-- 1. Add the **Filter Based Feature Selection** module to select 5 features by ChiSquared.  -->
4. **2 クラス ブースト デシジョン ツリー** モジュールを追加して、ブースト デシジョン ツリー分類器を初期化します。
<!-- 1. Add a **Two-Class Boosted Decision Tree** module to initialize a boosted decision tree classifier. -->
5. **モデルのトレーニング** モジュールを追加します。 前のステップの分類器を **Train Model** の左側の入力ポートに接続します。 フィルター ベースの特徴選択モジュールからフィルター処理されたデータセットをトレーニング データセットとして接続します。 **Train Model** は分類器をトレーニングします。
<!-- 1. Add a **Train Model** module. Connect the classifier from the previous step to the left input port of the **Train Model**. Connect the filtered dataset from Filter Based Feature Selection module as training dataset.  The **Train Model** will train the classifier. -->
6. Select Columns Transformation および Apply Transformation モジュールを追加して、同じ変換 (フィルタリングされたベースの特徴選択) をテスト データセットに適用します。
<!-- 1. Add Select Columns Transformation and Apply Transformation module to apply the same transformation (filtered based feature selection) to test dataset. -->
7. **Score Model** モジュールを追加し、**Train Model** モジュールをそれに接続します。 次に、テスト セット (特徴選択をテスト セットにも適用する Apply Transformation モジュールの出力) を **Score Model** に追加します。 **スコア モデル**が予測を行います。 その出力ポートを選択して、予測と正のクラスの確率を確認できます。<!-- 1. Add **Score Model** module and connect the **Train Model** module to it. Then add the test set (the output of Apply Transformation module which apply feature selection to test set too) to the **Score Model**. The **Score Model** will make the predictions. You can select its output port to see the predictions and the positive class probabilities. -->

    このパイプラインには 2 つのスコア モジュールがあり、右側のモジュールでは、予測を行う前にラベル列が除外されています。 これは、リアルタイム エンドポイントを展開する準備ができています。これは、Web サービス入力がラベルではなく機能のみを想定しているためです。
    <!-- This pipeline has two score modules, the one on the right has excluded label column before make the prediction. This is prepared to deploy a real-time endpoint, because the web service input will expect only features not label.  -->

8. **モデルの評価** モジュールを追加し、スコアリングされたデータセットを左側の入力ポートに接続します。 評価結果を表示するには、**Evaluate Model** モジュールの出力ポートを選択し、**Visualize** を選択します。
<!-- 1. Add an **Evaluate Model** module and connect the scored dataset to its left input port. To see the evaluation results, select the output port of the **Evaluate Model** module and select **Visualize**. -->


## 結果

![結果を評価する](media/binary-classification-feature-selection-income-prediction/sample3-evaluate-1225.png)

評価結果では、ROC、Precision-recall、混乱指標などの曲線が表示されます。
<!-- In the evaluation results, you can see that the curves like ROC, Precision-recall and confusion metrics.  -->


## リソースのクリーンアップ

[!INCLUDE [aml-ui-cleanup](../../includes/aml-ui-cleanup.md)]


## 次のステップ

デザイナーが利用できるその他のサンプルを調べます。
<!-- Explore the other samples available for the designer: -->

- [サンプル 1 - 回帰: 自動車の価格を予測する](regression-automobile-price-prediction-basic.md)
- [サンプル 2 - 回帰: 自動車価格予測のアルゴリズムを比較](regression-automobile-price-prediction-compare-algorithms.md)
- [サンプル 4 - 分類: 信用リスクの予測 (コスト重視)](binary-classification-python-credit-prediction.md)
- [サンプル 5 - 分類: チャーンを予測する](binary-classification-customer-relationship-prediction.md)
- [サンプル 6 - 分類: フライトの遅延を予測する](r-script-flight-delay-prediction.md)
- [サンプル 7 - テキスト分類: ウィキペディア SP 500 データセット](text-classification-wiki.md)


---


Original: https://github.com/Azure/MachineLearningDesigner/blob/master/articles/samples/binary-classification-feature-selection-income-prediction.md

<!-- # Build a classifier & use feature selection to predict income with Azure Machine Learning designer

**Designer sample 3**


Learn how to build a machine learning classifier without writing a single line of code using the designer. This sample trains a **two-class boosted decision tree** to predict adult census income (>=50K or <=50K).

Because the question is answering "Which one?", this is called a classification problem. However, you can apply the same fundamental process to tackle any type of machine learning problem - regression, classification, clustering, and so on.

Here's the final pipeline graph for this sample:

![Graph of the pipeline](./media/binary-classification-feature-selection-income-prediction/overall-graph.png)


## Data

The dataset contains 14 features and one label column. There are multiple types of features, including numerical and categorical. The following diagram shows an excerpt from the dataset:
![data](media/binary-classification-feature-selection-income-prediction/sample3-dataset-1225.png)



## Pipeline summary

Follow these steps to create the pipeline:

1. Drag the Adult Census Income Binary dataset module into the pipeline canvas.
1. Add a **Split Data** module to create the training and test sets. Set the fraction of rows in the first output dataset to 0.7. This setting specifies that 70% of the data will be output to the left port of the module and the rest to the right port. We use the left dataset for training and the right one for testing.
1. Add the **Filter Based Feature Selection** module to select 5 features by ChiSquared. 
1. Add a **Two-Class Boosted Decision Tree** module to initialize a boosted decision tree classifier.
1. Add a **Train Model** module. Connect the classifier from the previous step to the left input port of the **Train Model**. Connect the filtered dataset from Filter Based Feature Selection module as training dataset.  The **Train Model** will train the classifier.
1. Add Select Columns Transformation and Apply Transformation module to apply the same transformation (filtered based feature selection) to test dataset.
1. Add **Score Model** module and connect the **Train Model** module to it. Then add the test set (the output of Apply Transformation module which apply feature selection to test set too) to the **Score Model**. The **Score Model** will make the predictions. You can select its output port to see the predictions and the positive class probabilities.


    This pipeline has two score modules, the one on the right has excluded label column before make the prediction. This is prepared to deploy a real-time endpoint, because the web service input will expect only features not label. 

1. Add an **Evaluate Model** module and connect the scored dataset to its left input port. To see the evaluation results, select the output port of the **Evaluate Model** module and select **Visualize**.

## Results

![Evaluate the results](media/binary-classification-feature-selection-income-prediction/sample3-evaluate-1225.png)

In the evaluation results, you can see that the curves like ROC, Precision-recall and confusion metrics. 

## Clean up resources

[!INCLUDE [aml-ui-cleanup](../../includes/aml-ui-cleanup.md)]

## Next steps

Explore the other samples available for the designer:

- [Sample 1 - Regression: Predict an automobile's price](regression-automobile-price-prediction-basic.md)
- [Sample 2 - Regression: Compare algorithms for automobile price prediction](regression-automobile-price-prediction-compare-algorithms.md)
- [Sample 4 - Classification: Predict credit risk (cost sensitive)](binary-classification-python-credit-prediction.md)
- [Sample 5 - Classification: Predict churn](binary-classification-customer-relationship-prediction.md)
- [Sample 6 - Classification: Predict flight delays](r-script-flight-delay-prediction.md)
- [Sample 7 - Text Classification: Wikipedia SP 500 Dataset](text-classification-wiki.md) -->
