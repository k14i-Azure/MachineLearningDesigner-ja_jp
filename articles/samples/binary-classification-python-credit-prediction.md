# 分類器を構築し、Python スクリプトを使用して、Azure Machine Learning デザイナーを使用して信用リスクを予測する
<!-- # Build a classifier & use Python scripts to predict credit risk using Azure Machine Learning designer -->

**デザイナーサンプル4**

この記事では、デザイナーを使用して複雑な機械学習パイプラインを構築する方法について説明します。 Python スクリプトを使用してカスタム ロジックを実装し、複数のモデルを比較して最適なオプションを選択する方法を学習します。
<!-- This article shows you how to build a complex machine learning pipeline using the designer. You'll learn how to implement custom logic using Python scripts and compare multiple models to choose the best option. -->

このサンプルでは、信用履歴、年齢、クレジット カードの枚数などの信用情報を使用して、信用リスクを予測する分類器をトレーニングします。 ただし、この記事の概念を適用して、独自の機械学習の問題に取り組むことができます。
<!-- This sample trains a classifier to predict credit risk using credit application information such as credit history, age, and number of credit cards. However, you can apply the concepts in this article to tackle your own machine learning problems. -->

このパイプラインの完成したグラフは次のとおりです。
<!-- Here's the completed graph for this pipeline: -->

[![パイプラインのグラフ](./media/binary-classification-python-credit-prediction/graph.png)](./media/binary-classification-python-credit-prediction/graph.png#lightbox)


## データ

このサンプルでは、UC Irvine リポジトリのドイツのクレジット カード データセットを使用します。 これには、20 個の特徴と 1 つのラベルを持つ 1,000 個のサンプルが含まれています。 各サンプルは人を表します。 20 の特徴には、数値の特徴とカテゴリの特徴が含まれます。 データセットの詳細については、[UCI ウェブサイト](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29) を参照してください。 最後の列はラベルで、信用リスクを示し、取り得る値は 2 つだけです: 高信用リスク = 2、低信用リスク = 1。
<!-- This sample uses the German Credit Card dataset from the UC Irvine repository. It contains 1,000 samples with 20 features and one label. Each sample represents a person. The 20 features include numerical and categorical features. For more information about the dataset, see the [UCI website](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29). The last column is the label, which denotes the credit risk and has only two possible values: high credit risk = 2, and low credit risk = 1. -->


## パイプラインの概要

このパイプラインでは、モデルを生成するための 2 つの異なるアプローチを比較して、この問題を解決します。
<!-- In this pipeline, you compare two different approaches for generating models to solve this problem: -->

- 元のデータセットを使用したトレーニング。<!-- - Training with the original dataset. -->
- 複製されたデータセットを使用したトレーニング。<!-- - Training with a replicated dataset. -->

どちらのアプローチでも、結果がコスト関数と一致していることを確認するために、テスト データセットとレプリケーションを使用してモデルを評価します。 2 つの分類器を両方のアプローチでテストします: **2 クラス サポート ベクター マシン** と **2 クラス ブースト デシジョン ツリー**。
<!-- With both approaches, you evaluate the models by using the test dataset with replication to ensure that results are aligned with the cost function. Test two classifiers with both approaches: **Two-Class Support Vector Machine** and **Two-Class Boosted Decision Tree**. -->

低リスクの例を高と誤分類するコストは 1 で、高リスクの例を低と誤分類するコストは 5 です。この誤分類のコストを考慮するために、**Python スクリプトの実行** モジュールを使用します。
<!-- The cost of misclassifying a low-risk example as high is 1, and the cost of misclassifying a high-risk example as low is 5. We use an **Execute Python Script** module to account for this misclassification cost. -->


## データ処理

**Metadata Editor** モジュールを使用して列名を追加し、デフォルトの列名を、UCI サイトのデータセットの説明から取得した、より意味のある名前に置き換えることから始めます。 **Metadata Editor** の **New column** name フィールドに新しい列名をコンマ区切り値として指定します。
<!-- Start by using the **Metadata Editor** module to add column names to replace the default column names with more meaningful names, obtained from the dataset description on the UCI site. Provide the new column names as comma-separated values in the **New column** name field of the **Metadata Editor**. -->

次に、リスク予測モデルの開発に使用するトレーニング セットとテスト セットを生成します。 **Split Data** モジュールを使用して、元のデータセットを同じサイズのトレーニング セットとテスト セットに分割します。 同じサイズのセットを作成するには、**最初の出力データセットの行の割合** オプションを 0.7 に設定します。
<!-- Next, generate the training and test sets used to develop the risk prediction model. Split the original dataset into training and test sets of the same size by using the **Split Data** module. To create sets of equal size, set the **Fraction of rows in the first output dataset** option to 0.7. -->

### 新しいデータセットを生成する

リスクを過小評価するコストは高いため、誤分類のコストを次のように設定します。
<!-- Because the cost of underestimating risk is high, set the cost of misclassification like this: -->

- 低リスクと誤分類された高リスクのケースの場合: 5<!-- - For high-risk cases misclassified as low risk: 5 -->
- 高リスクと誤分類された低リスクのケースの場合: 1<!-- - For low-risk cases misclassified as high risk: 1 -->

このコスト関数を反映するには、新しいデータセットを生成します。 新しいデータセットでは、高リスクの例がそれぞれ 5 回複製されますが、低リスクの例の数は変わりません。 レプリケーションの前にデータをトレーニング データセットとテスト データセットに分割して、同じ行が両方のセットに含まれないようにします。
<!-- To reflect this cost function, generate a new dataset. In the new dataset, each high-risk example is replicated five times, but the number of low-risk examples doesn't change. Split the data into training and test datasets before replication to prevent the same row from being in both sets. -->

リスクの高いデータをレプリケートするには、次の Python コードを **Execute Python Script** モジュールに入れます。
<!-- To replicate the high-risk data, put this Python code into an **Execute Python Script** module: -->

```Python
import pandas as pd

def azureml_main(dataframe1 = None, dataframe2 = None):

    df_label_1 = dataframe1[dataframe1.iloc[:, 20] == 1]
    df_label_2 = dataframe1[dataframe1.iloc[:, 20] == 2]

    result = df_label_1.append([df_label_2] * 5, ignore_index=True)
    return result,
```

**Python スクリプトの実行** モジュールは、トレーニング データセットとテスト データセットの両方を複製します。
<!-- The **Execute Python Script** module replicates both the training and test datasets. -->

### 特徴量エンジニアリング

**2 クラス サポート ベクター マシン** アルゴリズムには、正規化されたデータが必要です。 **Normalize Data** モジュールを使用して、すべての数値特徴の範囲を `tanh`変換で正規化します。 `tanh`変換は、値の全体的な分布を維持しながら、すべての数値機能を 0 から 1 の範囲内の値に変換します。
<!-- The **Two-Class Support Vector Machine** algorithm requires normalized data. So use the **Normalize Data** module to normalize the ranges of all numeric features with a `tanh` transformation. A `tanh` transformation converts all numeric features to values within a range of 0 and 1 while preserving the overall distribution of values. -->

**Two-Class Support Vector Machine** モジュールは、文字列特徴を処理し、それらをカテゴリ特徴に変換してから、値が 0 または 1 のバイナリ特徴に変換します。 したがって、これらの機能を正規化する必要はありません。
<!-- The **Two-Class Support Vector Machine** module handles string features, converting them to categorical features and then to binary features with a value of zero or one. So you don't need to normalize these features. -->


## モデル

**2 クラス サポート ベクター マシン** (SVM) と **2 クラス ブースト デシジョン ツリー** の 2 つの分類器と 2 つのデータセットを適用したため、合計 4 つのモデルが生成されます。
<!-- Because you applied two classifiers, **Two-Class Support Vector Machine** (SVM) and **Two-Class Boosted Decision Tree**, and two datasets, you generate a total of four models: -->

- 元のデータでトレーニングされた SVM。<!-- - SVM trained with original data. -->
- レプリケートされたデータでトレーニングされた SVM。<!-- - SVM trained with replicated data. -->
- 元のデータでトレーニングされたブースティング ディシジョン ツリー。<!-- - Boosted Decision Tree trained with original data. -->
- レプリケートされたデータでトレーニングされたブーストされたデシジョン ツリー。<!-- - Boosted Decision Tree trained with replicated data. -->

このサンプルでは、標準のデータ サイエンス ワークフローを使用して、モデルを作成、トレーニング、およびテストします。
<!-- This sample uses the standard data science workflow to create, train, and test the models: -->

1. **2 クラス サポート ベクター マシン** と **2 クラス ブースト デシジョン ツリー** を使用して、学習アルゴリズムを初期化します。<!-- 1. Initialize the learning algorithms, using **Two-Class Support Vector Machine** and **Two-Class Boosted Decision Tree**. -->
1. **モデルのトレーニング**を使用してアルゴリズムをデータに適用し、実際のモデルを作成します。<!-- 1. Use **Train Model** to apply the algorithm to the data and create the actual model. -->
1. **スコア モデル**を使用して、テスト例を使用してスコアを生成します。<!-- 1. Use **Score Model** to produce scores by using the test examples. -->


次の図は、このパイプラインの一部を示しています。ここでは、元のトレーニング セットと複製されたトレーニング セットを使用して 2 つの異なる SVM モデルをトレーニングしています。 **Train Model** はトレーニング セットに接続され、**Score Model** はテスト セットに接続されます。
<!-- The following diagram shows a portion of this pipeline, in which the original and replicated training sets are used to train two different SVM models. **Train Model** is connected to the training set, and **Score Model** is connected to the test set. -->

![パイプライングラフ](./media/binary-classification-python-credit-prediction/score-part.png)

パイプラインの評価段階では、4 つのモデルそれぞれの精度を計算します。 このパイプラインでは、**モデルの評価**を使用して、誤分類コストが同じサンプルを比較します。
<!-- In the evaluation stage of the pipeline, you compute the accuracy of each of the four models. For this pipeline, use **Evaluate Model** to compare examples that have the same misclassification cost. -->

**Evaluate Model** モジュールは、最大 2 つのスコア付きモデルのパフォーマンス メトリックを計算できます。 したがって、**モデルの評価** の 1 つのインスタンスを使用して 2 つの SVM モデルを評価し、**モデルの評価** の別のインスタンスを使用して 2 つのブースト デシジョン ツリー モデルを評価できます。
<!-- The **Evaluate Model** module can compute the performance metrics for as many as two scored models. So you can use one instance of **Evaluate Model** to evaluate the two SVM models and another instance of **Evaluate Model** to evaluate the two Boosted Decision Tree models. -->

レプリケートされたテスト データセットが **Score Model** の入力として使用されていることに注意してください。 つまり、最終的な精度スコアには、ラベルを間違えた場合のコストが含まれます。
<!-- Notice that the replicated test dataset is used as the input for **Score Model**. In other words, the final accuracy scores include the cost for getting the labels wrong. -->


## 複数の結果を組み合わせる

**Evaluate Model** モジュールは、さまざまな指標を含む 1 行のテーブルを生成します。 精度結果の 1 つのセットを作成するには、まず **Add Rows** を使用して結果を 1 つのテーブルに結合します。 次に、**Execute Python Script** モジュールで次の Python スクリプトを使用して、結果のテーブルの各行にモデル名とトレーニング アプローチを追加します。
<!-- The **Evaluate Model** module produces a table with a single row that contains various metrics. To create a single set of accuracy results, we first use **Add Rows** to combine the results into a single table. We then use the following Python script in the **Execute Python Script** module to add the model name and training approach for each row in the table of results: -->

```Python
import pandas as pd

def azureml_main(dataframe1 = None, dataframe2 = None):

    new_cols = pd.DataFrame(
            columns=["Algorithm","Training"],
            data=[
                ["SVM", "weighted"],
                ["SVM", "unweighted"],
                ["Boosted Decision Tree","weighted"],
                ["Boosted Decision Tree","unweighted"]
            ])

    result = pd.concat([new_cols, dataframe1], axis=1)
    return result,
```


## 結果

パイプラインの結果を表示するには、最後の **Select Columns in Dataset** モジュールの Visualize 出力を右クリックします。
<!-- To view the results of the pipeline, you can right-click the Visualize output of the last **Select Columns in Dataset** module. -->

![出力の可視化](media/binary-classification-python-credit-prediction/visualize-output.png)

最初の列には、モデルの生成に使用された機械学習アルゴリズムがリストされています。
<!-- The first column lists the machine learning algorithm used to generate the model. -->

2 番目の列は、トレーニング セットのタイプを示します。
<!-- The second column indicates the type of the training set. -->

3 番目の列には、コストに敏感な精度の値が含まれています。
<!-- The third column contains the cost-sensitive accuracy value. -->

これらの結果から、**2 クラス サポート ベクター マシン** で作成され、複製されたトレーニング データセットでトレーニングされたモデルによって、最高の精度が提供されることがわかります。
<!-- From these results, you can see that the best accuracy is provided by the model that was created with **Two-Class Support Vector Machine** and trained on the replicated training dataset. -->


## 次のステップ

デザイナーが利用できるその他のサンプルを調べます。
<!-- Explore the other samples available for the designer: -->

- [サンプル 1 - 回帰: 自動車の価格を予測する](regression-automobile-price-prediction-basic.md)
- [サンプル 2 - 回帰: 自動車価格予測のアルゴリズムを比較](regression-automobile-price-prediction-compare-algorithms.md)
- [サンプル 3 - 特徴選択による分類: 所得予測](binary-classification-feature-selection-income-prediction.md)
- [サンプル 5 - 分類: チャーンを予測する](binary-classification-customer-relationship-prediction.md)
- [サンプル 6 - 分類: フライトの遅延を予測する](r-script-flight-delay-prediction.md)
- [サンプル 7 - テキスト分類: ウィキペディア SP 500 データセット](text-classification-wiki.md)


---

Original: https://github.com/Azure/MachineLearningDesigner/blob/master/articles/samples/binary-classification-python-credit-prediction.md



<!-- # Build a classifier & use Python scripts to predict credit risk using Azure Machine Learning designer

**Designer sample 4**

This article shows you how to build a complex machine learning pipeline using the designer. You'll learn how to implement custom logic using Python scripts and compare multiple models to choose the best option.

This sample trains a classifier to predict credit risk using credit application information such as credit history, age, and number of credit cards. However, you can apply the concepts in this article to tackle your own machine learning problems.

Here's the completed graph for this pipeline:

[![Graph of the pipeline](./media/binary-classification-python-credit-prediction/graph.png)](./media/binary-classification-python-credit-prediction/graph.png#lightbox)


## Data

This sample uses the German Credit Card dataset from the UC Irvine repository. It contains 1,000 samples with 20 features and one label. Each sample represents a person. The 20 features include numerical and categorical features. For more information about the dataset, see the [UCI website](https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29). The last column is the label, which denotes the credit risk and has only two possible values: high credit risk = 2, and low credit risk = 1.

## Pipeline summary

In this pipeline, you compare two different approaches for generating models to solve this problem:

- Training with the original dataset.
- Training with a replicated dataset.

With both approaches, you evaluate the models by using the test dataset with replication to ensure that results are aligned with the cost function. Test two classifiers with both approaches: **Two-Class Support Vector Machine** and **Two-Class Boosted Decision Tree**.

The cost of misclassifying a low-risk example as high is 1, and the cost of misclassifying a high-risk example as low is 5. We use an **Execute Python Script** module to account for this misclassification cost.

## Data processing

Start by using the **Metadata Editor** module to add column names to replace the default column names with more meaningful names, obtained from the dataset description on the UCI site. Provide the new column names as comma-separated values in the **New column** name field of the **Metadata Editor**.

Next, generate the training and test sets used to develop the risk prediction model. Split the original dataset into training and test sets of the same size by using the **Split Data** module. To create sets of equal size, set the **Fraction of rows in the first output dataset** option to 0.7.

### Generate the new dataset

Because the cost of underestimating risk is high, set the cost of misclassification like this:

- For high-risk cases misclassified as low risk: 5
- For low-risk cases misclassified as high risk: 1

To reflect this cost function, generate a new dataset. In the new dataset, each high-risk example is replicated five times, but the number of low-risk examples doesn't change. Split the data into training and test datasets before replication to prevent the same row from being in both sets.

To replicate the high-risk data, put this Python code into an **Execute Python Script** module:

```Python
import pandas as pd

def azureml_main(dataframe1 = None, dataframe2 = None):

    df_label_1 = dataframe1[dataframe1.iloc[:, 20] == 1]
    df_label_2 = dataframe1[dataframe1.iloc[:, 20] == 2]

    result = df_label_1.append([df_label_2] * 5, ignore_index=True)
    return result,
```

The **Execute Python Script** module replicates both the training and test datasets.

### Feature engineering

The **Two-Class Support Vector Machine** algorithm requires normalized data. So use the **Normalize Data** module to normalize the ranges of all numeric features with a `tanh` transformation. A `tanh` transformation converts all numeric features to values within a range of 0 and 1 while preserving the overall distribution of values.

The **Two-Class Support Vector Machine** module handles string features, converting them to categorical features and then to binary features with a value of zero or one. So you don't need to normalize these features.

## Models

Because you applied two classifiers, **Two-Class Support Vector Machine** (SVM) and **Two-Class Boosted Decision Tree**, and two datasets, you generate a total of four models:

- SVM trained with original data.
- SVM trained with replicated data.
- Boosted Decision Tree trained with original data.
- Boosted Decision Tree trained with replicated data.

This sample uses the standard data science workflow to create, train, and test the models:

1. Initialize the learning algorithms, using **Two-Class Support Vector Machine** and **Two-Class Boosted Decision Tree**.
1. Use **Train Model** to apply the algorithm to the data and create the actual model.
1. Use **Score Model** to produce scores by using the test examples.

The following diagram shows a portion of this pipeline, in which the original and replicated training sets are used to train two different SVM models. **Train Model** is connected to the training set, and **Score Model** is connected to the test set.

![Pipeline graph](./media/binary-classification-python-credit-prediction/score-part.png)

In the evaluation stage of the pipeline, you compute the accuracy of each of the four models. For this pipeline, use **Evaluate Model** to compare examples that have the same misclassification cost.

The **Evaluate Model** module can compute the performance metrics for as many as two scored models. So you can use one instance of **Evaluate Model** to evaluate the two SVM models and another instance of **Evaluate Model** to evaluate the two Boosted Decision Tree models.

Notice that the replicated test dataset is used as the input for **Score Model**. In other words, the final accuracy scores include the cost for getting the labels wrong.

## Combine multiple results

The **Evaluate Model** module produces a table with a single row that contains various metrics. To create a single set of accuracy results, we first use **Add Rows** to combine the results into a single table. We then use the following Python script in the **Execute Python Script** module to add the model name and training approach for each row in the table of results:

```Python
import pandas as pd

def azureml_main(dataframe1 = None, dataframe2 = None):

    new_cols = pd.DataFrame(
            columns=["Algorithm","Training"],
            data=[
                ["SVM", "weighted"],
                ["SVM", "unweighted"],
                ["Boosted Decision Tree","weighted"],
                ["Boosted Decision Tree","unweighted"]
            ])

    result = pd.concat([new_cols, dataframe1], axis=1)
    return result,
```

## Results

To view the results of the pipeline, you can right-click the Visualize output of the last **Select Columns in Dataset** module.

![Visualize output](media/binary-classification-python-credit-prediction/visualize-output.png)

The first column lists the machine learning algorithm used to generate the model.

The second column indicates the type of the training set.

The third column contains the cost-sensitive accuracy value.

From these results, you can see that the best accuracy is provided by the model that was created with **Two-Class Support Vector Machine** and trained on the replicated training dataset.


## Next steps

Explore the other samples available for the designer:

- [Sample 1 - Regression: Predict an automobile's price](regression-automobile-price-prediction-basic.md)
- [Sample 2 - Regression: Compare algorithms for automobile price prediction](regression-automobile-price-prediction-compare-algorithms.md)
- [Sample 3 - Classification with feature selection: Income Prediction](binary-classification-feature-selection-income-prediction.md)
- [Sample 5 - Classification: Predict churn](binary-classification-customer-relationship-prediction.md)
- [Sample 6 - Classification: Predict flight delays](r-script-flight-delay-prediction.md)
- [Sample 7 - Text Classification: Wikipedia SP 500 Dataset](text-classification-wiki.md) -->
