# 分類器を作成し、R を使用して Azure Machine Learning デザイナーでフライトの遅延を予測する
<!-- # Build a classifier & use R to predict flight delays with Azure Machine Learning designer -->

**デザイナーサンプル6**


このパイプラインは、過去のフライト データと気象データを使用して、予定されている旅客便が 15 分以上遅れるかどうかを予測します。 この問題は、2 つのクラス (遅延または時間通り) を予測する分類問題としてアプローチできます。
<!-- This pipeline uses historical flight and weather data to predict if a scheduled passenger flight will be delayed by more than 15 minutes. This problem can be approached as a classification problem, predicting two classes: delayed, or on time. -->

このサンプルの最終的なパイプライン グラフは次のとおりです。
<!-- Here's the final pipeline graph for this sample: -->

[![パイプラインのグラフ](media/r-script-flight-delay-prediction/pipeline-graph.png)](media/r-script-flight-delay-prediction/pipeline-graph.png#lightbox)


## データ

このサンプルでは、**Flight Delays Data** データセットを使用しています。 これは、米国運輸省の TranStats データ コレクションの一部です。 このデータセットには、2013 年 4 月から 10 月までのフライト遅延情報が含まれています。データセットは次のように前処理されています。
<!-- This sample uses the **Flight Delays Data** dataset. It's part of the TranStats data collection from the U.S. Department of Transportation. The dataset contains flight delay information from April to October 2013. The dataset has been pre-processed as follows: -->

* 米国本土で最も利用者の多い 70 の空港を含むようにフィルタリングされています。<!-- * Filtered to include the 70 busiest airports in the continental United States. -->
* キャンセルされたフライトを 15 分以上遅延したものとして再ラベル付けしました。<!-- * Relabeled canceled flights as delayed by more than 15 mins. -->
* 迂回されたフライトを除外しました。<!-- * Filtered out diverted flights. -->
* 14 列を選択。<!-- * Selected 14 columns. -->

フライト データを補足するために、**Weather Dataset** が使用されます。 気象データには、NOAA による 1 時間ごとの陸上気象観測が含まれており、フライト データセットと同じ期間をカバーする空港の気象観測所からの観測を表しています。 次のように前処理されています。
<!-- To supplement the flight data, the **Weather Dataset** is used. The weather data contains hourly, land-based weather observations from NOAA, and represents observations from airport weather stations, covering the same time period as the flights dataset. It has been pre-processed as follows: -->

* ウェザー ステーション ID は、対応する空港 ID にマッピングされました。<!-- * Weather station IDs were mapped to corresponding airport IDs. -->
* 最も混雑する 70 の空港に関連付けられていない気象観測所は削除されました。<!-- * Weather stations not associated with the 70 busiest airports were removed. -->
* Date 列は、年、月、日という個別の列に分割されました。<!-- * The Date column was split into separate columns: Year, Month, and Day. -->
* 26 列を選択しました。<!-- * Selected 26 columns. -->


## データの前処理

通常、データセットを分析するには、前処理が必要です。
<!-- A dataset usually requires some pre-processing before it can be analyzed. -->

![データ処理](./media/r-script-flight-delay-prediction/data-process.png)

### フライトデータ

列 **Carrier**、**OriginAirportID**、および **DestAirportID** は整数として保存されます。 ただし、これらはカテゴリ属性であるため、**Edit Metadata** モジュールを使用してカテゴリ属性に変換します。
<!-- The columns **Carrier**, **OriginAirportID**, and **DestAirportID** are saved as integers. However, they're  categorical attributes, use the **Edit Metadata** module to convert them to categorical. -->

![編集-メタデータ](./media/r-script-flight-delay-prediction/edit-metadata.png)

次に、データセット モジュールの **Select Columns** を使用して、潜在的なターゲット リーカー (推論用データが入手できないにもかかわらず学習時に特徴列として扱うことで、機械学習モデルに含まれてしまう、精度を歪める要因) であるデータセット列を除外します: **DepDelay**、**DepDel15**、**ArrDelay**、**Canceled**、**Year**。
<!-- Then use the **Select Columns** in Dataset module to exclude from the dataset columns that are possible target leakers: **DepDelay**, **DepDel15**, **ArrDelay**, **Canceled**, **Year**.  -->

フライト レコードを 1 時間ごとの気象レコードと結合するには、出発予定時刻を結合キーの 1 つとして使用します。 結合を行うには、CSRDepTime 列を最も近い時間に切り下げる必要があります。これは **R スクリプトの実行** モジュールで行われます。
<!-- To join the flight records with the hourly weather records, use the scheduled departure time as one of the join keys. To do the join, the CSRDepTime column must be rounded down to the nearest hour, which is done by in the **Execute R Script** module.  -->

### 気象データ

欠損値の割合が高い列は、**Project Columns** モジュールを使用して除外されます。 これらの列には、**ValueForWindCharacter**、**WetBulbFarenheit**、**WetBulbCelsius**、**PressureTendency**、**PressureChange**、**SeaLevelPressure**、**StationPressure** のすべての文字列値の列が含まれます。
<!-- Columns that have a large proportion of missing values are excluded using the **Project Columns** module. These columns include all string-valued columns: **ValueForWindCharacter**, **WetBulbFarenheit**, **WetBulbCelsius**, **PressureTendency**, **PressureChange**, **SeaLevelPressure**, and **StationPressure**. -->

次に、**Clean Missing Data** モジュールが残りの列に適用され、欠損データのある行が削除されます。
<!-- The **Clean Missing Data** module is then applied to the remaining columns to remove rows with missing data. -->

気象観測時間は、最も近い 1 時間に切り上げられます。 予定された飛行時間と気象観測時間は、モデルが飛行時間前の天気のみを使用するように、反対方向に丸められます。
<!-- Weather observation times are rounded up to the nearest full hour. Scheduled flight times and the weather observation times are rounded in opposite directions to ensure the model uses only weather before the flight time.  -->

気象データは現地時間で報告されるため、タイム ゾーンの違いは、出発予定時刻と気象観測時刻からタイム ゾーンの列を差し引くことによって考慮されます。 これらの操作は、**Execute R Script** モジュールを使用して実行されます。
<!-- Since weather data is reported in local time, time zone differences are accounted for by subtracting the time zone columns from the scheduled departure time and the weather observation time. These operations are done using the **Execute R Script** module. -->


### データセットの結合

フライト レコードは、**Join Data** モジュールを使用して、フライトの出発地 (**OriginAirportID**) の気象データと結合されます。
<!-- Flight records are joined with weather data at origin of the flight (**OriginAirportID**) using the **Join Data** module. -->

  ![出発地ごとにフライトと天気に参加](./media/r-script-flight-delay-prediction/join-origin.png)


フライト記録は、フライトの目的地 (**DestAirportID**) を使用して気象データと結合されます。
<!-- Flight records are joined with weather data using the destination of the flight (**DestAirportID**). -->

  ![目的地別のフライトと天気に参加](./media/r-script-flight-delay-prediction/join-destination.png)

### トレーニング サンプルとテスト サンプルの準備

**Split Data** モジュールは、トレーニング用の 4 月から 9 月のレコードとテスト用の 10 月のレコードにデータを分割します。
<!-- The **Split Data** module splits the data into April through September records for training, and October records for test. -->

  ![訓練データとテストデータの分割](./media/r-script-flight-delay-prediction/split.png)

年、月、タイムゾーンの列は、列の選択モジュールを使用してトレーニング データセットから削除されます。
<!-- Year, month, and timezone columns are removed from the training dataset using the Select Columns module. -->

## 特徴を定義する

機械学習では、特徴とは、関心のあるものの個々の測定可能なプロパティです。特徴の強力なセットを見つけるには、実験とドメイン知識が必要です。 一部の特徴は、他の特徴よりもターゲットの予測に適しています。 また、一部の特徴量のは他の特徴と強い相関関係にあり、モデルに新しい情報を追加しない場合があります。 これらの特徴は削除できます。
<!-- In machine learning, features are individual measurable properties of something you're interested in. Finding a strong set of features requires experimentation and domain knowledge. Some features are better for predicting the target than others. Also, some features may have a strong correlation with other features, and won't add new information to the model. These features can be removed. -->

モデルを構築するには、利用可能なすべての特徴を使用するか、特徴のサブセットを選択できます。
<!-- To build a model, you can use all the features available, or select a subset of the features. -->


## 学習アルゴリズムを選択して適用する

**2 クラス ロジスティック回帰** モジュールを使用してモデルを作成し、トレーニング データセットでトレーニングします。
<!-- Create a model using the **Two-Class Logistic Regression** module and train it on the training dataset.  -->

**モデルのトレーニング** モジュールの結果は、予測を行うために新しいサンプルをスコアリングするために使用できるトレーニング済みの分類モデルです。 テスト セットを使用して、トレーニング済みモデルからスコアを生成します。 次に、**モデルの評価** モジュールを使用して、モデルの品質を分析および比較します。
<!-- The result of the **Train Model** module is a trained classification model that can be used to score new samples to make predictions. Use the test set to generate scores from the trained models. Then use the **Evaluate Model** module to analyze and compare the quality of the models. -->

パイプラインを実行した後、出力ポートをクリックして **Visualize** を選択すると、**Score Model** モジュールからの出力を表示できます。 出力には、スコア付けされたラベルとラベルの確率が含まれます。
<!-- After you run the pipeline, you can view the output from the **Score Model** module by clicking the output port and selecting **Visualize**. The output includes the scored labels and the probabilities for the labels. -->

最後に、結果の品質をテストするために、**Evaluate Model** モジュールをパイプライン キャンバスに追加し、左側の入力ポートを Score Model モジュールの出力に接続します。 パイプラインを実行し、出力ポートをクリックして **Visualize** を選択して、**Evaluate Model** モジュールの出力を表示します。
<!-- Finally, to test the quality of the results, add the **Evaluate Model** module to the pipeline canvas, and connect the left input port to the output of the Score Model module. Run the pipeline and view the output of the **Evaluate Model** module, by clicking the output port and selecting **Visualize**. -->


## 評価

ロジスティック回帰モデルの AUC は、テスト セットで 0.631 です。
<!-- The logistic regression model has AUC of 0.631 on the test set. -->

  ![評価する](media/r-script-flight-delay-prediction/evaluate.png)


## 次のステップ

デザイナーが利用できるその他のサンプルを調べます。
<!-- Explore the other samples available for the designer: -->

- [サンプル 1 - 回帰: 自動車の価格を予測する](regression-automobile-price-prediction-basic.md)
- [サンプル 2 - 回帰: 自動車価格予測のアルゴリズムを比較](regression-automobile-price-prediction-compare-algorithms.md)
- [サンプル 3 - 特徴選択による分類: 所得予測](binary-classification-feature-selection-income-prediction.md)
- [サンプル 4 - 分類: 信用リスクの予測 (コスト重視)](binary-classification-python-credit-prediction.md)
- [サンプル 5 - 分類: チャーンを予測する](binary-classification-customer-relationship-prediction.md)
- [サンプル 7 - テキスト分類: ウィキペディア SP 500 データセット](text-classification-wiki.md) -->


---

Original: https://github.com/Azure/MachineLearningDesigner/blob/master/articles/samples/r-script-flight-delay-prediction.md


<!-- # Build a classifier & use R to predict flight delays with Azure Machine Learning designer

**Designer sample 6**


This pipeline uses historical flight and weather data to predict if a scheduled passenger flight will be delayed by more than 15 minutes. This problem can be approached as a classification problem, predicting two classes: delayed, or on time.

Here's the final pipeline graph for this sample:

[![Graph of the pipeline](media/r-script-flight-delay-prediction/pipeline-graph.png)](media/r-script-flight-delay-prediction/pipeline-graph.png#lightbox)


## Data

This sample uses the **Flight Delays Data** dataset. It's part of the TranStats data collection from the U.S. Department of Transportation. The dataset contains flight delay information from April to October 2013. The dataset has been pre-processed as follows:

* Filtered to include the 70 busiest airports in the continental United States.
* Relabeled canceled flights as delayed by more than 15 mins.
* Filtered out diverted flights.
* Selected 14 columns.

To supplement the flight data, the **Weather Dataset** is used. The weather data contains hourly, land-based weather observations from NOAA, and represents observations from airport weather stations, covering the same time period as the flights dataset. It has been pre-processed as follows:

* Weather station IDs were mapped to corresponding airport IDs.
* Weather stations not associated with the 70 busiest airports were removed.
* The Date column was split into separate columns: Year, Month, and Day.
* Selected 26 columns.

## Pre-process the data

A dataset usually requires some pre-processing before it can be analyzed.

![data-process](./media/r-script-flight-delay-prediction/data-process.png)

### Flight data

The columns **Carrier**, **OriginAirportID**, and **DestAirportID** are saved as integers. However, they're  categorical attributes, use the **Edit Metadata** module to convert them to categorical.

![edit-metadata](./media/r-script-flight-delay-prediction/edit-metadata.png)

Then use the **Select Columns** in Dataset module to exclude from the dataset columns that are possible target leakers: **DepDelay**, **DepDel15**, **ArrDelay**, **Canceled**, **Year**. 

To join the flight records with the hourly weather records, use the scheduled departure time as one of the join keys. To do the join, the CSRDepTime column must be rounded down to the nearest hour, which is done by in the **Execute R Script** module. 

### Weather data

Columns that have a large proportion of missing values are excluded using the **Project Columns** module. These columns include all string-valued columns: **ValueForWindCharacter**, **WetBulbFarenheit**, **WetBulbCelsius**, **PressureTendency**, **PressureChange**, **SeaLevelPressure**, and **StationPressure**.

The **Clean Missing Data** module is then applied to the remaining columns to remove rows with missing data.

Weather observation times are rounded up to the nearest full hour. Scheduled flight times and the weather observation times are rounded in opposite directions to ensure the model uses only weather before the flight time. 

Since weather data is reported in local time, time zone differences are accounted for by subtracting the time zone columns from the scheduled departure time and the weather observation time. These operations are done using the **Execute R Script** module.

### Joining Datasets

Flight records are joined with weather data at origin of the flight (**OriginAirportID**) using the **Join Data** module.

 ![join flight and weather by origin](./media/r-script-flight-delay-prediction/join-origin.png)


Flight records are joined with weather data using the destination of the flight (**DestAirportID**).

 ![Join flight and weather by destination](./media/r-script-flight-delay-prediction/join-destination.png)

### Preparing Training and Test Samples

The **Split Data** module splits the data into April through September records for training, and October records for test.

 ![Split training and test data](./media/r-script-flight-delay-prediction/split.png)

Year, month, and timezone columns are removed from the training dataset using the Select Columns module.

## Define features

In machine learning, features are individual measurable properties of something you're interested in. Finding a strong set of features requires experimentation and domain knowledge. Some features are better for predicting the target than others. Also, some features may have a strong correlation with other features, and won't add new information to the model. These features can be removed.

To build a model, you can use all the features available, or select a subset of the features.

## Choose and apply a learning algorithm

Create a model using the **Two-Class Logistic Regression** module and train it on the training dataset. 

The result of the **Train Model** module is a trained classification model that can be used to score new samples to make predictions. Use the test set to generate scores from the trained models. Then use the **Evaluate Model** module to analyze and compare the quality of the models.
pipeline
After you run the pipeline, you can view the output from the **Score Model** module by clicking the output port and selecting **Visualize**. The output includes the scored labels and the probabilities for the labels.

Finally, to test the quality of the results, add the **Evaluate Model** module to the pipeline canvas, and connect the left input port to the output of the Score Model module. Run the pipeline and view the output of the **Evaluate Model** module, by clicking the output port and selecting **Visualize**.

## Evaluate
The logistic regression model has AUC of 0.631 on the test set.

 ![evaluate](media/r-script-flight-delay-prediction/evaluate.png)

## Next steps

Explore the other samples available for the designer:

- [Sample 1 - Regression: Predict an automobile's price](regression-automobile-price-prediction-basic.md)
- [Sample 2 - Regression: Compare algorithms for automobile price prediction](regression-automobile-price-prediction-compare-algorithms.md)
- [Sample 3 - Classification with feature selection: Income Prediction](binary-classification-feature-selection-income-prediction.md)
- [Sample 4 - Classification: Predict credit risk (cost sensitive)](binary-classification-python-credit-prediction.md)
- [Sample 5 - Classification: Predict churn](binary-classification-customer-relationship-prediction.md)
- [Sample 7 - Text Classification: Wikipedia SP 500 Dataset](text-classification-wiki.md) -->
