# Azure Machine Learning デザイナーを使用して、会社のカテゴリを予測する分類器を構築する
<!-- # Build a classifier to predict company category using Azure Machine Learning designer. -->

**デザイナーサンプル7**


このサンプルでは、テキスト分析モジュールを使用して、Azure Machine Learning デザイナーでテキスト分類パイプラインを構築する方法を示します。
<!-- This sample demonstrates how to use text analytics modules to build a text classification pipeline in Azure Machine Learning designer. -->

テキスト分類の目的は、テキストの一部を 1 つ以上の事前定義されたクラスまたはカテゴリに割り当てることです。 テキストの断片には、ドキュメント、ニュース記事、検索クエリ、電子メール、ツイート、サポート チケット、顧客からのフィードバック、ユーザーの製品レビューなどがあります。テキスト分類のアプリケーションには、新聞記事やニュース ワイヤ (オンラインでニュースを送受信するシステム) のコンテンツをトピックに分類すること、Web ページを階層的に編成することが含まれます。 カテゴリ、スパム メールのフィルタリング、センチメント分析、検索クエリからのユーザーの意図の予測、サポート チケットのルーティング、顧客フィードバックの分析などです。
<!-- The goal of text classification is to assign some piece of text to one or more predefined classes or categories. The piece of text could be a document, news article, search query, email, tweet, support tickets, customer feedback, user product review etc. Applications of text classification include categorizing newspaper articles and news wire contents into topics, organizing web pages into hierarchical categories, filtering spam email, sentiment analysis, predicting user intent from search queries, routing support tickets, and analyzing customer feedback.  -->

このパイプラインは、**多クラス ロジスティック回帰分類器**をトレーニングして、**Wikipedia から抽出した Wikipedia SP 500 データセット**を使用して会社のカテゴリを予測します。
<!-- This pipeline trains a **multiclass logistic regression classifier** to predict the company category with **Wikipedia SP 500 dataset derived from Wikipedia**.   -->

テキスト データを使用した機械学習モデルのトレーニングの基本的な手順は次のとおりです。
<!-- The fundamental steps of a training machine learning model with text data are: -->

1. データを取得する<!-- 1. Get the data -->
1. テキストデータの前処理<!-- 1. Pre-process the text data -->
1. 特徴量エンジニアリング<!-- 1. Feature Engineering -->
   - 特徴ハッシュなどの特徴抽出モジュールを使用してテキスト特徴を数値特徴に変換し、テキスト データから n-gram 特徴を抽出します。<!-- Convert text feature into the numerical feature with feature extracting module such as feature hashing, extract n-gram feature from the text data. -->
2. モデルをトレーニングする<!-- 1. Train the model -->
3. スコアデータセット<!-- 1. Score dataset -->
4. モデルを評価する<!-- 1. Evaluate the model -->

これが、これから取り組むパイプラインの最終的な完成グラフです。 同様の決定を自分で行うことができるように、すべてのモジュールの理論的根拠を提供します。
<!-- Here's the final, completed graph of the pipeline we'll be working on. We'll provide the rationale for all the modules so you can make similar decisions on your own. -->

[![パイプラインのグラフ](./media/text-classification-wiki/nlp-modules-overall.png)](./media/text-classification-wiki/nlp-modules-overall.png#lightbox)


## データ

このパイプラインでは、**Wikipedia SP 500** データセットを使用します。 データセットは、S&P 500 の各企業に関する記載項目に基づいて Wikipedia (https://www.wikipedia.org/) から取得されます。 Azure Machine Learning デザイナーにアップロードする前に、データセットは次のように処理されました。
<!-- In this pipeline, we use the **Wikipedia SP 500** dataset. The dataset is derived from Wikipedia (https://www.wikipedia.org/) based on articles of each S&P 500 company. Before uploading to Azure Machine Learning designer, the dataset was processed as follows: -->

- 特定の企業ごとにテキスト コンテンツを抽出する<!-- - Extract text content for each specific company -->
- Wiki のフォーマッティングを削除<!-- - Remove wiki formatting -->
- 英数字以外の文字を削除<!-- - Remove non-alphanumeric characters -->
- すべてのテキストを小文字に変換します<!-- - Convert all text to lowercase -->
- 既知の企業カテゴリが追加されました<!-- - Known company categories were added -->

一部の企業では記事が見つからなかったため、レコードの数は 500 未満です。
<!-- Articles could not be found for some companies, so the number of records is less than 500. -->


## テキストデータの前処理

**Preprocess Text** モジュールを使用して、文の検出、文のトークン化など、テキスト データを前処理します。 [**Preprocess Text**](algorithm-module-reference/preprocess-text.md) の記事で、サポートされているすべてのオプションを見つけることができます。
<!-- We use the **Preprocess Text** module to preprocess the text data, including detect the sentences, tokenize sentences and so on. You would found all supported options in the [**Preprocess Text**](algorithm-module-reference/preprocess-text.md) article.  -->
テキスト データを前処理した後、**Split Data** モジュールを使用して入力データをランダムに分割し、トレーニング データセットに元のデータの 50% が含まれ、テスト データセットに元のデータの 50% が含まれるようにします。
<!-- After pre-processing text data, we use the **Split Data** module to randomly divide the input data so that the training dataset contains 50% of the original data and the testing dataset contains 50% of the original data. -->

## 特徴量エンジニアリング

このサンプルでは、特徴量エンジニアリングを実行する 2 つの方法を使用します。
<!-- In this sample, we will use two methods performing feature engineering. -->

### 特徴ハッシング

[**Feature Hashing**](algorithm-module-reference/feature-hashing.md) モジュールを使用して、記事のプレーン テキストを整数に変換し、整数値をモデルへの入力特徴として使用しました。
<!-- We used the [**Feature Hashing**](algorithm-module-reference/feature-hashing.md) module to convert the plain text of the articles to integers and used the integer values as input features to the model.  -->

**Feature Hashing** モジュールを使用すると、Vowpal Wabbit ライブラリによって提供される 32 ビットの murmurhash v3 ハッシュ化 メソッドを使用して、可変長テキスト ドキュメントを等しい長さの数値特徴ベクトルに変換できます。 特徴ハッシングを使用する目的は、次元削減です。 また、特徴ハッシングは、文字列比較の代わりにハッシュ値比較を使用するため、分類時の特徴重みの検索を高速化します。
<!-- The **Feature Hashing** module can be used to convert variable-length text documents to equal-length numeric feature vectors, using the 32-bit murmurhash v3 hashing method provided by the Vowpal Wabbit library. The objective of using feature hashing is dimensionality reduction; also feature hashing makes the lookup of feature weights faster at classification time because it uses hash value comparison instead of string comparison. -->

サンプル パイプラインでは、ハッシュ ビットの数を 14 に設定し、n グラムの数を 2 に設定します。これらの設定により、ハッシュ テーブルは 2^14 エントリを保持でき、ハッシュ化したそれぞれの特徴は 1 つ以上の n-gram 特徴を表します。 そしてその値は、テキスト インスタンスにおける n-gram の出現頻度を表します。 多くの問題では、このサイズのハッシュ テーブルで十分ですが、場合によっては、衝突を避けるために、より多くのスペースが必要になることがあります。 さまざまなビット数を使用して、機械学習ソリューションのパフォーマンスを評価します。
<!-- In the sample pipeline, we set the number of hashing bits to 14 and set the number of n-grams to 2. With these settings, the hash table can hold 2^14 entries, in which each hashing feature represents one or more n-gram features and its value represents the occurrence frequency of that n-gram in the text instance. For many problems, a hash table of this size is more than adequate, but in some cases, more space might be needed to avoid collisions. Evaluate the performance of your machine learning solution using different number of bits.  -->

### テキストから N-gram 特徴を抽出する

n-gram は、指定された一連のテキストからの n 個の用語の連続したシーケンスです。 サイズ 1 の n-gram はユニグラムと呼ばれます。 サイズ 2 の n-gram はバイグラムです。 サイズ 3 の n-gram はトライグラムです。 より大きなサイズの N グラムは、n の値によって参照されることがあります。たとえば、「4 グラム」、「5 グラム」などです。
<!-- An n-gram is a contiguous sequence of n terms from a given sequence of text. An n-gram of size 1 is referred to as a unigram; an n-gram of size 2 is a bigram; an n-gram of size 3 is a trigram. N-grams of larger sizes are sometimes referred to by the value of n, for instance, "four-gram", "five-gram", and so on. -->

特徴量エンジニアリングの別のソリューションとして、[**Extract N-Gram Feature from Text**](algorithm-module-reference/extract-n-gram-features-from-text.md) モジュールを使用しました。 このモジュールは、最初に n-gram のセットを抽出し、n-gram に加えて、各 n-gram がテキスト内に出現するドキュメントの数をカウントします (DF)。 このサンプルでは、TF-IDF メトリクスを使用して特徴値を計算します。 次に、非構造化テキスト データを等しい長さの数値特徴ベクトルに変換します。各特徴は、テキスト インスタンス内の n-gram の TF-IDF を表します。
<!-- We used [**Extract N-Gram Feature from Text**](algorithm-module-reference/extract-n-gram-features-from-text.md) module as another solution for feature engineering. This module first extracts the set of n-grams, in addition to the n-grams, the number of documents where each n-gram appears in the text is counted(DF). In this sample, TF-IDF metric is used to calculate feature values. Then, it converts unstructured text data into equal-length numeric feature vectors where each feature represents the TF-IDF of an n-gram in a text instance. -->

テキスト データを数値の特徴ベクトルに変換した後、**Select Column** モジュールを使用して、データセットからテキスト データを削除します。
<!-- After converting text data into numeric feature vectors, A **Select Column** module is used to remove the text data from the dataset.  -->


## モデルをトレーニングする

アルゴリズムの選択は、多くの場合、ユース ケースの要件によって異なります。
<!-- Your choice of algorithm often depends on the requirements of the use case.  -->
このパイプラインの目的は会社のカテゴリを予測することであるため、多クラス分類モデルは適切な選択です。 特徴の数が多く、これらの特徴がスパース(疎, 希薄, まばら)であることを考慮して、このパイプラインには **Multiclass Logistic Regression** モデルを使用します。
<!-- Because the goal of this pipeline is to predict the category of company, a multi-class classifier model is a good choice. Considering that the number of features is large and these features are sparse, we use **Multiclass Logistic Regression** model for this pipeline. -->

## テスト、評価、比較

データセットを分割し、異なるデータセットを使用してモデルをトレーニングおよびテストし、モデルの評価に客観性を持たせます。
<!-- We split the dataset and use different datasets to train and test the model to make the evaluation of the model more objective. -->

モデルがトレーニングされた後、**Score Model** および **Evaluate Model** モジュールを使用して、予測結果を生成し、モデルを評価します。 ただし、**Score Model** モジュールを使用する前に、トレーニング中に行ったのと同じように機能エンジニアリングを実行する必要があります。
<!-- After the model is trained, we would use the **Score Model** and **Evaluate Model** modules to generate predicted results and evaluate the models. However, before using the **Score Model** module, performing feature engineering as what we have done during training is required.  -->

**Feature Hashing** モジュールの場合、トレーニング フローとして、スコアリング フローに関する特徴量エンジニアリングを簡単に実行できます。 **Feature Hashing** モジュールを直接使用して、入力テキスト データを処理します。
<!-- For **Feature Hashing** module, it is easy to perform feature engineer on scoring flow as training flow. Use **Feature Hashing** module directly to process the input text data. -->

**Extract N-Gram Feature from Text** モジュールの場合、トレーニング データフローの **Result Vocabulary output** をスコアリング データフローの **Input Vocabulary** に接続します。そして、**Vocabulary mode** パラメータを **ReadOnly** に設定します。
<!-- For **Extract N-Gram Feature from Text** module, we would connect the **Result Vocabulary output** from the training dataflow to the **Input Vocabulary** on the scoring dataflow, and set the **Vocabulary mode** parameter to **ReadOnly**. -->
![n-gramスコアのグラフ](./media/text-classification-wiki/n-gram.png)

エンジニアリング ステップが完了したら、**スコア モデル**を使用して、トレーニング済みモデルを使用してテスト データセットの予測を生成できます。 結果を確認するには、**Score Model** の出力ポートを選択し、**Visualize** を選択します。
<!-- After finishing the engineering step, **Score Model** could be used to generate predictions for the test dataset by using the trained model. To check the result, select the output port of **Score Model** and then select **Visualize**. -->

次に、スコアを **Evaluate Model** モジュールに渡して、評価指標を生成します。 **モデルの評価**には 2 つの入力ポートがあるため、さまざまな方法で生成されたスコア付けされたデータセットを評価および比較できます。 このサンプルでは、特徴ハッシング方式と n-gram 方式、それぞれで生成された結果のパフォーマンスを比較します。
<!-- We then pass the scores to the **Evaluate Model** module to generate evaluation metrics. **Evaluate Model** has two input ports, so that we could evaluate and compare scored datasets that are generated with different methods. In this sample, we compare the performance of the result generated with feature hashing method and n-gram method. -->
結果を確認するには、**Evaluate Model** の出力ポートを選択し、**Visualize** を選択します。
<!-- To check the result, select the output port of the **Evaluate Model** and then select **Visualize**. -->

### 推論パイプラインを構築してリアルタイム エンドポイントをデプロイする

上記のトレーニング パイプラインを正常に送信すると、赤い矩形で囲まれたモジュールの出力をデータセットとして登録できます。
<!-- After submitting the training pipeline above successfully, you can register the output of the circled module as dataset. -->

![出力語彙の登録データセット1](./media/text-classification-wiki/extract-n-gram-training-pipeline-score-model.png)

データセットを登録するには、**Extract N-Gram Feature from Text** モジュールを (ダブルクリックで) 選択し、右側のペインで **Outputs+logs** (出力とログ) タブに切り替える必要があります。 **Register dataset** (データの登録) をクリックし、ポップアップ ウィンドウに入力します。
<!-- To register dataset, you need to select **Extract N-Gram Feature from Text** module and switch to **Outputs+logs** tab in the right pane. Click on **Register dataset** and fill in the pop-up window. -->

![出力語彙の登録データセット2](./media/text-classification-wiki/extract-n-gram-output-voc-register-dataset.png)

データセットをトレーニング パイプラインに正常に登録したら、リアルタイム推論パイプラインを作成できます。 次のグラフに合わせて、推論パイプラインを手動で調整する必要があります。
<!-- After register dataset successfully in the training pipeline, you can create real-time inference pipeline. You need to adjust your inference pipeline manually to the following graph: -->

![推論パイプライン](./media/text-classification-wiki/extract-n-gram-inference-pipeline.png)

次に、推論パイプラインを送信し、リアルタイム エンドポイントをデプロイします。
<!-- Then submit the inference pipeline, and deploy a real-time endpoint. -->

## 次のステップ

デザイナーが利用できるその他のサンプルを調べます。
<!-- Explore the other samples available for the designer: -->

- [サンプル 1 - 回帰: 自動車の価格を予測する](regression-automobile-price-prediction-basic.md)
- [サンプル 2 - 回帰: 自動車価格予測のアルゴリズムを比較](regression-automobile-price-prediction-compare-algorithms.md)
- [サンプル 3 - 特徴選択による分類: 所得予測](binary-classification-feature-selection-income-prediction.md)
- [サンプル 4 - 分類: 信用リスクの予測 (コスト重視)](binary-classification-python-credit-prediction.md)
- [サンプル 5 - 分類: チャーンを予測する](binary-classification-customer-relationship-prediction.md)
- [サンプル 6 - 分類: フライトの遅延を予測する](r-script-flight-delay-prediction.md) -->


---


Original: https://github.com/Azure/MachineLearningDesigner/blob/master/articles/samples/text-classification-wiki.md


<!-- # Build a classifier to predict company category using Azure Machine Learning designer.

**Designer sample 7**


This sample demonstrates how to use text analytics modules to build a text classification pipeline in Azure Machine Learning designer.

The goal of text classification is to assign some piece of text to one or more predefined classes or categories. The piece of text could be a document, news article, search query, email, tweet, support tickets, customer feedback, user product review etc. Applications of text classification include categorizing newspaper articles and news wire contents into topics, organizing web pages into hierarchical categories, filtering spam email, sentiment analysis, predicting user intent from search queries, routing support tickets, and analyzing customer feedback. 

This pipeline trains a **multiclass logistic regression classifier** to predict the company category with **Wikipedia SP 500 dataset derived from Wikipedia**.  

The fundamental steps of a training machine learning model with text data are:

1. Get the data

2. Pre-process the text data

3. Feature Engineering

   Convert text feature into the numerical feature with feature extracting module such as feature hashing, extract n-gram feature from the text data.

4. Train the model

5. Score dataset

6. Evaluate the model

Here's the final, completed graph of the pipeline we'll be working on. We'll provide the rationale for all the modules so you can make similar decisions on your own.

[![Graph of the pipeline](./media/text-classification-wiki/nlp-modules-overall.png)](./media/text-classification-wiki/nlp-modules-overall.png#lightbox)

## Data

In this pipeline, we use the **Wikipedia SP 500** dataset. The dataset is derived from Wikipedia (https://www.wikipedia.org/) based on articles of each S&P 500 company. Before uploading to Azure Machine Learning designer, the dataset was processed as follows:

- Extract text content for each specific company
- Remove wiki formatting
- Remove non-alphanumeric characters
- Convert all text to lowercase
- Known company categories were added

Articles could not be found for some companies, so the number of records is less than 500.

## Pre-process the text data

We use the **Preprocess Text** module to preprocess the text data, including detect the sentences, tokenize sentences and so on. You would found all supported options in the [**Preprocess Text**](algorithm-module-reference/preprocess-text.md) article. 
After pre-processing text data, we use the **Split Data** module to randomly divide the input data so that the training dataset contains 50% of the original data and the testing dataset contains 50% of the original data.

## Feature Engineering
In this sample, we will use two methods performing feature engineering.

### Feature Hashing
We used the [**Feature Hashing**](algorithm-module-reference/feature-hashing.md) module to convert the plain text of the articles to integers and used the integer values as input features to the model. 

The **Feature Hashing** module can be used to convert variable-length text documents to equal-length numeric feature vectors, using the 32-bit murmurhash v3 hashing method provided by the Vowpal Wabbit library. The objective of using feature hashing is dimensionality reduction; also feature hashing makes the lookup of feature weights faster at classification time because it uses hash value comparison instead of string comparison.

In the sample pipeline, we set the number of hashing bits to 14 and set the number of n-grams to 2. With these settings, the hash table can hold 2^14 entries, in which each hashing feature represents one or more n-gram features and its value represents the occurrence frequency of that n-gram in the text instance. For many problems, a hash table of this size is more than adequate, but in some cases, more space might be needed to avoid collisions. Evaluate the performance of your machine learning solution using different number of bits. 

### Extract N-Gram Feature from Text

An n-gram is a contiguous sequence of n terms from a given sequence of text. An n-gram of size 1 is referred to as a unigram; an n-gram of size 2 is a bigram; an n-gram of size 3 is a trigram. N-grams of larger sizes are sometimes referred to by the value of n, for instance, "four-gram", "five-gram", and so on.

We used [**Extract N-Gram Feature from Text**](algorithm-module-reference/extract-n-gram-features-from-text.md) module as another solution for feature engineering. This module first extracts the set of n-grams, in addition to the n-grams, the number of documents where each n-gram appears in the text is counted(DF). In this sample, TF-IDF metric is used to calculate feature values. Then, it converts unstructured text data into equal-length numeric feature vectors where each feature represents the TF-IDF of an n-gram in a text instance.

After converting text data into numeric feature vectors, A **Select Column** module is used to remove the text data from the dataset. 

## Train the model

Your choice of algorithm often depends on the requirements of the use case. 
Because the goal of this pipeline is to predict the category of company, a multi-class classifier model is a good choice. Considering that the number of features is large and these features are sparse, we use **Multiclass Logistic Regression** model for this pipeline.

## Test, evaluate, and compare

 We split the dataset and use different datasets to train and test the model to make the evaluation of the model more objective.

After the model is trained, we would use the **Score Model** and **Evaluate Model** modules to generate predicted results and evaluate the models. However, before using the **Score Model** module, performing feature engineering as what we have done during training is required. 

For **Feature Hashing** module, it is easy to perform feature engineer on scoring flow as training flow. Use **Feature Hashing** module directly to process the input text data.

For **Extract N-Gram Feature from Text** module, we would connect the **Result Vocabulary output** from the training dataflow to the **Input Vocabulary** on the scoring dataflow, and set the **Vocabulary mode** parameter to **ReadOnly**.
![Graph of n-gram score](./media/text-classification-wiki/n-gram.png)

After finishing the engineering step, **Score Model** could be used to generate predictions for the test dataset by using the trained model. To check the result, select the output port of **Score Model** and then select **Visualize**.

We then pass the scores to the **Evaluate Model** module to generate evaluation metrics. **Evaluate Model** has two input ports, so that we could evaluate and compare scored datasets that are generated with different methods. In this sample, we compare the performance of the result generated with feature hashing method and n-gram method.
To check the result, select the output port of the **Evaluate Model** and then select **Visualize**.

### Build inference pipeline to deploy a real-time endpoint

After submitting the training pipeline above successfully, you can register the output of the circled module as dataset.

![register dataset of output vocabulary1](./media/text-classification-wiki/extract-n-gram-training-pipeline-score-model.png)

To register dataset, you need to select **Extract N-Gram Feature from Text** module and switch to **Outputs+logs** tab in the right pane. Click on **Register dataset** and fill in the pop-up window.

![register dataset of output vocabulary2](./media/text-classification-wiki/extract-n-gram-output-voc-register-dataset.png)

After register dataset successfully in the training pipeline, you can create real-time inference pipeline. You need to adjust your inference pipeline manually to the following graph:

![inference pipeline](./media/text-classification-wiki/extract-n-gram-inference-pipeline.png)

Then submit the inference pipeline, and deploy a real-time endpoint.

## Next steps

Explore the other samples available for the designer:

- [Sample 1 - Regression: Predict an automobile's price](regression-automobile-price-prediction-basic.md)
- [Sample 2 - Regression: Compare algorithms for automobile price prediction](regression-automobile-price-prediction-compare-algorithms.md)
- [Sample 3 - Classification with feature selection: Income Prediction](binary-classification-feature-selection-income-prediction.md)
- [Sample 4 - Classification: Predict credit risk (cost sensitive)](binary-classification-python-credit-prediction.md)
- [Sample 5 - Classification: Predict churn](binary-classification-customer-relationship-prediction.md)
- [Sample 6 - Classification: Predict flight delays](r-script-flight-delay-prediction.md) -->