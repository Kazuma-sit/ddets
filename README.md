# ddets
# Emotion-based Video Highlight Generation System

## 概要
動画内の人物の表情から感情を分析し、盛り上がり（ハイライト）シーンを自動抽出・生成するシステムです。（未完成）
大学、大学院の研究活動を通して開発に取り組んでいます。

## 使用技術
* **Language**: Python 3.x
* **ML/Analysis**: scikit-learn (KMeans, StandardScaler), Pandas, NumPy
* **Computer Vision**: OpenCV, dlib, hsemotion
* **Data Processing**: yt_dlp, pydub, chat_downloader

## 機能
1.  **感情分析**: 動画フレームから顔を検出し、8種類の感情スコアを時系列で算出。
2.  **クラスタリング**: 教師なし学習（KMeans）を用いて感情の傾向を分析。
3.  **ハイライト生成**: 分析結果に基づき、動画の重要シーンを自動で切り抜き・結合。
4.  **マルチモーダル分析**: 音声（音圧）やYouTubeチャットの盛り上がりも補助指標として活用。

## 環境
* Ubuntu (LXD container)
* Anaconda
