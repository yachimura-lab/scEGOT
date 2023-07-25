# scEGOT

single cell trajectory inference framework based on Entropic Gaussian mixture Optimal Transport

## setup

開発環境では poetry の仮想環境を用いてパッケージの管理をしています。
poetry のインストールについての公式ドキュメントは[こちら](https://python-poetry.org/docs/)です。
poetry をインストール後に、poetry.toml があるディレクトリまたはそれ以下のディレクトリで`$ poetry shell`を実行すると仮想環境が activate されます。仮想環境を抜けたいときは、`$ exit`を実行します。
`$ poetry install`を実行すると scegot の動作に必要なパッケージが全てインストールされます。

## dependnecies

- Graphviz
  GRN の描画に Graphviz を用いています。これは poetry ではインストールできないため、手動でインストールする必要があります。Graphviz のインストールについての公式ドキュメントは[こちら](https://graphviz.org/download/)です。
