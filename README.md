# SUPIR-GUI

SUPIR (Scaling-UP Image Restoration) モデルを使った画像超解像デスクトップアプリケーションです。

低解像度の画像を AI の力で 2x / 3x / 4x に高画質化できます。面倒な Python 環境の構築は不要で、初回起動時にすべて自動でセットアップされます。

## 主な機能

- **ワンクリック超解像** - 画像を選んでボタンを押すだけ
- **一括処理** - フォルダ内の画像をまとめて処理
- **拡大率選択** - 2x / 3x / 4x から選択可能
- **複数出力形式** - PNG / JPEG / WebP / HEIC に対応
- **GPU 自動検出** - NVIDIA (CUDA) / AMD・Intel (DirectML) / CPU を自動判別
- **自動セットアップ** - 初回起動時に Python 環境・依存ライブラリ・モデルチェックポイントを自動取得

## スクリーンショット

<!-- TODO: スクリーンショットを追加 -->

## 動作要件

- **OS**: Windows 10 / 11 (64bit)
- **ランタイム**: .NET 10
- **メモリ**: 16 GB 以上推奨
- **ストレージ**: 約 30 GB の空き容量（モデル・Python 環境含む）
- **GPU** (推奨): NVIDIA GPU (CUDA) / AMD・Intel GPU (DirectML)
  - GPU がない場合は CPU モードで動作（低速）

## インストール

[Releases](../../releases) ページからインストーラをダウンロードして実行してください。

Velopack によるセルフアップデートに対応しています。

## ビルド方法

### 前提条件

- [.NET 10 SDK](https://dotnet.microsoft.com/download)
- [Git](https://git-scm.com/)

### ビルド手順

```bash
git clone https://github.com/<your-org>/SUPIR-GUI.git
cd SUPIR-GUI

# アイコン生成 (PowerShell)
.\icon\generate_icon.ps1

# ビルド
dotnet build -c Release
```

## アーキテクチャ

```
SUPIR-GUI
├── Views/              # Avalonia UI ビュー (XAML)
├── ViewModels/         # MVVM ViewModel (CommunityToolkit.Mvvm)
├── Models/             # データモデル・設定
├── Services/
│   ├── GpuDetectionService     # WMI による GPU 自動検出
│   ├── PythonBootstrapService  # 埋め込み Python 環境の自動構築
│   └── SupirWrapperService     # Python ワーカーとの JSON IPC
├── Controls/           # カスタム UserControl
├── Converters/         # 値コンバータ
├── python/
│   └── supir_worker.py # SUPIR 推論ワーカー (stdin/stdout JSON IPC)
└── icon/               # アプリアイコン・生成スクリプト
```

### 処理フロー

1. **GPU 検出** - WMI で搭載 GPU を識別し、最適なデバイス (CUDA / DirectML / CPU) を選択
2. **Python 環境構築** - 埋め込み Python をダウンロードし、PyTorch・SUPIR 依存パッケージを自動インストール
3. **モデル取得** - Hugging Face Hub から SDXL ベースモデルと SUPIR チェックポイントをダウンロード
4. **ワーカー起動** - Python サブプロセスを JSON IPC で起動し、ping で接続確認
5. **画像処理** - GUI からのリクエストを JSON でワーカーに送信し、SUPIR による超解像処理を実行

### 技術スタック

| レイヤ | 技術 |
|--------|------|
| UI フレームワーク | [Avalonia UI](https://avaloniaui.net/) 11.x |
| MVVM | [CommunityToolkit.Mvvm](https://learn.microsoft.com/ja-jp/dotnet/communitytoolkit/mvvm/) |
| ランタイム | .NET 10 (Windows) |
| AI モデル | [SUPIR](https://github.com/Fanghua-Yu/SUPIR) (Scaling-UP Image Restoration) |
| 推論エンジン | PyTorch (CUDA / DirectML / CPU) |
| IPC | JSON over stdin/stdout |
| インストーラ | [Velopack](https://velopack.io/) |
| ロギング | [NLog](https://nlog-project.org/) |

## ライセンス

<!-- TODO: ライセンスを指定 -->
