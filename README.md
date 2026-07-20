# GROMACS Recovery Agent (LangGraph + RFdiffusion)

PDBファイルを `gmx pdb2gmx` に通す際に発生するエラーを自動診断し、欠損残基の規模に応じて **RFdiffusion**（6残基以上の欠損）または **PDBfixer**（1〜5残基の欠損）で構造を修復したうえで、GROMACSの前処理（`pdb2gmx`）が成功するまで自動リトライする自律エージェントです。

制御フローは [LangGraph](https://github.com/langchain-ai/langgraph) の `StateGraph` で実装されており、マルチマー複合体における鎖の順序不一致にも耐えうる、構造生物学のドメイン知識を深く組み込んだ配列復元ロジックを搭載しています。

> **⚠️ RFdiffusionの出力仕様について**
> RFdiffusionは **バックボーン (N, Cα, C, O) のみを生成する構造生成モデル** であり、アミノ酸配列（側鎖）は設計しません。公式リポジトリのドキュメントで明記されている通り、新規に設計された残基は常に **ポリグリシン (GLY)** として出力されます。これはバグではなく仕様です — 側鎖の座標には損失関数が適用されておらず、そのまま信頼できるアミノ酸種として扱うことはできません。
> 配列（どのアミノ酸を配置すべきか）の決定は、Nature論文でも RFdiffusion とは別のステップ（論文では ProteinMPNN）として行われています。本エージェントではこれを `sequence_recovery.py` による独立した後処理ステップとして実装しています。

## 🔄 処理の流れ

```mermaid
graph TD
    Start(("開始")) --> Check{"欠損判定<br>missing_residues.py"}

    Check -->|6残基以上| RFDiff["① RFdiffusion実行<br>rfdiffusion_repair.py<br>（バックボーンのみ生成・新規残基はGLY）"]
    Check -->|1〜5残基| PDBFix["局所修復<br>repair.py (PDBFixer)"]
    Check -->|0残基| RunGmx["pdb2gmx実行<br>observation.py"]

    RFDiff --> Merge["構造マージ<br>rfdiffusion_repair.py<br>（GLYのまま複合体PDBへ統合）"]
    Merge --> SeqRec["② 配列復元<br>sequence_recovery.py<br>（FASTAとのアライメントでGLY→正しい残基名に置換）"]
    SeqRec --> FillAtoms["③ 側鎖原子補完<br>PDBFixer<br>（置換後の残基名に対応する側鎖を生成）"]

    FillAtoms --> RunGmx
    PDBFix --> RunGmx

    RunGmx -->|成功| Save(("完了"))
    RunGmx -->|失敗| Diag["エラー診断<br>diagnosis.py"]

    Diag -->|既知| Update["PDB更新<br>repair.py"]
    Diag -->|未知| Fail(("中断"))

    Update --> RunGmx

    style Start fill:#f9f9f9,stroke:#333
    style RunGmx fill:#e1f5fe,stroke:#01579b
    style RFDiff fill:#fff3e0,stroke:#ff6f00
    style Merge fill:#fff3e0,stroke:#ff6f00
    style PDBFix fill:#e8f5e9,stroke:#2e7d32
    style FillAtoms fill:#e8f5e9,stroke:#2e7d32
    style SeqRec fill:#f3e5f5,stroke:#7b1fa2
    style Save fill:#e8f5e9,stroke:#2e7d32
    style Fail fill:#ffebee,stroke:#c62828
```

> `graph.py` の `rfdiffusion_node` は上図の①〜③を1つのノード内で順番に呼び出します（`run_rfdiffusion()` → `apply_sequence_recovery()` → `_fill_missing_atoms()`）。RFdiffusion自体はあくまでバックボーン生成器であり、配列の正しさに責任を持たない点が本フローの要です。

---

### 🧠 理論的背景：堅牢な復元を支える2つのアルゴリズム
本エージェントが「PDBとFASTAで鎖の順番やIDが異なっていても正しく復元できる」理由は、以下の2つのアルゴリズムの概念を応用しているためです。
#### 1. MM-align 的思想によるグローバルな配列マッピング
引用論文: Mukherjee & Zhang, "MM-align: a quick algorithm for aligning multiple-chain protein complex structures using iterative dynamic programming", Nucleic Acids Research, 2009.
マルチチェーン複合体において、PDBの鎖IDとFASTAの鎖IDが一致しない場合でも、「複合体全体のアライメントスコアの総和」が最大となる1対1の鎖マッピングをハンガリー法で探索します。これにより、鎖の順序が入れ替わっていても、配列パターンに基づいて数学的に最適な対応関係を自動で見つけ出し、RFdiffusionによって新規生成された領域（GLYのまま）に入るべき正しいアミノ酸配列を、RFdiffusion実行後に特定します。

📊 MM-align のアルゴリズムフロー
```mermaid
graph TD
    A[入力: ターゲット複合体構造 T と モデル構造 M] --> B[初期化: 鎖長の一致や配列類似度に基づく初期の鎖マッピング π を仮定]
    B --> C{反復ループ開始: Iterative Dynamic Programming}
    C --> D[1. 残基レベルのアライメント]
    D --> |仮定された鎖マッピング π に基づき| E[2. 空間変換行列 R, T の計算]
    E --> |Kabsch algorithm等で最適重ね合わせ| F[3. 複合体全体の TM-score を計算]
    F --> G[4. 鎖間類似度行列の作成]
    G --> |全鎖ペア間の TM-score または距離を計算| H[5. 鎖マッピング π の更新]
    H --> |ハンガリアン法または貪欲法で、スコア和が最大になる割り当てを探索| I{収束判定}
    I --> |鎖マッピング π または TM-score が変化しなくなった / 最大反復回数に到達| J[出力: 最大 TM-score, 最適鎖マッピング π, 重ね合わせられた構造]
    I --> |まだ変化が見られる| C
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#ff9,stroke:#333,stroke-width:2px
```

#### 2. RFdiffusion: バックボーン専用の拡散生成モデル (Backbone-Only Generation)
引用論文: Watson et al., "De novo design of protein structure and function with RFdiffusion", Nature, 2023.
RFdiffusionは `contigmap.provide_seq` で「既知領域」の配列を条件として与えることはできますが、これは**構造をより安定にサンプリングするための条件付け**であり、**未知（欠損）領域の配列自体を生成する機能ではありません**。未知領域は一貫してGLYとして出力されます。
そのため本エージェントでは、RFdiffusionを「バックボーンの妥当な立体配座を提案するモデル」として使い切ったあと、`sequence_recovery.py` が別途RCSB FASTAとのアラインメントによって配列を推定し、GLYの残基名を正しいアミノ酸名に書き換えます。これにより、RFdiffusionに存在しない機能（配列生成）を誤って期待することなく、モデルの仕様に忠実な形で物理的に妥当性の高い修復を実現します。

📊 RFdiffusion の生成アルゴリズムフロー
```mermaid
graph TD
    A[開始: 純粋なガウスノイズ構造 x_T の生成] --> B{条件付け Conditioning の適用}
    B --> |Inpainting/部分固定の場合| C[既知の残基座標・配列を固定し、未知領域のみをノイズとして保持]
    B --> |De novo設計の場合| D[対称性制約 Symmetry や バインダー条件などを適用]
    C --> E[逆拡散ループ開始: t = T down to 1]
    D --> E
    E --> F[1. RoseTTAFoldネットワークへの入力]
    F --> |x_t と 時間ステップ t| G[2. ノイズの予測 ε_θ^trans, ε_θ^rot]
    G --> H{Classifier-Free Guidance CFG の適用}
    H --> |条件付き予測と無条件予測を線形結合| I[3. 誘導されたノイズ予測の計算]
    I --> J["4. 逆拡散ステップ: x_t-1 のサンプリング"]
    J --> |DDIM または DDPMスケジューラに基づく更新| K{投影・制約の適用 Projection}
    K --> |Inpaintingの場合| L[既知の残基座標を元の値に強制的に上書き]
    K --> |対称性設計の場合| M[対称操作を適用して構造を強制対称化]
    L --> N{t > 1 ?}
    M --> N
    N --> |Yes| E
    N --> |No| O["出力: ノイズが除去された最終バックボーン構造 x_0<br>（新規設計領域は側鎖未定義・常にGLY）"]
    O --> P["後処理①: sequence_recovery.py によるFASTAベース配列復元<br>（本エージェントの実装）"]
    P --> Q["後処理②: PDBFixerによる側鎖原子の補完"]
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style O fill:#bbf,stroke:#333,stroke-width:2px
    style L fill:#ff9,stroke:#333,stroke-width:2px
    style H fill:#9f9,stroke:#333,stroke-width:2px
    style P fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
```

> 📝 **注**: 元のRFdiffusion論文・公式実装では、生成されたバックボーンへの配列付与は ProteinMPNN が担当します。本エージェントはProteinMPNNの代わりに、対象タンパク質のRCSB FASTA配列が既知であるという前提のもとで、MM-align的なグローバル配列アラインメントによって欠損領域の配列を復元するアプローチを取っています。

---

## 🛠️ 1. 環境構築

### 1.1 前提条件
| ソフトウェア | 用途 | 備考 |
|---|---|---|
| Anaconda / Miniconda | 環境構築 | `conda` コマンドが使用可能であること |
| GROMACS (`gmx`) | 前処理（pdb2gmx）の実行 | `gmx` が PATH 上に存在すること |
| NVIDIA GPU (CUDA 12.8+) | RFdiffusion の推論 | CPU のみでは現実的な時間で完了しません |

### 1.2 GROMACS のインストール
conda環境とは別に、公式手順に従ってビルド・インストールし、`gmx` コマンドがPATHに通っていることを確認してください。
```bash
gmx --version
```

### 1.3 依存環境のセットアップ (conda)
本リポジトリ同梱の `environment.yml` を使用し、RFdiffusion および本エージェントに必要なパッケージを一括で構築します。

```bash
# 1. 環境の作成
conda env create -f environment.yml

# 2. 環境の有効化
conda activate gromacs_recovery_env # environment.yml で指定された名前を使用

# 3. RFdiffusion モデル重みのダウンロード (RFdiffusion リポジトリ内で行う場合)
# bash scripts/download_models.sh ./models
```

> **💡 注意点:**  
> 提供された `environment.yml` には `dgl==2.4.0+cu121` や `torch==2.7.1+cu128` など、最新かつ互換性の取れたバージョンが定義されています。手動でパッケージを入れる必要はありません。

---

## ⚙️ 2. 設定 (`config.yaml`)

```yaml
gromacs:
  force_field: "amber99sb-ildn"   # gmx pdb2gmx -ff
  water_model: "tip3p"            # gmx pdb2gmx -water

agent:
  max_attempts: 10                # pdb2gmx の最大試行回数
  log_dir: "log"                  # 実行ログ (recovery.log) と作業用一時ディレクトリの出力先
  keep_work_dir: false            # 作業用一時ディレクトリ (work_YYYYMMDD_HHMMSS) を残すか
  output_dir: "results"           # 修復成功後の最終 PDB の出力先

rfdiffusion:
  script_path: "/path/to/RFdiffusion/scripts/run_inference.py"   # 環境に合わせて絶対パスに変更
  model_directory_path: "/path/to/RFdiffusion/models"            # 環境に合わせて絶対パスに変更
  min_residues_for_rfdiffusion: 6   # この残基数以上の欠損は RFdiffusion へ、未満は PDBfixer へ
  num_designs: 1                    # RFdiffusion の生成数
  timeout_sec: 1800                 # RFdiffusion 実行のタイムアウト（秒）
  reassign_sequence_from_fasta: true # RFdiffusion実行後（新規領域がGLYの状態）に、FASTAとの
                                      # アラインメントで配列を復元する後処理ステップを実行するか。
                                      # RFdiffusion自体には配列を生成する機能がないため、この
                                      # フラグをfalseにすると新規領域はGLYのままpdb2gmxへ渡る。
  fasta_cache_dir: "log/fasta_cache" # FASTA 取得時のキャッシュディレクトリ
```

> `pdb_id` は `graph.py` の実行時 `state` に `pdb_id` として渡します（`config.yaml` 直下ではなく呼び出しコード側で指定、詳細は [3.2](#32-コードから直接利用する場合) 参照）。`reassign_sequence_from_fasta: true` でも `pdb_id` が未指定の場合、配列復元はスキップされ新規領域はGLYのまま扱われます。

---

## 🚀 3. 使い方

### 3.1 最小実行例
`main.py` は同一ディレクトリの `broken_test.pdb` を読み込んで修復を試みます。
```bash
# 修復したい PDB ファイルを配置
cp your_broken_structure.pdb broken_test.pdb

# エージェント実行
python main.py
```
成功すると `results/broken_test_final.pdb` に修復済み PDB が保存されます。すべての経過は `log/recovery.log` に記録されます。

### 3.2 コードから直接利用する場合
```python
import yaml
import os
from recovery_agent.graph import build_graph

# 設定の読み込み
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# グラフの構築
app = build_graph(config)

# 初期状態の定義
state = {
    "pdb_path": "your_structure.pdb",
    "pdb_id": "6XYZ",              # RFdiffusion後の配列復元でRCSB FASTAを取得するために使用
                                    # (未指定の場合、新規生成領域はGLYのままとなる)
    "work_dir": os.path.join("log", "work_custom"),
    "attempt": 0,
    "repair_history": [],
    "extra_flags": [],
}

# 実行
result = app.invoke(state, config={"recursion_limit": 100})

if result.get("success"):
    print(f"Success! Saved to: {result.get('pdb_path')}")
else:
    print(f"Failed: {result.get('status')}")
```

---

## 🔍 4. トラブルシューティング

本エージェントは、実データの PDB フォーマット特有の罠に対して以下の**堅牢な対策をコードレベルで実装済み**です。
- ✅ **挿入コード (Insertion Code) 対策**: `_parse_resnum` により `100A` のような残基IDでも数値部分を安全に抽出。
- ✅ **`.trb` パースエラー対策**: `con_hal_pdb_idx` の文字列リストを安全にスライスして分解。
- ✅ **HETATM 重複除去**: RFdiffusion 統合時、同一残基番号を持つ水分子等の HETATM を自動的に一掃。

それでも問題が発生した場合は以下を確認してください。

| 症状 | 対処 |
|---|---|
| `EnvironmentError: GROMACS ('gmx' command) is not found` | GROMACS をインストールし、`gmx` に PATH を通してください。 |
| `ModuleNotFoundError: No module named 'rfdiffusion'` | `environment.yml` で定義された環境が正しく activate されているか確認してください。 |
| `pdb2gmx` が毎回同じエラーで失敗し、`failed_no_candidates` になる | `log/recovery.log` を確認し、`diagnosis.py` の分類ルールに該当事象が定義されているか確認してください。 |
| RFdiffusion がタイムアウトする | `config.yaml` の `timeout_sec` を延長するか、GPU のメモリ不足 (`CUDA out of memory`) が発生していないか確認してください。 |

---

## 📂 5. ディレクトリ構成

```text
gromacs_recovery/
├── main.py                          # エントリーポイント
├── config.yaml                      # 設定ファイル
├── environment.yml                  # 依存環境定義 (Python 3.10 / PyTorch 2.7 / DGL 2.4)
├── recovery_agent/
│   ├── graph.py                     # LangGraph による修復フロー本体（①RFdiffusion→②配列復元→③側鎖補完を統括）
│   ├── missing_residues.py          # 欠損残基数のカウント（分岐閾値の判定自体は graph.py 側）
│   ├── rfdiffusion_repair.py        # RFdiffusion 呼び出し・構造マージ（バックボーンのみ生成、新規残基は常にGLY）
│   ├── sequence_recovery.py         # MM-align的思想に基づくグローバル配列復元（RFdiffusion実行後の独立ステップ）
│   ├── observation.py               # gmx pdb2gmx の実行と出力キャプチャ
│   ├── diagnosis.py                 # pdb2gmx のエラー分類
│   ├── repair.py                    # PDBfixer/Biopython による個別修復関数群
│   └── utils.py                     # タイムアウト付き関数実行
├── log/                             # 実行時に自動作成
└── results/                         # 修復成功後の最終 PDB 出力先
```
