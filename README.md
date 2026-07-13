# gromacs_recovery

`gmx pdb2gmx` がPDBファイルの構造的な問題(原子の欠損、未知の残基、水素の命名不一致など)で失敗したとき、
エラーメッセージを自動診断し、適切な修復処理を試したうえで再実行する自己修復エージェントです。

修復案を1つ試して失敗したら次の候補を試す、というループを最大試行回数まで繰り返し、
最終的に `gmx pdb2gmx` が成功するPDBファイルを生成することを目指します。

## 特徴

- `gmx pdb2gmx` の標準エラー出力を正規表現でパースし、エラーの種類を自動分類
- 分類ごとに用意された修復候補(PDBFixer / Biopythonベース)を順番に試行
- 1つの修復候補が効かなくても、履歴を見て次の候補にフォールバック
- 6残基以上のループが構造から丸ごと欠損している場合は、単純な幾何学的補間ではなく
  [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) によるループ再構築を優先的に試行
- 各試行の診断結果・修復内容・所要時間をJSON Lines形式でログ出力

## 動作要件

- Python 3.9+
- [GROMACS](https://www.gromacs.org/)(`gmx` コマンドがPATH上にあること)
- 以下のPythonパッケージ
  - `pyyaml`
  - `biopython`
  - `openmm`
  - `pdbfixer`
- (任意)ループ再構築を使う場合: [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) 本体一式
  (GPU環境・学習済み重みが別途必要です)

```bash
pip install pyyaml biopython openmm pdbfixer
```

## 使い方

1. `config.yaml` を必要に応じて編集する
2. 修復したいPDBファイルを用意する
3. `main.py` の `test_pdb` に対象ファイルのパスを指定して実行する

```bash
python main.py
```

成功すると、最終的に `gmx pdb2gmx` が通ったPDBファイルが `agent.output_dir`(デフォルト `results/`)に
`<元のファイル名>_final.pdb` として保存されます。

各試行の詳細ログは `agent.log_dir`(デフォルト `logs/`)に `<元のファイル名>_recovery.jsonl` として
1行1試行のJSON Lines形式で出力されます。

## ディレクトリ構成

```
gromacs_recovery/
├── main.py                      # エントリーポイント
├── config.yaml                  # 設定ファイル
├── recovery_agent/
│   ├── agent.py                 # 観測→診断→修復のメインループ(RecoveryAgent)
│   ├── observation.py           # gmx pdb2gmxの実行とエラー取得(ObservationModule)
│   ├── diagnosis.py             # 標準エラー出力からのエラー分類
│   ├── repair.py                # 各種修復処理の実装
│   └── utils.py                 # 別プロセスでのタイムアウト付き関数実行
└── tests/
    └── test_diagnosis.py
```

## 処理の流れ

```
┌─────────────┐   失敗    ┌──────────┐   ┌────────────┐   ┌──────────┐
│ gmx pdb2gmx │ ───────▶ │ 診断      │──▶│ 欠損残基数  │──▶│ 修復候補  │
│ を実行       │          │(diagnosis)│   │ チェック     │   │ を1つ実行 │
└─────────────┘          └──────────┘   └────────────┘   └──────────┘
      ▲                                  6残基以上の            │
      │                                  ループ欠損なら           │
      │                                  RFdiffusionを優先        │
      └──────────────────────────────────────────────────────────┘
         修復後のPDBで再度 gmx pdb2gmx を実行(最大 max_attempts 回)
```

1. **観測 (Observation)**: `gmx pdb2gmx` を実行し、成功/失敗と標準エラー出力を取得
2. **診断 (Diagnosis)**: 標準エラー出力からエラーの種類を分類
3. **欠損残基数チェック**: PDBFixerの `findMissingResidues()` でSEQRESとATOMを比較し、
   ループごと欠損している残基数を数える。閾値(デフォルト6残基)以上ならRFdiffusionでの
   再構築を優先的に選択する
4. **修復 (Repair)**: 診断カテゴリに対応する修復候補のうち、まだ試していないものを1つ実行
5. 修復に成功したら1に戻って再度 `gmx pdb2gmx` を実行。失敗・タイムアウトしても
   次の候補があれば継続し、候補を使い切ったら終了する

## 診断カテゴリと修復候補

| 診断カテゴリ | 意味 | 修復候補(順に試行) |
|---|---|---|
| `MISSING_HYDROGEN` | 水素原子の命名がrtpエントリと不一致 | `pdb2gmx_with_ignh_flag`(`-ignh`で無視)→ `pdbfixer_add_missing_atoms_and_hydrogens` |
| `MISSING_ATOM` | 既存残基内の重原子が不足 | `pdbfixer_add_missing_atoms`(6残基以上のループ欠損が別途検出された場合はRFdiffusionが優先) |
| `MISSING_RESIDUE_DB_ENTRY` | 力場のデータベースに無い残基名(糖鎖・補因子等) | `strip_unknown_residue` → `pdbfixer_replace_nonstandard_residues` → `strip_hetero_cofactors` |
| `HETERO_CHAIN_TYPE_MISMATCH` | HETATMの型が一貫していない | `strip_hetero_cofactors` |
| `CHAIN_SPLIT` | 同じチェインIDが非連続なブロックで使われている | `rename_duplicate_chain_ids` |
| `TERMINUS_ISSUE` | 末端(N末/C末)の扱いに問題がある | `pdb2gmx_with_explicit_ter_flag`(`-ter`) |
| `UNKNOWN` / `AMBIGUOUS(...)` | 未対応のエラー、または複数カテゴリに一致 | なし(修復候補が尽き次第 `failed_no_candidates` で終了) |

`-ignh` や `-ter` のようなフラグ系の修復は、以降の(フラグを返さない)修復処理が実行されても
上書きされず維持されます。

## RFdiffusionによるループ再構築

構造中に6残基以上のループが丸ごと欠損している場合、PDBFixerの単純な幾何学的補間では
立体構造が破綻しやすく、巨大な複合体では処理も非常に重くなります。そのため、
`count_missing_residues()` で検出した欠損残基数が閾値以上のときは、他の修復候補より
先にRFdiffusionでの再構築(`rfdiffusion_rebuild_missing_loops`)を試みます。

処理内容:

1. 最大の欠損ギャップからRFdiffusionのcontig文字列(例: `A1-54/6-6/A61-120`)を組み立てる
2. `run_inference.py` をサブプロセスとして実行し、ループ部分を新規に構造予測させる
3. 出力された骨格構造(backboneのみ)のうち、元の構造に無かった(=欠損していた)残基だけを
   抽出し、正しい順序で元のPDBにマージする(既存残基の実測座標はそのまま保持)
4. 側鎖原子・水素は後続の `pdbfixer_add_missing_atoms` 等のステップで付加される

RFdiffusionが未インストール・GPU無し等で失敗/タイムアウトした場合は、その旨を履歴に記録し、
次の試行で通常の修復候補(pdbfixer等)にフォールバックします。

### 設定

`config.yaml` の `rfdiffusion` セクションで調整できます(未設定時は環境変数にフォールバック)。

```yaml
rfdiffusion:
  missing_residue_threshold: 6      # この残基数以上でRFdiffusionを優先
  run_inference_script: null        # run_inference.pyへのパス
  timeout_sec: null                 # RFdiffusion本体のタイムアウト(秒)
```

| 設定項目 | 環境変数 | デフォルト |
|---|---|---|
| `run_inference_script` | `RFDIFFUSION_RUN_INFERENCE` | `/opt/RFdiffusion/scripts/run_inference.py` |
| `timeout_sec` | `RFDIFFUSION_TIMEOUT_SEC` | `1800`(30分) |

> **注意**: RFdiffusion本体(GPU・学習済み重み含む)は別途インストールが必要です。
> また、contig文字列の書式やマージ処理は、お使いのRFdiffusionのバージョン・出力形式で
> 実データを使って一度検証することを推奨します。

## config.yaml 設定項目

```yaml
gromacs:
  force_field: "amber99sb-ildn"   # gmx pdb2gmxに渡す力場
  water_model: "tip3p"            # gmx pdb2gmxに渡す水モデル
agent:
  max_attempts: 10                # 最大試行回数
  log_dir: "logs"                 # ログ出力先
  keep_work_dir: false            # 作業用一時ディレクトリを試行後も残すか
  output_dir: "results"           # 成功したPDBの保存先
  repair_timeout_sec: 300         # 各修復処理のタイムアウト(秒、RFdiffusion実行時は自動延長)
rfdiffusion:
  missing_residue_threshold: 6
  run_inference_script: null
  timeout_sec: null
```

## ログの形式

`logs/<ファイル名>_recovery.jsonl` に、試行ごとの状態が1行1JSONで記録されます。主なフィールド:

| フィールド | 内容 |
|---|---|
| `attempt` | 試行回数(0始まり) |
| `current_pdb` | その試行で使われたPDBファイルのパス |
| `success` | `gmx pdb2gmx` が成功したか |
| `diagnosis_category` | 診断されたエラーカテゴリ |
| `missing_residue_count` | 検出された欠損残基数 |
| `selected_repair` | 選択された修復関数名 |
| `repair_extra_flags` | `gmx pdb2gmx` に追加される累積フラグ(`-ignh`等) |
| `structure_altered` | その修復が構造そのものを変更したか(残基除去等) |
| `status` | `repaired_and_continuing` / `success` / `repair_timeout` / `repair_error` / `failed_no_candidates` / `error_repair_execution_failed` / `max_attempts_exceeded` のいずれか |
| `duration_sec` | その試行にかかった時間 |

## テスト

```bash
python -m pytest tests/
```

## ライセンス

[LICENSE](./LICENSE) を参照してください。
