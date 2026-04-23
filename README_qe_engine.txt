# QE Engine

日本株アノマリー分析エンジン

## 機能
- 日経平均データ自動取得
- スコアリング（季節性・曜日・イベントなど）
- 売買シグナル生成
- バックテスト
- 未来予測（60営業日）

## 使い方
```bash
python qe_engine.py --min-score 7 --future-days 60