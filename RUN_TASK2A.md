# Task2A 再試行

最新のDQN優先モード試験：`diagnostics/dqn_first_fastcloud/summary.json`。
開始付近 (2.512, 1.913) から (2.687, 2.461) まで進み、スコア0→30、Room2検出1、
走行中の接触通知なし。更新点群が進路を塞いだため停止した。内部ゴールには未到達。
30点の採点内訳と、リセットからの再現性は未確認。以下の0点の記録は以前の試験。

2026-10-16現在、途中位置から内部ゴール (2.0, 3.5) の12 cm以内への到達はGazeboで確認済み。
最終の継続実行は8ステップ、終了距離0.1185 m。公式スコアは0で、Room2検出は1。
これは公式Task2A成功の確認ではない。接触・すり抜けの疑いと無加点の原因は未解決。
途中で複数の修正・停止を挟んだため、初期位置からの通し試験も未完了。
`diagnostics/precise_completion/summary.json` の旧 `success` は内部到達だけを指す。
新しい実行では `internal_goal_reached` と `official_success_verified` を区別して保存する。

## 初期化

Gazeboと採点ノードを普段の起動手順で再起動する。Gazeboの位置だけのリセットでは
採点ノードの履歴が初期化されるとは限らない。採点状態を直接書き換えない。
既存の `wrs_main`、`task2a_cloud_step`、`task2a_precise_run` を同時に動かさない。
従来の `start_all.sh` / `task2a_dqn.launch` は別の制御を起動するため併用しない。
この版は物体検出ラベルを使わないため、DQN走行にFRCNNは不要。

各ターミナルでROS環境を読み込む（学習用 `.venv` は使わない）。

```bash
source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash
cd /workspace/task2a_dqn_starter
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

## 開始位置へ準備

```bash
python3 task2a_prepare.py --execute
```

既存設定の `look_at_near_floor` 姿勢にして、move_baseで `standby_2a` (2.5,1.85,90度)
へ移動する。これはDQN走行前の準備。新規の準備スクリプトは構文検証済みだが、
このスクリプトを使った初期状態からの実移動はまだ検証していない。
障害物区間に既にいる場合は開始を拒否するので、先にシミュレータを初期化する。

## ターミナルA：点群

```bash
python3 -u pointcloud_obstacles.py _self_filter_radius:=0.23
```

同名の旧ノードが残っている場合は先にそのプロセスを終了する。
0.23 mは現在のHSR姿勢の自己形状除外用。衝突判定の安全距離とは別。

## ターミナルB：プレビュー、その後に走行

```bash
python3 -u task2a_precise_run.py --dqn-first --diagnostics-dir diagnostics/restart_preview
python3 -u task2a_precise_run.py --execute --dqn-first --steps 80 --diagnostics-dir diagnostics/restart_run
```

再試行では診断フォルダを別名にして過去の記録を残す。
プレビューは移動しない。姿勢調整が必要と表示された場合、実行版は開始時に姿勢を補正する。
`INTERNAL GOAL REACHED` は内部ゴール到達であり、公式加点の保証ではない。
終了コード0も内部到達／プレビュー成功を表す。

### 障害物区間後のDQN終了（標準動作）

この走行開始後にスコアが増え、Room2検出が1以上、自己位置のmap Yが2.2 m以上に
なったら、DQNの行動選択を終了する。途中の速度制御も停止してから、`move_base`へ
内部ゴール (2.0, 3.5, 90度) を渡す。通常ナビゲーションの障害物回避は維持する。
切り替え後はDQNへ戻らない。接触通知・ナビゲーション失敗・120秒の時間切れでは
目標をキャンセルして終了する。切り替え理由はsummary.jsonの`handoff`に残る。
以前から残っている30点では切り替えないため、最初から試す際は採点も初期化する。

位置だけの切り替えには `--obstacle-exit-y Y` を指定できる。Yはこの地図の進行方向
である+Yに対し、障害物と機体後端の余裕を含めて通過済みになる確認済みの境界。
壁も含む点群全体の最大Yを障害物終端とは扱わない。境界値は未確定のため標準では無効。
従来の内部ゴールまでDQNを使う動作は `--no-handoff` で選択できる。
切り替え条件とキャンセル処理は自動テスト済み。Gazeboで切り替えを含む完走は未確認。

## ターミナルC：採点確認

```bash
rostopic echo /score
# 別ターミナルで確認する場合
rostopic echo /hsrb_in_room2_detector/count
rostopic echo /undesired_contact_detector/detect
```

各echoはCtrl+Cで終了する。現在の接触falseだけでは過去の非接触は証明できない。
走行中に接触trueを受信した場合、この制御は停止する。
結果は `diagnostics/restart_run/summary.json` に保存される。
scoreの開始値・終了値、Room2検出、接触通知、内部到達を確認する。

## 制御の範囲

### DQN優先モード（実験中）

```bash
python3 -u task2a_precise_run.py --dqn-first --diagnostics-dir diagnostics/dqn_preview_new
python3 -u task2a_precise_run.py --execute --dqn-first --steps 80 --diagnostics-dir diagnostics/dqn_run_new
```

`--dqn-first` は学習済みQ値の順位を使い、経路からの許容距離を1.2 cmから10 cmへ
広げる。経路がなければ計画用の追加余裕1 cmを外して再計画する。
実行時の半径25 cm＋丸め誤差2 mm、静的地図33 cm、接触停止は維持する。
それでも経路がない場合、未訪問の安全なDQN行動を最大6回連続で試す。
これは局所探索であり、ゴールへの到達保証ではない。
`dqn_guarded` / `dqn_local_exploration` / `route_tracking` をログで区別する。
モデルの再学習はしていない。今回の試験では30点の加点を確認したが、
成功率向上・リセットからの通し実行・内部ゴール到達は未確認。
点群ノードも変更しているため、古いプロセスを終了してから上記手順で起動する。

- 学習済みDQNの重みは変更していない。DQNの順位を経路ガードで制限するハイブリッド制御。
- 8方向で表現できない場所は `route_tracking` と明記した経路追従へ切り替える。
- 点群表面への実行距離0.25 m、計画距離0.26 mに2 mmの離散化余裕を加える。
  静的地図への距離は元の0.33 mを維持。低い障害物に対する台車半径はメッシュ計測約0.228 m。
- 現在の格納姿勢・Gazebo環境での試験用。別姿勢や実機の外形を検証したものではない。
- 点群は観測した表面なので、未観測の障害物や裏側まで保証するものではない。
- 点群遅延、自己位置補正では停止して再観測。繰り返し回数に上限がある。

```bash
python3 -m unittest -v test_task2a_navigation
```
