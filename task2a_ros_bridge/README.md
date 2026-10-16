# Kanami担当: 学習済みDQNのROS接続

既存のカメラ物体検出→BBox→点群→map座標を使います。
Gazeboの物体正解座標は使用しません。新規学習は不要です。
円形近似は学習時と同じ: Foam=0.08 m、Lego=0.19 m、ロボット0.30 m、余裕0.03 m。

## 1. 配置
ZIP内の task2a_ros_bridge フォルダを既存の task2a_dqn_starter 内に置きます。
既存 environment.py、train_dqn.py、地図、モデルは変更しません。

## 2. 学習側のPython 3.11で重みを書き出す
```bash
cd ~/workspace/task2a_dqn_starter
PYTHONPATH=. .venv/bin/python task2a_ros_bridge/export_dqn_numpy.py \
  --model task2a_dqn_2_obstacles_6000steps.zip
```
task2a_policy_numpy.npz を作成。SB3とNumPyのQ値・行動一致を確認します。
既に同名出力がある場合は上書きしません。別名の --output を指定してください。
このNPZをROS側で使うので、ROSのPython3.6にSB3/Torch/Scipyを追加する必要はありません。

## 3. メイン内に担当関数を追加
```bash
.venv/bin/python task2a_ros_bridge/install_ros_bridge.py \
  --main ~/workspace/src/wrs_algorithm21/scripts/wrs_algorithm/algorithm/wrs_main_node.py
cp task2a_ros_bridge/task2a_dqn.launch \
  ~/workspace/src/wrs_algorithm21/launch/task2a_dqn.launch
```
インストーラはメインのバックアップを作り、担当コメントで囲んだメソッド群を
WrsMainController内に追加します。run()冒頭に専用モードの分岐を追加。
元のTask1・Task2a関数は置換しません。変更済みのメインに対しても構造を確認して処理します。
二重インストールは拒否します。git diff等で変更を確認できます。

## 4. ROSターミナル（既存catkin_py3）の起動
学習venvをactivateしていない既存のROSターミナルで、シミュレータと認識を起動。
通常のstart_all.shは元のmainも同時に起動するため、そのまま併用しないでください。
既存のmainが動いている場合は終了し、認識ノードを別ターミナルで起動します。

```bash
roslaunch hsr_perception pcd_to_tf.launch
```
```bash
roslaunch wrs_detector frcnn_finetuned.launch
```

最初の確認（台車・アームを動かさない）:
```bash
roslaunch wrs_algorithm task2a_dqn.launch dry_run:=true
```
現在位置で1回認識・推論し、RVizのMarkerArray /task2a_dqn/markers に
物体の円、除外領域、ロボット半径、選んだ移動方向を表示して待機します。
学習姿勢と同じ +90度で、障害物が見える位置・頭の姿勢が必要です。
現在の姿勢が違う/対象が見えない場合は理由を表示して終了します。
認識が見える待機位置に既にいるなら、このモードから確認してください。

## 5. Gazeboで移動
dry-runノードをCtrl+Cで終了してから:
```bash
roslaunch wrs_algorithm task2a_dqn.launch dry_run:=false
```
頭をlook_at_near_floorへ変更→既存standby_2aへ移動→認識→推論→0.15 mの移動。
各移動後に再観測し、ゴールまで繰り返します。Task1/Task2bは実行しません。
通常のstart_task.launchは専用パラメータがないため元のrun()で動きます。

## 評価・制限
- 2Dの73%はDQN単独。ROS版は既知領域を確認し、安全な候補の中から
  Q値最大の方向へ切り替えるガードを標準で使用。介入をログに出します。
  ROS版は「DQN＋ガード」であり、DQN単独の成績と混同しません。
- /move_base/make_planで計画経路を検査します。走行中も現在位置を監視。
  ただし既知障害物だけの検査で、将来の局所経路や見落としに対する安全保証ではありません。
- 既存goto_nameによるstandbyへの接近は元の移動処理です。
- カメラ視野は学習側の仮定(+map-y,100度,1.5 m)を第3入力に再現。
  実カメラの校正・頭姿勢・向きの差は残ります。
- 静的地図は学習地図と比較して一致しない場合は停止します。
- 点群由来の把持座標と物体中心のずれ、3cm余裕の妥当性は未検証。
- ラベルはfoam_brick/lego_duploに限定。未対応ラベルや異常な高さは停止。
- 未検出と空き領域を完全には区別できません。1個検出できても全物体が見えている保証はありません。
- 姿勢が90度から15度超ずれる、全候補がふさがる、移動失敗・時間切れ、
  短い往復の繰り返しで停止します。
- 認識位置を記憶するので、観測の揺れによる除外領域の拡大があり得ます。
- Python3.6構文、学習観測との数値一致、NumPyネットワーク処理を作成側で確認。
  実際の学習済み重みの一致検証はexport時に実行します。
  このパッケージのGazebo動作は未検証です。異常ログを確認して調整してください。
