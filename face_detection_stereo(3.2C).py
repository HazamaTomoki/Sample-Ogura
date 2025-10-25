import cv2
import numpy as np
import os

# 視差_X座標のオフセット（px）
# 実際のステレオカメラでは、この値は左右の画像検出結果の差 (u_L - u_R) 
STEREO_OFFSET_X = 50 

# Haar-like特徴量ファイル
CASCADE_PATH = './haarcascade_frontalface_default.xml'

# カメラの仕様
SPEC = {
    'W': 640,       # 幅 
    'H': 480,       # 高さ 
    'B': 120,       # ベースライン
    'f_cam': 4.0,   # 焦点距離
    'S_W': 5.12,    # センササイズ 幅
}

# 1. 検出された顔の情報を格納するクラス
class FaceInfo:
    def __init__(self):
        self.center_x = 0  # 顔領域の中心X座標
        self.center_y = 0  # 顔領域の中心Y座標 
        self.found = False # 顔が検出されたか

# 2. 顔検出と中心座標の計算を行う関数
def detect_face(frame, face_cascade):
    """フレームから顔を検出し、中心座標を計算する"""
    frame_copy = frame.copy()
   
    gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    
    # 顔検出の実行
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, # スケール因子
        minNeighbors=5,  # 最小近傍数
        minSize=(30, 30) # 最小検出サイズ
    )

    info = FaceInfo()
    
    if len(faces) > 0:
        # 最も大きく検出された顔（リストの最初の要素）を採用
        x, y, w, h = faces[0]
        
        # 顔の中心座標を計算
        center_x = x + w // 2
        center_y = y + h // 2
        
        info.center_x = center_x
        info.center_y = center_y
        info.found = True

        # デバッグ用に顔領域と中心点を描画
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame_copy, (center_x, center_y), 5, (0, 0, 255), -1)

    return frame_copy, info

# 3. 距離を計算する関数
def calculate_depth_Z(u_L, u_R, spec):
    # 視差 D: 左右のXピクセル座標の差 [pixel]
    D = u_L - u_R
    
    # ピクセルサイズ p: センサ幅 / 解像度幅 [mm/pixel]
    p = spec['S_W'] / spec['W']
    
    # 分母: D * p
    denominator = D * p
    
    if D <= 0 or denominator == 0: 
        return 0.0 # 視差がない、またはエラー
    
    # 分子: B * f
    numerator = spec['B'] * spec['f_cam']
    
    # 奥行き Z [mm]
    Z_mm = numerator / denominator
    
    # 結果をメートル [m] に変換して返す
    return Z_mm / 1000.0


def main_c():
    if not os.path.exists(CASCADE_PATH):
        print(f" エラー: {CASCADE_PATH} が見つかりません。同じディレクトリに配置してください。")
        return

    # Haar-like特徴量のカスケード分類器をロード
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    # カメラのオープン
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    print("✅ 3.2 C 実行中: 顔までの距離を計測します。ESCキーで終了します。")

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            break

        # 左カメラの検出を実行
        result_left, info_L = detect_face(frame_orig.copy(), face_cascade)
        
        # 右カメラの検出結果はシミュレーション
        result_right, _ = detect_face(frame_orig.copy(), face_cascade)
        
        distance_Z = 0.0
        display_text = "No Face Detected"

        if info_L.found:
            # 1. 左カメラのX座標 (uL) を取得
            u_L = info_L.center_x
            
            # 2. 右カメラのX座標 (uR) をシミュレーションで計算
            # 検出された左のX座標から固定の視差分を引く
            u_R_simulated = u_L - STEREO_OFFSET_X 
            
            # 3. 距離 Z [m] を計算
            distance_Z = calculate_depth_Z(u_L, u_R_simulated, SPEC)
            
            # 結果表示用の文字列
            display_text = (
                f"L(u): {u_L}, R(u): {u_R_simulated} | "
                f"D: {STEREO_OFFSET_X} px | "
                f"Z: {distance_Z:.2f} m"
            )
        
        # 距離表示を左フレームに合成
        cv2.putText(result_left, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 左右のフレームを結合して表示
        combined_frame = cv2.hconcat([result_left, result_right])
        cv2.imshow('3.2 C: Depth Measurement (Simulated)', combined_frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27: # ESCキー
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_c()