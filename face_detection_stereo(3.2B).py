import cv2
import numpy as np
import os

STEREO_OFFSET_X = 50 


CASCADE_PATH = './haarcascade_frontalface_default.xml'


class FaceInfo:
    def __init__(self):
        self.center_x = 0  # 顔領域の中心X座標 
        self.center_y = 0  # 顔領域の中心Y座標 
        self.found = False # 顔が検出されたか

# 顔検出と中心座標の計算を行う関数
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
        # 最も大きい顔（
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

def main_b():
    if not os.path.exists(CASCADE_PATH):
        print(f"エラー: {CASCADE_PATH} が見つかりません。")
        return

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    print(" 実行中: 左右のカメラをシミュレートし、顔の中心座標を検出します。")
    print("ESCキーで終了します。")

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            break

        # 左カメラの検出
        result_left, info_L = detect_face(frame_orig, face_cascade)
        
        # 右カメラの検出X(座標をずらす)
        # 実際には右カメラのフレームを読み込んで検出する
        result_right, _ = detect_face(frame_orig, face_cascade)
        
        # 左右のフレームを横に結合して表示
        combined_frame = cv2.hconcat([result_left, result_right])
        
        if info_L.found:
            #左右の中心座標を求める（右はシミュレート）
            u_L = info_L.center_x
            u_R = info_L.center_x - STEREO_OFFSET_X 
            
            # 結果出力
            print(f"L (uL, vL): ({u_L}, {info_L.center_y}) | R (uR, vR) Simulated: ({u_R}, {info_L.center_y})")
            
            # 検出座標を画面に表示
            cv2.putText(result_left, f"L(u): {u_L}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(result_right, f"R(u) Sim: {u_R}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


        cv2.imshow('3.2 B: Stereo Face Detection (Left | Right)', combined_frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_b()