import cv2
import numpy as np
import os

def main():
    image_path = 'input.jpg' 
    
    if not os.path.exists(image_path):
        print(f"エラー: {image_path} が見つかりません。")
        print("'input.jpg' という名前の画像を置いてください。")
        return

        
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"エラー: {image_path} の読み込みに失敗しました。ファイルの確認してください。")
        return

    # 画像を4分割するための中心座標を計算
    height, width = img.shape[:2]
    h_half = height // 2
    w_half = width // 2
    
    # 1. 4つの領域に分割（NumPyスライス）
    top_left = img[0:h_half, 0:w_half]           # 左上 (TL)
    top_right = img[0:h_half, w_half:width]      # 右上 (TR)
    bottom_left = img[h_half:height, 0:w_half]   # 左下 (BL)
    bottom_right = img[h_half:height, w_half:width] # 右下 (BR)
    
    # 2. 領域を入れ替えて結合: 左上(TL) <-> 右下(BR)
    # 上半分: BR と TR を結合
    top_concat = cv2.hconcat([bottom_right, top_right])
    
    # 下半分: BL と TL を結合
    bottom_concat = cv2.hconcat([bottom_left, top_left])
    
    # 上下を結合
    result_img = cv2.vconcat([top_concat, bottom_concat])

    # 結果の表示と保存
    cv2.imshow('1. Original Image', img)
    cv2.imshow('2. Swapped Image (TL <-> BR)', result_img)
    cv2.imwrite('swapped_image_result.png', result_img)
    print("✅ 3.1 A 完了: 画像を保存しました: swapped_image_result.png")

    cv2.waitKey(0) # キー入力があるまで待機
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()