import cv2
import numpy as np
import os

# 変換後の目標座標 
# 順番[左上, 右上, 右下, 左下]
# [0, 0], [200, 0], [200, 200], [0, 200]
DST_POINTS = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])

# ユーザーがクリックして取得する、変換前の4隅の座標
src_points = []
img_display = None
img_original = None

def mouse_callback(event, x, y, flags, param):
    """マウスイベントを処理し、4点取得後にホモグラフィ変換を実行する"""
    global src_points, img_display, img_original

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_points) < 4:
            src_points.append((x, y))
            
            # クリックした位置に点を描画
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(img_display, str(len(src_points)), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow('Image Correction', img_display)
            print(f"点 {len(src_points)}: ({x}, {y}) を取得しました。")

            # 4点取得完了
            if len(src_points) == 4:
                src_np = np.float32(src_points)

                # 1. ホモグラフィ行列 H の算出 
                # cv2.findHomography は DLT (Direct Linear Transformation) を用いてHを算出
                H, _ = cv2.findHomography(src_np, DST_POINTS)
                print("\n◯算出したホモグラフィ行列 H (3x3):\n", H)

                # 2. ホモグラフィ行列を用いて画像を変換 
                # 変換後の画像サイズは (200, 200)
                corrected_img = cv2.warpPerspective(img_original, H, (200, 200))

                # 補正画像を新しいウィンドウで表示
                cv2.imshow('Corrected Image (Front View)', corrected_img)
                cv2.imwrite('corrected_ar_marker.png', corrected_img)
                print("\n◯画像補正完了、'corrected_ar_marker.png'として保存")
                print("ESCキーを押して終了")


def main_33c():
    global img_display, img_original

    # ARマーカーを含む画像ファイル
    image_path = 'ar.jpg' 
    img_original = cv2.imread(image_path)
    
    if img_original is None:
        print(f"エラー: 画像ファイル '{image_path}' ない")
        print("画像を 'ar.jpg' という名前で配置して")
        return

    img_display = img_original.copy()

    cv2.namedWindow('Image Correction')
    cv2.setMouseCallback('Image Correction', mouse_callback)

    print("=== 実行中 ===")
    print("ARマーカーの四隅を、以下の順序でクリック:")
    print("1. 左上, 2. 右上, 3. 右下, 4. 左下")
    
    cv2.imshow('Image Correction', img_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main_33c()