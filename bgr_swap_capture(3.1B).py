import cv2

def main():
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        return

    cv2.namedWindow('B-R Swapper', cv2.WINDOW_AUTOSIZE)
    print("実行中: 青と赤が入れ替わったカメラ映像が表示されました。ESCキーを押すと終了。")
    
    while(1):
        ret, frame = cap.read()
        
        if not ret:
            print("フレームの読み込みに失敗しました。")
            break
        
       
        swapped_frame = frame[:, :, [2, 1, 0]]
        
        cv2.imshow('B-R Swapper', swapped_frame)
        
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()