import cv2

def main():
    # 初始化摄像头（0表示默认摄像头）
    cap = cv2.VideoCapture(0)
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 键退出程序")
    
    while True:
        # 读取一帧图像
        ret, frame = cap.read()
        
        # 如果正确读取帧，ret为True
        if not ret:
            print("无法获取画面")
            break
            
        # 显示图像
        cv2.imshow('摄像头画面', frame)
        
        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放摄像头资源
    cap.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 