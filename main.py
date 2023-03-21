import cv2, dlib, sys 
import numpy as np

frame_scale = 0.5
mask_scale = 1.5
"""
http://dlib.net/python/index.html#dlib_pybind11.shape_predictor
http://devdoc.net/c/dlib-19.7/python/
"""
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#캠에서 데이터 읽어온다 여러대일경우 0 1 2 3...
cap = cv2.VideoCapture(0)

overlay = cv2.imread('IMG_9323.png', cv2.IMREAD_UNCHANGED)

def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    try:
        bg_img = background_img.copy()
        #convert 3channels to 4channels
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b,g,r,a =cv2.split(img_to_overlay_t)
        mask = cv2.medianBlur(a,5)

        h,w,_ = img_to_overlay_t.shape

        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)
        
        #conver 4channels to 4channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)
        return bg_img
    except:
        return background_img

if not cap.isOpened():
    print("camera error")
    exit(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (int(frame.shape[1] * frame_scale), int(frame.shape[0] * frame_scale)))
    # original_frame = frame.copy()

    #얼굴인식기
    faces = detector(frame)
    #디텍터는 여러개의 얼굴을 반환한다.
    result = None
    if faces:
        face = faces[0]
        #facial landmarks 추출
        dlib_shape = predictor(frame, face)
        #facial landmarks 68개의 좌표를 반환함 
        shape_2d = np.array([[part.x, part.y] for part in dlib_shape.parts()])

        #얼굴 범위 지정
        top_left = np.min(shape_2d, axis=0)
        bottom_right = np.max(shape_2d, axis=0)
        face_size = int(max(bottom_right - top_left) * mask_scale)
        #센터
        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.intc)
        #오리지널 프레임에 오버레이 이미지를 좌표에 맞게 마스킹한다 (비트연산)
        frame = overlay_transparent(frame, overlay, center_x, center_y, overlay_size=(face_size, face_size))

        # for s in shape_2d:
        #     cv2.circle(frame, center=tuple(s), radius=2, color=(255, 255, 255))

        # frame = cv2.rectangle(frame, pt1=(face.left(),face.top()), pt2=(face.right(), face.bottom())\
        #                 ,color=(255,255,255), thickness=1, lineType=cv2.LINE_AA)


    frame = cv2.flip(frame, 1)
    cv2.imshow('window', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
