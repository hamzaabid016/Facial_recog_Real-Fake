import cv2
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks, draw_marks

face_model = get_face_detector()
landmark_model = get_landmark_model()

outer_points = [[49, 59], [50, 58], [51, 57], [52, 56], [53, 55]]
d_outer = [0] * 5
inner_points = [[61, 67], [62, 66], [63, 65]]
d_inner = [0] * 3
font = cv2.FONT_HERSHEY_SIMPLEX 

def setMouth(face):
    global d_outer, d_inner
    rects = find_faces(face, face_model)
    for rect in rects:
        shape = detect_marks(face, landmark_model, rect)
        draw_marks(face, shape)
        cv2.putText(face, 'Press r to record Mouth distances', (30, 30), font,
                    1, (0, 255, 255), 2)

    for i in range(100):
        for i, (p1, p2) in enumerate(outer_points):
            d_outer[i] += shape[p2][1] - shape[p1][1]
        for i, (p1, p2) in enumerate(inner_points):
            d_inner[i] += shape[p2][1] - shape[p1][1]

    d_outer[:] = [x / 115 for x in d_outer]
    d_inner[:] = [x / 115 for x in d_inner]


def DetectRealMouth(img):
    global d_outer, d_inner
    rects = find_faces(img, face_model)
    for rect in rects:
        shape = detect_marks(img, landmark_model, rect)
        cnt_outer = 0
        cnt_inner = 0
        draw_marks(img, shape[48:])
        for i, (p1, p2) in enumerate(outer_points):
            if d_outer[i] + 3 < shape[p2][1] - shape[p1][1]:
                cnt_outer += 1 
        for i, (p1, p2) in enumerate(inner_points):
            if d_inner[i] + 2 < shape[p2][1] - shape[p1][1]:
                cnt_inner += 1
        if cnt_outer > 3 and cnt_inner > 2:
            return 'mouth Open'
 