import cv2
import numpy as np
import os
from architecture import InceptionResNetV2
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer
from face_detection import RetinaFace
import random
from head_pose_estimation import DetectHeadPose
from eye_tracker import DetectRealEyes
from mouth_opening_detector import DetectRealMouth, setMouth

# Initialize models and normalizer
package_file = os.path.abspath(os.path.dirname(__file__))
l2_normalizer = Normalizer('l2')

confidence_t = 0.97
recognition_t = 0.50
required_size = (160, 160)

face_encoder = InceptionResNetV2()
path_m = "facenet_keras_weights.h5"
face_encoder.load_weights(os.path.join(package_file, path_m))
face_detector = RetinaFace()

def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode

def change_box_value(bbox):
    return [int(bbox[0]), int(bbox[1]), int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])]


def DetectReal(frame, movement, func):
    try:
        if func == 'Head':
            if movement in DetectHeadPose(frame):
                return True
        if func == 'Eye':
            if movement in DetectRealEyes(frame):
                return True
        elif func == 'Mouth':
            if movement in DetectRealMouth(frame):
                return True
    except:
        pass

    # return False



def detect_live_stream(data_encoding_list):    
    HeadMovement = ['Left', 'Right']
    EyeMovement = ['Up', 'Left', 'Right']
    read = ['I am a User', 'Atlas Honda is an Automobile Company', 'I am a Driver']
    h = random.randint(0, 1)
    e = random.randint(0, 2)
    func = ['Head', 'Eye', 'Mouth']
    movements = [HeadMovement[h], EyeMovement[e], 'Open']
    
    level = 0
    count = 0

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not Working")
        return

    stm = 0    

    while True:
        if level <= 1:
            print("Move "+func[level]+" towards: ", movements[level])
        else:
            if stm == 0:
                input('Please Give One Picture on Closed Mouth')
                setMouth(frame)
                print("Please Read: ", read[e])
                stm += 1

        ret, frame = cap.read()
        detectFrame = frame
        if not ret:
            print("Failed to capture frame")
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector(img_rgb)
        if len(results) == 0:
            print("No face detected")
            continue
        for res in results:
            if res[2] < confidence_t:
                continue
            bbox = change_box_value(res[0])
            face, pt_1, pt_2 = get_face(img_rgb, bbox)
            encode = get_encode(face_encoder, face, required_size)
            encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
            name = 'unknown'
            for result in data_encoding_list:
                dist = cosine(result[5], encode)
                if dist < recognition_t:
                    print('recog')
                    if DetectReal(detectFrame, movements[level], func[level]):
                        count += 1
                        break
        cv2.imshow('frame', frame)
        print('count', count, 'level', level)
        if count > 3:
            level += 1
            count *= 0
        print('count', count, 'level', level)
        if level > 2:
            print('User Successfully Detected!')
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()





# def detect_live_stream(data_encoding_list):    
#     # HeadMovement = ['Up', 'Down', 'Left', 'Right']
#     # EyeMovement = ['Up', 'Left', 'Right']
#     # read = ['I am a User', 'Atlas Honda is an Automobile Company', 'I am a Driver']
#     # h = random.randint(0, 3)
#     # e = random.randint(0, 2)
#     # detections = []
#     # print("Move Head towards: ", HeadMovement[h])
#     # detections.append(take_frames(HeadMovement[h], 'Head'))
#     # print('\n\nPlace Head to Center....\n\n')
#     # print("Move Eyes towards: ", EyeMovement[e])
#     # detections.append(take_frames(EyeMovement[e], 'Eye'))
#     # print('\n\nPlace Head to Center....\n\nPlease Keep your Mouth Close!\n\n')
#     # print("Please Read: ", read[e])
#     # detections.append(take_frames('Open', 'Mouth'))

#     # print(detections)

#     # input('continue??')

#     # cv2.destroyAllWindows()

#     cap = cv2.VideoCapture(0)  
#     frames = []    
#     total_frames = 0
#     while True:
#         ret, frame = cap.read()
#         # if not ret:
#         #     break
#         frames.append(frame)
#         total_frames+=1
#         if total_frames==25:
#             break
        
#     frame = frames[1]

#     if frame:
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_detector(img_rgb)
#         for res in results:
#             if res[2] < confidence_t:
#                 continue
#             bbox = change_box_value(res[0])
#             face, pt_1, pt_2 = get_face(img_rgb, bbox)
#             encode = get_encode(face_encoder, face, required_size)
#             encode = l2_normalizer.transform(encode.reshape(1, -1))[0]

#             name = 'unknown'

#             for result in data_encoding_list:
#                 min_dic = {}
#                 dist = cosine(result[5], encode)
#                 if dist < recognition_t:
#                     min_dic['name'] = result[1]
#                     min_dic['ref_no'] = result[2]
#                     min_dic['Summary'] = result[3]
#                     min_dic['image_bytes'] = result[4]
#                     name = result[1]
#                     print(f"Match found: {name}, distance: {dist:.2f}")

#                 if name == 'unknown':
#                     cv2.rectangle(frame, pt_1, pt_2, (0, 0, 255), 2)
#                     cv2.putText(frame, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
#                 else:
#                     cv2.rectangle(frame, pt_1, pt_2, (0, 255, 0), 2)
#                     cv2.putText(frame, name + f'__{dist:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 2)


def data_setting(data):
    data_final = []
    for re in data:
        result_image_encodes = np.frombuffer(re[5], dtype=np.float32)
        data_final.append((re[0], re[1], re[2], re[3], re[4], result_image_encodes))
    return data_final
