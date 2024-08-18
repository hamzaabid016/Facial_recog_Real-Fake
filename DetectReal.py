from head_pose_estimation import DetectHeadPose
from eye_tracker import DetectRealEyes
from mouth_opening_detector import DetectRealMouth
import random


def DetectRealPerson(frames, HeadMovement, EyeMovement):
    head = 0
    eye = 0
    mouth = 0

    for frame in frames:
        # try:
            hd = DetectHeadPose(frame)
            ey = DetectRealEyes(frame)
            if type(hd) == str and head == 0:
                if HeadMovement in hd:
                    head += 1
            if type(ey) == str and eye == 0:
                if EyeMovement in ey:
                    eye += 1
            if 'Open' in DetectRealMouth(frame) and mouth == 0:
                mouth += 1
            print(head, eye, mouth)
            print(EyeMovement, ey)
            print(HeadMovement, hd)

            if head > 0 and eye > 0 and mouth > 0:
                return "Person is Real"
        # except:
            # pass

    return "Person is Fake"