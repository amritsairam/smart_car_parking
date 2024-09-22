import cv2
import numpy as np
from utils import get_parking_spots_bboxes,empty_or_not

def calc_diff(img1,img2):
    return abs(np.mean(img2)-np.mean(img1))

video_path='/Users/sairam/Desktop/desktop/computer vision projects/parking_spot_detector/data/parking_1920_1080.mp4'

mask_path='/Users/sairam/Desktop/desktop/computer vision projects/parking_spot_detector/mask_1920_1080.png'

mask=cv2.imread(mask_path,0)

cap=cv2.VideoCapture(video_path)

connected_components=cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S)

spots=get_parking_spots_bboxes(connected_components)

frame_nmr=0
prev_frame=None

spots_status=[None for j in spots]
diffs=[None for j in spots]

update=30

while True:
    ret, frame = cap.read()

    if frame_nmr%update==0 and prev_frame is not None:

        for spot_index,spot in enumerate(spots):
            x1,y1,w,h= spot

            spot_crop=frame[y1:y1+h, x1:x1+w, :]
            diffs[spot_index]=calc_diff(spot_crop,prev_frame[y1:y1+h, x1:x1+w, :])
        
        arr_=[]
        for j in np.argsort(diffs):
            if diffs[j]/np.amax(diffs)>0.4:
                arr_.append(j)
        print(arr_)

        for spot_index in arr_:
            spot=spots[spot_index]
            x1,y1,w,h= spot

            spot_crop=frame[y1:y1+h, x1:x1+w, :]
            spots_status[spot_index]=empty_or_not(spot_crop)

    if frame_nmr%update==0:
        prev_frame=frame.copy()

    for spot_index,spot in enumerate(spots):
        x1,y1,w,h= spots[spot_index]
        if spots_status[spot_index]:
            frame=cv2.rectangle(frame, (x1,y1), (x1+w, y1+h), (0,255,0), 2)
        else:
            frame=cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)

    cv2.putText(frame, 'Available spots: {}/{}'.format(str(sum(s for s in spots_status if s is not None)), str(len(spots_status))), (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 5)


    cv2.imshow('Frame', frame)

        
    # Press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
