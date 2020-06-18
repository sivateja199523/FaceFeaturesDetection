
import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

#QT_QPA_PLATFORM_PLUGIN_PATH=/Users/ABC/qt/plugins
#DYLD_PRINT_LIBRARIES=1
#QT_DEBUG_PLUGINS=1
image = cv2.imread("1.jpg")
result = detector.detect_faces(image)

print(result)

bounding_box = result[0]['box']
keypoints = result[0]['keypoints']
cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)
cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
cv2.imwrite("ivan_drawn.jpg", image)
cv2.namedWindow("image")
cv2.imshow("image",image)
cv2.waitKey(0)

# cap = cv2.VideoCapture('Naman Bday.mp4')
# while True:
#     # Capture frame-by-frame
#     __, frame = cap.read()
#
#     # Use MTCNN to detect faces
#     result = detector.detect_faces(frame)
#     if result != []:
#         for person in result:
#             bounding_box = person['box']
#             keypoints = person['keypoints']
#
#             cv2.rectangle(frame,
#                           (bounding_box[0], bounding_box[1]),
#                           (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
#                           (0, 155, 255),
#                           2)
#
#             cv2.circle(frame, (keypoints['left_eye']), 2, (0, 155, 255), 2)
#             cv2.circle(frame, (keypoints['right_eye']), 2, (0, 155, 255), 2)
#             cv2.circle(frame, (keypoints['nose']), 2, (0, 155, 255), 2)
#             cv2.circle(frame, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
#             cv2.circle(frame, (keypoints['mouth_right']), 2, (0, 155, 255), 2)
#     # display resulting frame
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # When everything's done, release capture
# cap.release()
# cv2.destroyAllWindows()