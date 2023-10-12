# pip install opencv-python
# pip install mediapipe
# pip install numpy
# 192.168.8.188:6677
import cv2,os,face_recognition
import numpy as np
import mediapipe as mp
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
def recognize_face(frame_rgb,known_face_encodings: list,known_face_names: list)->(str):
    face_locations = face_recognition.face_locations(frame_rgb)
    face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(frame_rgb, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame_rgb, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame_rgb, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        
    return frame_rgb
    
def visualize(
    image,
    detection_result,
    known_encode_faces,
    known_faces_name,
) -> (np.ndarray):
    """Find the person and recognize it then draws bounding boxes on the input image and return it.
    Args:
        image: The input RGB image.
        detection_result: The list of all "Detection" entities to be visualize.
    Returns:
        Image with bounding boxes.
    """
    MARGIN = 10  # pixels
    ROW_SIZE = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        name = None
        if category_name =='person':
            TEXT_COLOR = (0, 255, 0)  # green
            image = recognize_face(image,known_encode_faces,known_faces_name)
        else:
            TEXT_COLOR = (255, 0, 0)  # red
        # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)
            # Draw label and score
            category = detection.categories[0]
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                                MARGIN + ROW_SIZE + bbox.origin_y)
            
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image

def register_face()->(list):
    faces = os.listdir('faces')
    name_list = [face.split('.')[0] for face in faces]
    faces = ['faces/'+face for face in faces ]
    return faces,name_list
def encode_faces(faces: list):
    faces_encoding = []
    for face in faces:
        face_image = face_recognition.load_image_file(face)
        faces_encoding.append(face_recognition.face_encodings(face_image)[0])
    return faces_encoding

if __name__ == '__main__':
    faces,faces_name_list = register_face()
    encode_faces_list = encode_faces(faces)
    
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        if ret==False:
            break
        # cv2.imshow('frame', frame)

        # STEP 2: Create an ObjectDetector object.
        base_options = python.BaseOptions(model_asset_path='efficientdet_lite2.tflite')
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                                score_threshold=0.5,
                                                category_allowlist = ['person','cell phone'])
        detector = vision.ObjectDetector.create_from_options(options)


        # Create an Image object from the RGB frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        image = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

        # STEP 4: Detect objects in the input image.
        detection_result = detector.detect(image)

        # STEP 5: Process the detection result. In this case, visualize it.
        image_copy = np.copy(image.numpy_view())
        annotated_image = visualize(image_copy, detection_result,encode_faces_list,faces_name_list)
        rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Output',rgb_annotated_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

