import face_recognition
import cv2
import time

ramadhan_image = face_recognition.load_image_file('me.jpg')
ramadhan_face_encoding = face_recognition.face_encodings(ramadhan_image)[0]

webcam = cv2.VideoCapture(0)

# Set the desired fps
desired_fps = 13820

while True:
    start_time = time.time()  # Record the start time of the loop

    # Capture frame by frame
    ret, frame = webcam.read()

    # Finding the face locations and the encoding in the image
    face_locations = face_recognition.face_locations(frame)
    face_encoding = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encoding):
        # Check if the image matches the jpg image
        matches = face_recognition.compare_faces([ramadhan_face_encoding], face_encoding)


        if matches[0]:
            name = 'Welcome Ramadhan Hussein!'
            # Drawing the Rectangle on the image with the labels
            cv2.rectangle(frame, (left, top), (bottom, right), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the fps in the frame
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Testing Webcam', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        webcam.release()
        break

    # Calculate the time elapsed for each iteration
    elapsed_time = time.time() - start_time

    # Pause to achieve the desired fps
    time.sleep(max(0, 1/desired_fps - elapsed_time))

