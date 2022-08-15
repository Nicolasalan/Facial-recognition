#!/usr/bin/env python3
import sys
import os
import rospy
from sensor_msgs.msg import Image
import rospkg
import fnmatch
import cv2
import time
import face_recognition
from cv_bridge import CvBridge, CvBridgeError

class FaceRecogniser(object):

    def __init__(self):
        rospy.loginfo("Start FaceRecogniser Init process...")
       # obtém uma instância do RosPack com os caminhos de pesquisa padrão
        self.rate = rospy.Rate(5)
        rospack = rospkg.RosPack()
        # obtém o caminho do arquivo para my_face_recogniser
        self.path_to_package = rospack.get_path('hera_face')

        self.bridge_object = CvBridge()
        rospy.loginfo("Start camera suscriber...")
        self.topic = "/usb_cam/image_raw"
        self._check_cam_ready()
        self.image_sub = rospy.Subscriber(self.topic,Image,self.camera_callback)
        rospy.loginfo("Finished FaceRecogniser Init process...Ready")

    def _check_cam_ready(self):
      self.cam_image = None
      while self.cam_image is None and not rospy.is_shutdown():
         try:
               self.cam_image = rospy.wait_for_message(self.topic, Image, timeout=1.0)
               rospy.logdebug("Current "+self.topic+" READY=>" + str(self.cam_image))

         except:
               rospy.logerr("Current "+self.topic+" not ready yet, retrying.")

    def camera_callback(self,data):
        self.cam_image = data

    def loop(self):

        while not rospy.is_shutdown():
            self.recognise(self.cam_image)
            self.rate.sleep()

    def recognise(self,data):

        # Obtenha uma referência para a webcam # 0 (a padrão)
        try:
            video_capture = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)
        
        self.people_dir = '/home/hera/catkin_hera/src/3rdParty/vision_system/hera_face/face_images/'

        files = fnmatch.filter(os.listdir(self.people_dir), '*.jpg')

        faces_images = []
        known_face_encodings = []
        known_face_names = []
        for i in range(0,len(files)):
            faces_images.append(face_recognition.load_image_file(self.people_dir + files[i]))
            known_face_encodings.append(face_recognition.face_encodings(faces_images[i])[0])
            known_face_names.append(files[i].replace('.jpg',''))


        # robot vision
        small_frame = cv2.resize(video_capture, (0, 0), fx=0.5, fy=0.5)

        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Veja se o rosto é compatível com o(s) rosto(s) conhecido(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Face"

            # Se uma correspondência foi encontrada em known_face_encodings, apenas use a primeira.
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            face_names.append(name)


        for (top, right, bottom, left), name in zip(face_locations, face_names):

            # Redimensionar os locais de rosto, pois o quadro em que detectamos foi dimensionado para 1/4 do tamanho
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Desenhe uma caixa ao redor do rosto
            cv2.rectangle(video_capture, (left, top), (right, bottom), (0, 0, 255), 2)

            # Desenhe uma etiqueta com um nome abaixo do rosto
            cv2.rectangle(video_capture, (left, bottom - 35), (right, bottom), (0, 0, 255))
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(video_capture, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            height, width = video_capture.shape[:2]
            print("Largura da image: ",width)
            
            print("Posicao em x:",right)

        face_names_str = " - ".join(face_names)
        rospy.loginfo("People: " + face_names_str)


        # Display the resulting image
        cv2.imshow("Image window", video_capture)
        cv2.waitKey(1)

def main():
    rospy.init_node('face_recognising_python_node', anonymous=True, log_level=rospy.INFO)

    face_recogniser_object = FaceRecogniser()

    face_recogniser_object.loop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
