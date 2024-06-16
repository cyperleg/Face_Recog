from FaceRecogniser import FaceRecogniser
import time


start_time = time.time()
img_in = 'in.png'

img_verify = 'verify.jpg'


#print(FaceRecogniser.check_face_presence(img_in), FaceRecogniser.check_face_presence(img_verify))

img_verify_encode = FaceRecogniser.get_face_encoding(img_verify)


recog1 = FaceRecogniser(img_in, img_verify_encode)

print(recog1())

print(float(time.time() - start_time))
