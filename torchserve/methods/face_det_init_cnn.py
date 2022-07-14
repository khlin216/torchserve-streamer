from facenet_pytorch import MTCNN, InceptionResnetV1
from methods.constants import IMG_SHAPE, MAP_LOCATION
# If required, create a face detection pipeline using MTCNN:

def create_mtcnn():
    mtcnn = MTCNN(
        image_size=IMG_SHAPE, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=MAP_LOCATION
    )
    return mtcnn

def create_resnet():
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(MAP_LOCATION)
    return resnet