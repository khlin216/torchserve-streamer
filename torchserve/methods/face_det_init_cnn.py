from facenet_pytorch import MTCNN, InceptionResnetV1
from methods.constants import IMG_WIDTH, MAP_LOCATION, MIN_FACE_SIZE
# If required, create a face detection pipeline using MTCNN:


def create_mtcnn(device=MAP_LOCATION):
    print(IMG_WIDTH, MIN_FACE_SIZE)
    mtcnn = MTCNN(
        image_size=IMG_WIDTH, margin=0, min_face_size=MIN_FACE_SIZE,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    return mtcnn


def create_resnet():
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(MAP_LOCATION)
    return resnet
