
import torch.nn.functional as F
import torch
import warnings

class CoordinatesTranslator:
    def __init__(self, x0, y0, x1, y1, img_index, triangle_index, device) -> None:
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1)) 
        y1 = int(round(y1))
        H, W = (640, 640)
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, W)
        y1 = min(y1, H)
        self.device = device
        self.resize_shape = (64, 64)
        self.coords = [x0, y0, x1, y1]
        self.img_index = img_index
        self.triangle_index = triangle_index
        self.tri = (x1 - x0, y1 - y0)


    def is_positive_area(self, x0, y0, x1, y1):
        
        return  x1 > x0 and y1 > y0


    def crop_triangle(self, img):

        img = img.to(self.device)
        if list(img.shape) != [3, 640, 640]:
            raise KeyError(f"img shape should be (640, 640), 3 but found {list(img.shape)}")
        x0, y0, x1, y1 = self.coords
        triangle_ = img[:,y0:y1, x0:x1]#.permute(1, 2, 0)
        triangle_ = triangle_.reshape(1, *triangle_.shape)
        triangle_ = F.interpolate(triangle_,(64,64))[0].permute(1,2,0)

        return triangle_
    


    def translate_triangle_to_image(self, x_, y_):
        """
            it translates from coordinates inside the space of triangle to within the space of img
        """ 
        x0, y0, x1, y1 = self.coords
        if x_ < 0 or x_ > self.resize_shape[0] or y_ > self.resize_shape[1] or y_ < 0:
            
            warnings.warn("vod_triangles returning vertex out of the original triangle??"
                f"Condition is {x_} < 0 or {x_} > {self.resize_shape[0]} or {y_} > {self.resize_shape[1]} or {y_} < 0")
            
        return (
                (x_ * self.tri[0] / self.resize_shape[0]) + x0, 
                (y_ * self.tri[1] / self.resize_shape[1]) + y0
            )

    def translate_x_to_img(self, x_):
        
        return self.translate_triangle_to_image(x_, 0)[0]

    def translate_y_to_img(self, y_):
        return self.translate_triangle_to_image(0, y_)[1]

    def __str__(self) -> str:
        return (f"Triangle coords in img(x0,y0,x1,y1) =({self.coords})\n"
              f"using {self.device}\n"
              f"origin triangle size tri.shape = {self.tri}\n"
              f"resizing to {self.resize_shape}")


def fetch_triangles_translators_batches(yolo_output, imgs, n_batch, device):

    # cut triangle
    imgs_triangles = []
    translators = []
    for img_num in range(len(yolo_output)):
        for tr_num in range(len(yolo_output[img_num])):

            translator = CoordinatesTranslator(
                *yolo_output[img_num][tr_num][:4], 
                img_index=img_num,
                triangle_index=tr_num, 
                device=device
            )
            imgs_triangles.append(
                translator.crop_triangle(imgs[img_num,:]) 
            )
            translators.append(translator)
            if len(imgs_triangles) == n_batch:
                yield torch.stack(imgs_triangles), translators
                imgs_triangles = []
                translators = []
    if len(imgs_triangles) != 0:
        yield torch.stack(imgs_triangles), translators


def convert_coords_list2dicts(coords, translator = None):
    vert_dicts = []
    
    for x_, y_ in coords:
        if translator is None:
            vert_dicts.append({
                "x" : x_,
                "y" : y_
            })
        else:
            vert_dicts.append({
                "x" : int(translator.translate_x_to_img(x_)), 
                "y" : int(translator.translate_y_to_img(y_))
            })
    return vert_dicts


def convert_yolo_output2dict(yolo_single_res : list):
    
    x0, y0, x1, y1, c, _ = yolo_single_res
    vert_dicts = [
        {"x" : int(x0), "y" : int(y0)},
        {"x" : int(x1), "y" : int(y1)}
    ]
    return vert_dicts, c


def broaden_yolo_output(yolo_output, bump=0.15):
    """Increase yolo window by :bump: on each side, as percentage of width and height.
    """
    for yolo_output_per_image in yolo_output:
        for bbox in yolo_output_per_image:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            bbox[0] = bbox[0] - bump * w
            bbox[1] = bbox[1] - bump * h
            bbox[2] = bbox[2] + bump * w
            bbox[3] = bbox[3] + bump * h
