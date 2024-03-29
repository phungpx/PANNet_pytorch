from torch import nn
from .resnet import ResNet
from .seg_head import FPEM_FFM


resnet_out_channels = {
    'resnet18': [64, 128, 256, 512],
    'resnet34': [64, 128, 256, 512],
    'resnet50': [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
    'resnet152': [256, 512, 1024, 2048],
}


class PANNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        backbone_pretrained: bool = False,
        num_FPEMs: int = 2,
        binary_threshold: float = 0.5,
        shrink_ratio: float = 0.5,
    ):
        super(PANNet, self).__init__()
        self.resnet = ResNet(
            backbone_name=backbone_name,
            pretrained=backbone_pretrained
        )

        self.seg_head = FPEM_FFM(
            backbone_out_channels=resnet_out_channels[backbone_name],
            num_FPEMs=num_FPEMs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2:]

        features = self.resnet(x)  # c2, c3, c4, c5
        x = self.seg_head(features)  # N x 6 x H / 4, W / 4
        x = nn.functional.interpolate(
            x, size=(H, W), mode='bilinear', align_corners=True
        )  # N x 6 x H x W

        return x

    def predict(self, x: torch.Tensor):
        with torch.no_grad():
            preds = self.forward(x)  # N x 6 x H x W
            preds[:2, :, :] = nn.Sigmoid()(preds[:2, :, :])

        images_boxes = []
        for pred in torch.split(preds, split_size_or_sections=1, dim=0):
            pred = pred.squeeze(dim=0).cpu().numpy()  # 6 x H x W
            image_boxes = self.get_boxes(pred=pred)
            images_boxes.append(image_boxes)

        return images_boxes

    def get_boxes(
        self, pred: np.ndarray, fx: float = 1, fy: float = 1, min_area: int = 5
    ) -> List[List[Tuple[float, float]]]:
        text_region = pred[0] > self.binary_threshold  # text region
        kernel = (pred[1] > self.binary_threshold) * text_region  # kernel of text region

        num_labels, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)

        boxes = []
        for i in range(1, num_labels):
            # find bounding box
            contours = cv2.findContours(np.uint8(label == i), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            if len(contours) != 1:
                raise RuntimeError('Found more than one contour in one connected component.')

            box = cv2.boxPoints(cv2.minAreaRect(contours[0])).tolist()
            if not Polygon(box).is_valid:
                # raise ValueError('must be valid polygon.')
                continue

            box = self.unshrink_polygon(box, r=self.shrink_ratio) 
            if not len(box):
                continue

            box = cv2.boxPoints(cv2.minAreaRect(np.array(box)))
            box = self.order_points(box)
            box = [(int(round(x * fx)), int(round(y * fy))) for x, y in box]

            boxes.append(box)

        return boxes

    def unshrink_polygon(self, points, r: float = 0.5):
        offseter = pyclipper.PyclipperOffset()

        poly = Polygon(points)
        d = poly.area * (1 + r) / poly.length

        points = [tuple(point) for point in points]
        offseter.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        polys = offseter.Execute(d)

        return polys

    def order_points(self, points: List[List[float]]) -> List[List[float]]:
        tl = min(points, key=lambda p: p[0] + p[1])
        br = max(points, key=lambda p: p[0] + p[1])
        tr = max(points, key=lambda p: p[0] - p[1])
        bl = min(points, key=lambda p: p[0] - p[1])

        return [tl, tr, br, bl]
