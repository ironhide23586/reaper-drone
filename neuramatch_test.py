"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|

Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""
import torch
import cv2
import numpy as np

from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()

from torchvision import transforms

from neural_matcher.nn import NeuraMatch


tensor_transform = transforms.ToTensor()

input_transforms = transforms.Compose([transforms.CenterCrop(3024),
                                       transforms.Resize(480),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

def drawMatches(img1, kp1, img2, kp2, matches):
    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1, rows2]),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = img1

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2] = img2

    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mi in range(matches.shape[0]):
        # mat = matches[mi]
        # Get the matching keypoints for each of the images
        # img1_idx = mat[:2]
        # img2_idx = mat[2:]

        # x - columns
        # y - rows
        (x1,y1) = kp1[mi]
        (x2,y2) = kp2[mi]

        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        cv2.circle(out, (int(x1), int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1, int(y2)), 4, (255, 0, 0), 1)

        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

    return out


if __name__ == '__main__':
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     x = torch.ones(1, device=device)
    #     print(x)
    # else:
    #     device = torch.device("cpu")
    #     print("MPS device not found."
    device = torch.device("cpu")

    ima = Image.open('scratchspace/IMG_3806.HEIC')
    imb = Image.open('scratchspace/IMG_3807.HEIC')

    im_a = tensor_transform(ima).to(device)
    im_b = tensor_transform(imb).to(device)

    t_a = torch.stack([input_transforms(im_a), input_transforms(im_a)])
    t_b = torch.stack([input_transforms(im_b), input_transforms(im_b)])

    nmatch = NeuraMatch()

    heatmap, masked_outs, unmasked_outs = nmatch(t_a, t_b)

    masked_matches, _ = masked_outs

    pxy_matches, conf_matches, desc_matches = masked_matches

    a = cv2.resize(np.array(ima), (480, 480))[:, :, [2, 1, 0]]
    b = cv2.resize(np.array(imb), (480, 480))[:, :, [2, 1, 0]]

    img3 = drawMatches(a, pxy_matches[0][:, :2].numpy(), b, pxy_matches[0][:, 2:].numpy(), pxy_matches[0].numpy())
    cv2.imwrite('im.png', img3)

    hm = (heatmap[0][0].detach().numpy() * 255).astype(np.uint8)
    cv2.imwrite('hm.png', hm)

    k = 0
