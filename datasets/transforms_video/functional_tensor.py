"""
https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py
Commit id: 24f16a3
"""

import torch
import torchvision.transforms.functional as F
from torch import Tensor
from torch.jit.annotations import Optional, List, BroadcastingList2, Tuple


def _is_tensor_a_torch_image(input):
    # [C, T, H, W]
    return len(input.shape) == 4


def vflip(img):
    # type: (Tensor) -> Tensor
    """Vertically flip the given the Image Tensor.
    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].
    Returns:
        Tensor:  Vertically flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-2)


def hflip(img):
    # type: (Tensor) -> Tensor
    """Horizontally flip the given the Image Tensor.
    Args:
        img (Tensor): Image Tensor to be flipped in the form [C, H, W].
    Returns:
        Tensor:  Horizontally flipped image Tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return img.flip(-1)


def crop(img, top, left, height, width):
    # type: (Tensor, int, int, int, int) -> Tensor
    """Crop the given Image Tensor.
    Args:
        img (Tensor): Image to be cropped in the form [C, H, W]. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        Tensor: Cropped image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    # Crop makes it not contiguous
    region = img[..., top:top + height, left:left + width]
    return region.contiguous()


def to_grayscale(img, num_output_channels=3):
    # type: (Tensor, int) -> Tensor
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
    Returns:
        Tensor: Grayscale image.
    """
    if img.shape[0] != 3:
        raise TypeError('Input Image does not contain 3 Channels')

    grey = (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).to(img.dtype)

    if num_output_channels == 1:
        return grey
    elif num_output_channels == 3:
        return grey.expand_as(img)
    else:
        raise ValueError('num_output_channels should be either 1 or 3')


@torch.jit.script
def rgb_to_grayscale(img: Tensor) -> Tensor:
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W]. [C, T, H, W]
    Returns:
        Tensor: Grayscale image.
    """
    grey = (0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]).to(img.dtype)
    # result = torch.stack([grey, grey, grey], dim=0)
    return grey.expand_as(img).contiguous()


def _blend(img1, img2, ratio):
    # type: (Tensor, Tensor, float) -> Tensor
    bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64] else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)


@torch.jit.script
def adjust_brightness(img, brightness_factor):
    # type: (Tensor, float) -> Tensor
    """Adjust brightness of an RGB image.
    Args:
        img (Tensor): Image to be adjusted.
        brightness_factor (float):  How much to adjust the brightness. Can be
            any non negative number. 0 gives a black image, 1 gives the
            original image while 2 increases the brightness by a factor of 2.
    Returns:
        Tensor: Brightness adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, torch.zeros_like(img), brightness_factor)


@torch.jit.script
def adjust_contrast(img, contrast_factor):
    # type: (Tensor, float) -> Tensor
    """Adjust contrast of an RGB image.
    Args:
        img (Tensor): Image to be adjusted.
        contrast_factor (float): How much to adjust the contrast. Can be any
            non negative number. 0 gives a solid gray image, 1 gives the
            original image while 2 increases the contrast by a factor of 2.
    Returns:
        Tensor: Contrast adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    mean = torch.mean(rgb_to_grayscale(img).to(torch.float))

    return _blend(img, mean, contrast_factor)


@torch.jit.script
def adjust_saturation(img, saturation_factor):
    # type: (Tensor, float) -> Tensor
    """Adjust color saturation of an RGB image.
    Args:
        img (Tensor): Image to be adjusted.
        saturation_factor (float):  How much to adjust the saturation. 0 will
            give a black and white image, 1 will give the original image while
            2 will enhance the saturation by a factor of 2.
    Returns:
        Tensor: Saturation adjusted image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    return _blend(img, rgb_to_grayscale(img), saturation_factor)


def center_crop(img, output_size):
    # type: (Tensor, BroadcastingList2[int]) -> Tensor
    """Crop the Image Tensor and resize it to desired size.
    Args:
        img (Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
        output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
    Returns:
            Tensor: Cropped image.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    _, image_width, image_height = img.size()
    crop_height, crop_width = output_size
    crop_top = int(round((image_height - crop_height) / 2.))
    crop_left = int(round((image_width - crop_width) / 2.))

    return crop(img, crop_top, crop_left, crop_height, crop_width)


def five_crop(img, size):
    # type: (Tensor, BroadcastingList2[int]) -> List[Tensor]
    """Crop the given Image Tensor into four corners and the central crop.
    .. Note::
        This transform returns a List of Tensors and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
           int instead of sequence like (h, w), a square crop (size, size) is
           made.
    Returns:
       List: List (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    assert len(size) == 2, "Please provide only two dimensions (h, w) for size."

    _, image_width, image_height = img.size()
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    tl = crop(img, 0, 0, crop_width, crop_height)
    tr = crop(img, image_width - crop_width, 0, image_width, crop_height)
    bl = crop(img, 0, image_height - crop_height, crop_width, image_height)
    br = crop(img, image_width - crop_width, image_height - crop_height, image_width, image_height)
    center = center_crop(img, (crop_height, crop_width))

    return [tl, tr, bl, br, center]


def ten_crop(img, size, vertical_flip=False):
    # type: (Tensor, BroadcastingList2[int], bool) -> List[Tensor]
    """Crop the given Image Tensor into four corners and the central crop plus the
        flipped version of these (horizontal flipping is used by default).
    .. Note::
        This transform returns a List of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.
    Args:
       size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
       vertical_flip (bool): Use vertical flipping instead of horizontal
    Returns:
       List: List (tl, tr, bl, br, center, tl_flip, tr_flip, bl_flip, br_flip, center_flip)
                Corresponding top left, top right, bottom left, bottom right and center crop
                and same for the flipped image's tensor.
    """
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')

    assert len(size) == 2, "Please provide only two dimensions (h, w) for size."
    first_five = five_crop(img, size)

    if vertical_flip:
        img = vflip(img)
    else:
        img = hflip(img)

    second_five = five_crop(img, size)

    return first_five + second_five


@torch.jit.script
def hsv_to_rgb(image: Tensor) -> Tensor:
    r"""Convert an HSV image to RGB
    https://torchgeometry.readthedocs.io/en/latest/_modules/kornia/color/hsv.html#rgb_to_hsv
    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.


    Returns:
        torch.Tensor: HSV version of the image.
    """

    # if not torch.is_tensor(image):
    #     raise TypeError("Input type is not a torch.Tensor. Got {}".format(
    #         type(image)))

    if len(image.shape) != 4 or image.shape[0] != 3:
        raise ValueError("Input size must have a shape of (3, T, H, W). Got {}"
                         .format(image.shape))

    # h: torch.Tensor = image[..., 0, :, :]
    # s: torch.Tensor = image[..., 1, :, :]
    # v: torch.Tensor = image[..., 2, :, :]
    flatten_image = image.view(3, -1)
    h: torch.Tensor = flatten_image[0]
    s: torch.Tensor = flatten_image[1]
    v: torch.Tensor = flatten_image[2]

    hi: torch.Tensor = torch.floor(h * 6)
    f: torch.Tensor = h * 6 - hi

    vtpq = torch.empty(4, flatten_image.size(-1), device=flatten_image.device)
    vtpq[0] = v
    vtpq[1] = v * (1 - (1 - f) * s)
    vtpq[2] = v * (1 - s)
    vtpq[3] = v * (1 - f * s)

    index: torch.Tensor = hi.to(dtype=torch.long) % 6
    channel_map = torch.tensor([
        [0, 3, 2, 2, 1, 0],
        [1, 0, 0, 3, 2, 2],
        [2, 2, 1, 0, 0, 3],
    ], dtype=torch.long, device=index.device)
    gather_index = torch.gather(channel_map, dim=1, index=index.expand_as(flatten_image))

    return torch.gather(vtpq, dim=0, index=gather_index).view_as(image)


@torch.jit.script
def rgb_to_hsv(image: Tensor):
    r"""Convert an RGB image to HSV.

    Args:
        input (torch.Tensor): RGB Image to be converted to HSV.

    Returns:
        torch.Tensor: HSV version of the image.
    """

    # if not torch.is_tensor(image):
    #     raise TypeError("Input type is not a torch.Tensor. Got {}".format(
    #         type(image)))

    if len(image.shape) != 4 or image.shape[0] != 3:
        raise ValueError("Input size must have a shape of (3, T, H, W). Got {}"
                         .format(image.shape))

    flatten_image = image.view(3, -1)
    r: torch.Tensor = flatten_image[0]
    g: torch.Tensor = flatten_image[1]
    b: torch.Tensor = flatten_image[2]

    maxc, indices = flatten_image.max(0)
    minc: torch.Tensor = flatten_image.min(0).values

    v: torch.Tensor = maxc  # brightness

    deltac: torch.Tensor = maxc - minc
    s: torch.Tensor = torch.where(v == 0, torch.zeros(1, device=v.device), deltac / v)  # saturation

    all_h = torch.empty_like(flatten_image)
    all_h[0] = (g - b) / deltac
    all_h[1] = (b - r) / deltac + 2.0
    all_h[2] = (r - g) / deltac + 4.0

    h = torch.gather(all_h, dim=0, index=indices.unsqueeze(0)).squeeze(0)
    h.masked_fill_(deltac == 0, 0.0)

    h: torch.Tensor = (h / 6.0) % 1.0

    return torch.stack([h, s, v], dim=0).view_as(image)


def rgb_to_hsv1(arr: Tensor) -> Tensor:
    out = torch.zeros_like(arr)
    arr_max = arr.max(0).values
    arr_min = arr.min(0).values
    ipos = arr_max > 0
    delta = arr_max - arr_min

    s = torch.zeros_like(delta)
    s[ipos] = delta[ipos] / arr_max[ipos]
    ipos = delta > 0
    # red is max
    idx = (arr[0] == arr_max) & ipos
    out[0, idx] = (arr[1, idx] - arr[2, idx]) / delta[idx]
    # green is max
    idx = (arr[1] == arr_max) & ipos
    out[0, idx] = 2. + (arr[2, idx] - arr[0, idx]) / delta[idx]
    # blue is max
    idx = (arr[2] == arr_max) & ipos
    out[0, idx] = 4. + (arr[0, idx] - arr[1, idx]) / delta[idx]

    out[0] = (out[0] / 6.0) % 1.0
    out[1] = s
    out[2] = arr_max

    return out


@torch.jit.script
def adjust_hue(img: Tensor, hue_factor: float) -> Tensor:
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    See `Hue`_ for more details.
    .. _Hue: https://en.wikipedia.org/wiki/Hue
    Args:
        img (PIL Image): PIL Image to be adjusted.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
        PIL Image: Hue adjusted image.
    """
    if not (-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor is not in [-0.5, 0.5].'.format(hue_factor))

    # Grayscale image cannot be changed
    input_channels = img.shape[0]
    if input_channels == 1:
        return img

    hsv = rgb_to_hsv(img)

    # uint8 addition take cares of rotation across boundaries
    # with np.errstate(over='ignore'):
    #     np_h += np.uint8(hue_factor * 255)
    # h = Image.fromarray(np_h, 'L')
    # hsv[0] += hue_factor * 255
    hsv[0] = (hsv[0] + hue_factor) % 1.0

    # img = Image.merge('HSV', (h, s, v)).convert(input_mode)
    img = hsv_to_rgb(hsv)

    return img


# =============================== Gaussian Start

def gaussian(window_size, sigma):
    def gauss_fcn(x):
        return -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)

    gauss = torch.stack(
        [torch.exp(torch.tensor(gauss_fcn(x))) for x in range(window_size)])
    return gauss / gauss.sum()


def get_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Function that returns Gaussian filter coefficients.

    Args:
        kernel_size (int): filter size. It should be odd and positive.
        sigma (float): gaussian standard deviation.
        device:
    Returns:
        Tensor: 1D tensor with gaussian filter coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size})`

    Examples::

        >>> kornia.image.get_gaussian_kernel(3, 2.5)
        tensor([0.3243, 0.3513, 0.3243])

        >>> kornia.image.get_gaussian_kernel(5, 1.5)
        tensor([0.1201, 0.2339, 0.2921, 0.2339, 0.1201])
    """
    if not isinstance(kernel_size, int) or kernel_size % 2 == 0 or \
            kernel_size <= 0:
        raise TypeError("kernel_size must be an odd positive integer. "
                        "Got {}".format(kernel_size))
    window_1d: torch.Tensor = gaussian(kernel_size, sigma)
    return window_1d


def get_gaussian_kernel2d(kernel_size: Tuple[int, int],
                          sigma: Tuple[float, float]) -> torch.Tensor:
    r"""Function that returns Gaussian filter matrix coefficients.

    Args:
        kernel_size (Tuple[int, int]): filter sizes in the x and y direction.
         Sizes should be odd and positive.
        sigma (Tuple[int, int]): gaussian standard deviation in the x and y
         direction.
        device
    Returns:
        Tensor: 2D tensor with gaussian filter matrix coefficients.

    Shape:
        - Output: :math:`(\text{kernel_size}_x, \text{kernel_size}_y)`

    Examples::

        >>> kornia.image.get_gaussian_kernel2d((3, 3), (1.5, 1.5))
        tensor([[0.0947, 0.1183, 0.0947],
                [0.1183, 0.1478, 0.1183],
                [0.0947, 0.1183, 0.0947]])

        >>> kornia.image.get_gaussian_kernel2d((3, 5), (1.5, 1.5))
        tensor([[0.0370, 0.0720, 0.0899, 0.0720, 0.0370],
                [0.0462, 0.0899, 0.1123, 0.0899, 0.0462],
                [0.0370, 0.0720, 0.0899, 0.0720, 0.0370]])
    """
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of length two. Got {}"
                        .format(kernel_size))
    if not isinstance(sigma, tuple) or len(sigma) != 2:
        raise TypeError("sigma must be a tuple of length two. Got {}"
                        .format(sigma))
    ksize_x, ksize_y = kernel_size
    sigma_x, sigma_y = sigma
    kernel_x: torch.Tensor = get_gaussian_kernel(ksize_x, sigma_x)
    kernel_y: torch.Tensor = get_gaussian_kernel(ksize_y, sigma_y)
    kernel_2d: torch.Tensor = torch.matmul(
        kernel_x.unsqueeze(-1), kernel_y.unsqueeze(-1).t())
    return kernel_2d

# =============================== Gaussian End
