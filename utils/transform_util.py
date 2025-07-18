import numpy as np
import torch as th
import polarTransform
import operator
import random
import numpy as np
import cv2
from skimage.transform import resize
from scipy.ndimage import rotate, shift
# -------- FFT transform --------

def fftc_np(image):
    """
    Orthogonal FFT2 transform image to kspace data, numpy.array to numpy.array.

    :param image: numpy.array of complex with shape of (h, w), mri image.

    :return: numpy.array of complex with shape of (h, w), kspace data with center low-frequency, keep dtype.
    """
    kspace = np.fft.fftshift(np.fft.fft2(image, norm="ortho"))
    kspace = kspace.astype(image.dtype)
    return kspace


def ifftc_np(kspace):
    """
    Inverse orthogonal FFT2 transform kspace data to image, numpy.array to numpy.array.

    :param kspace: numpy.array of complex with shape of (h, w) and center low-frequency, kspace data.

    :return: numpy.array of complex with shape of (h, w), transformed image, keep dtype.
    """
    image = np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho")
    image = image.astype(kspace.dtype)
    return image


def fftc_th(image):
    """
    Orthogonal FFT2 transform image to kspace data, th.Tensor to th.Tensor.

    :param image: th.Tensor of real with shape of (..., 2, h, w), mri image.

    :return: th.Tensor of real with shape of (..., 2, h, w), kspace data with center low-frequency, keep dtype.
    """
    image = image.permute(0, 2, 3, 1).contiguous()
    kspace = th.fft.fftshift(th.fft.fft2(th.view_as_complex(image), norm="ortho"), dim=(-1, -2))
    kspace = th.view_as_real(kspace).permute(0, 3, 1, 2).contiguous()
    return kspace


def ifftc_th(kspace):
    """
    Inverse orthogonal FFT2 transform kspace data to image, th.Tensor to th.Tensor.

    :param kspace: th.Tensor of real with shape of (..., 2, h, w), kspace data with center low-frequency.

    :return: th.Tensor of real with shape of (..., 2, h, w), mri image, keep dtype.
    """
    kspace = kspace.permute(0, 2, 3, 1).contiguous()
    image = th.fft.ifft2(th.fft.ifftshift(th.view_as_complex(kspace), dim=(-1, -2)), norm="ortho")
    image = th.view_as_real(image).permute(0, 3, 1, 2).contiguous()
    return image


# -------- dtype transform --------

def complex2real_np(x):
    """
    Change a complex numpy.array to a real array with two channels.

    :param x: numpy.array of complex with shape of (h, w).

    :return: numpy.array of real with shape of (2, h, w).
    """
    return np.stack([x.real, x.imag])


def real2complex_np(x):
    """
    Change a real numpy.array with two channels to a complex array.

    :param x: numpy.array of real with shape of (2, h, w).

    :return: numpy.array of complex64 with shape of (h, w).
    """
    complex_x = np.zeros_like(x[0, ...], dtype=np.complex64)
    complex_x.real, complex_x.imag = x[0], x[1]
    return complex_x


def np2th(x):
    return th.tensor(x)


def th2np(x):
    return x.detach().cpu().numpy()


def np_comlex_to_th_real2c(x):
    """
    Transform numpy.array of complex to th.Tensor of real with 2 channels.

    :param x: numpy.array of complex with shape of (h, w).

    :return: th.Tensor of real with 2 channels with shape of (h, w, 2).
    """
    return np2th(complex2real_np(x).transpose((1, 2, 0)))


def th_real2c_to_np_complex(x):
    """
    Transform th.Tensor of real with 2 channels to numpy.array of complex.

    :param x: th.Tensor of real with 2 channels with shape of (h, w, 2).

    :return: numpy.array of complex with shape of (h, w).
    """
    return real2complex_np(th2np(x.permute(2, 0, 1)))


def th2np_magnitude(x):
    """
    Compute the magnitude of torch.Tensor with shape of (b, 2, h, w).

    :param x: th.Tensor of real with 2 channels with shape of (b, 2, h, w).

    :return: numpy.array of real with shape of (b, h, w).
    """
    x = th2np(x)
    return np.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)


def pad_to_pool(x: th.tensor, num_layer: int, step_scale: int) -> th.tensor:
    """
    reshape a tensor so that it's dimension fits to up/down pool by given number of layers

    :param x: input tensor, expected shape [D, H, W, ...]
    :param num_layer: number of layers to up/down pool
    :param step_scale: scale coefficient for each up/down scale
    :return: x_rescale, new tensor with shape [D, H_new, W_new, ...], pad extra elements with zero
    """
    H = x.shape[1]
    W = x.shape[2]

    H_new = (step_scale ** num_layer) * round(H / (step_scale ** num_layer))
    W_new = (step_scale ** num_layer) * round(W / (step_scale ** num_layer))

    x_shape = list(x.shape)
    x_shape[1] = H_new
    x_shape[2] = W_new
    x_rescale = th.zeros(x_shape)

    # pad given tensor slightly to make it down-poolable
    x_rescale[:, H_new // 2 - min(H_new, H) // 2: H_new // 2 + min(H_new, H) // 2,
    W_new // 2 - min(W_new, W) // 2: W_new // 2 + min(W_new, W) // 2, :] \
        = x[:, H // 2 - min(H_new, H) // 2: H // 2 + min(H_new, H) // 2,
          W // 2 - min(W_new, W) // 2: W // 2 + min(W_new, W) // 2, :]

    return x_rescale


def normalize_one_to_one(x: th.tensor) -> th.tensor:
    """
    Args: normalize tensor value to range of [-1, 1]
    :param x: input tensor
    :return: x_norm: normalized input tensor
    """
    x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1

    return x_norm, x.max(), x.min()


def normalize_zero_to_one(x: th.tensor) -> th.tensor:
    """
    Args: normalize tensor value to range of [0, 1]
    :param x: input tensor
    :return: x_norm: normalized input tensor
    """
    x_norm = (x - x.min()) / (x.max() - x.min()) + 0.0

    return x_norm, x.max(), x.min()


def denormalize_one_to_one(x_norm: th.tensor, x_max, x_min) -> th.tensor:
    """
    Args: denormalize tensor value from range of [-1, 1]

    :param x_norm: input normalized tensor
    :param x_max: upper bound of x
    :param x_min: lower bound of x
    :return: x: denormalized input tensor
    """
    x = (x_norm + 1) * (x_max - x_min) / 2 + x_min

    return x


def denormalize_zero_to_one(x_norm: th.tensor, x_max, x_min) -> th.tensor:
    """
    Args: denormalize tensor value from range of [0, 1]

    :param x_norm: input normalized tensor
    :param x_max: upper bound of x
    :param x_min: lower bound of x
    :return: x: denormalized input tensor
    """
    x = x_norm * (x_max - x_min) + x_min + 0.0

    return x


def add_gaussian_noise(x: th.tensor, coef) -> th.tensor:
    """
    Args: add gaussian noise with scale coef to img, img should have scale [-1, 1]

    :param x: input one_to_one normalized image tensor [C, H, W]
    :param coef: scale coefficient for gaussian noise
    :return: x_noisy: gaussian noise output tensor [C, H, W]
    """

    noise = th.randn_like(x)
    x_noisy = x + coef * noise

    return x_noisy


def cartersianToPolar(img, order=0):
    """
    Args:
        img: [H, W, d] input image tensor
        order : :class:`int` (0-5), optional
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        The following orders have special names:

            * 0 - nearest neighbor
            * 1 - bilinear
            * 3 - bicubic

    returns:
        polarImage: [H, W, d] polar image of input image
        ptSettings: recorded settings during transform

    """

    H, W, d = img.shape

    # initialize first channel, [H, W]
    polarImage, ptSettings = polarTransform.convertToPolarImage(img[:, :, 0], order=order)
    polarImage = th.tensor(polarImage).unsqueeze(-1)  # [H, W, 1]

    for i in range(d):
        if i > 0:
            # [H, W]
            polarImageTemp, _ = polarTransform.convertToPolarImage(img[:, :, i], order=order)
            polarImage = th.cat((polarImage, th.tensor(polarImageTemp).unsqueeze(-1)), -1)  # [H, W, d]

    return polarImage, ptSettings


def polarToCartersian(polarImage, order=0):
    """
    Args:
        img: [H, W, d] input image tensor
        order : :class:`int` (0-5), optional
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        The following orders have special names:

            * 0 - nearest neighbor
            * 1 - bilinear
            * 3 - bicubic

    returns:
        cartesianImage: [H, W, d] polar image of input image
        ptSettings: recorded settings during transform

    """

    H, W, d = polarImage.shape

    # initialize first channel, [H, W]
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(polarImage[:, :, 0], order=order)
    cartesianImage = th.tensor(cartesianImage).unsqueeze(-1)  # [H, W, 1]

    for i in range(d):
        if i > 0:
            # [H, W]
            cartesianImageTemp, _ = polarTransform.convertToCartesianImage(polarImage[:, :, i], order=order)
            cartesianImage = th.cat((cartesianImage, th.tensor(cartesianImageTemp).unsqueeze(-1)),
                                    -1)  # [H, W, d]

    return cartesianImage, ptSettings


def polarToCartersian_given_setting(polarImage, ptSettings, order=0):
    """
    Args:
        polarImage: [H, W, d] input image tensor
        ptSettings: setting recorded from cartersianToPolar
        order : :class:`int` (0-5), optional
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        The following orders have special names:

            * 0 - nearest neighbor
            * 1 - bilinear
            * 3 - bicubic

    returns:
        cartesianImage: [H, W, d] cartesian image of input image

    """

    H, W, d = polarImage.shape

    # initialize first channel, [H, W]
    cartesianImage = ptSettings.convertToCartesianImage(polarImage[:, :, 0], order=0)
    cartesianImage = th.tensor(cartesianImage).unsqueeze(-1)  # [H, W, 1]

    for i in range(d):
        if i > 0:
            # [H, W]
            cartesianImageTemp = ptSettings.convertToCartesianImage(polarImage[:, :, i], order=order)
            cartesianImage = th.cat((cartesianImage, th.tensor(cartesianImageTemp).unsqueeze(-1)),
                                    -1)  # [H, W, d]

    return cartesianImage


def center_crop_np(img, bounding):
    """
    Args:
        img: 2d or 3d numpy
        bounding: input tuple show center_crop size (H, W)

    Returns:

    """
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


def center_crop_with_pad(x: th.tensor, center_crop_h, center_crop_w) -> th.tensor:
    """
    reshape a tensor so that it's dimension fits to up/down pool by given number of layers

    :param x: input tensor, expected shape [D, H, W, ...]
    :param num_layer: number of layers to up/down pool
    :param step_scale: scale coefficient for each up/down scale
    :return: x_rescale, new tensor with shape [D, H_new, W_new, ...], pad extra elements with zero
    """
    H = x.shape[1]
    W = x.shape[2]

    H_new = center_crop_h
    W_new = center_crop_w

    x_shape = list(x.shape)
    x_shape[1] = H_new
    x_shape[2] = W_new
    x_rescale = th.zeros(x_shape)

    # pad given tensor slightly to make it down-poolable
    x_rescale[:, H_new // 2 - min(H_new, H) // 2: H_new // 2 + min(H_new, H) // 2,
    W_new // 2 - min(W_new, W) // 2: W_new // 2 + min(W_new, W) // 2] \
        = x[:, H // 2 - min(H_new, H) // 2: H // 2 + min(H_new, H) // 2,
          W // 2 - min(W_new, W) // 2: W // 2 + min(W_new, W) // 2]

    return x_rescale


def center_crop_with_pad_np(x, center_crop_h, center_crop_w):
    """
    reshape a tensor so that it's dimension fits to up/down pool by given number of layers

    :param x: input tensor, expected shape [H, W]
    :param num_layer: number of layers to up/down pool
    :param step_scale: scale coefficient for each up/down scale
    :return: x_rescale, new tensor with shape [D, H_new, W_new, ...], pad extra elements with zero
    """
    H = x.shape[0]
    W = x.shape[1]

    H_new = center_crop_h
    W_new = center_crop_w

    x_shape = list(x.shape)
    x_shape[0] = H_new
    x_shape[1] = W_new
    x_rescale = np.zeros(x_shape)

    # pad given tensor slightly to make it down-poolable
    x_rescale[H_new // 2 - min(H_new, H) // 2: H_new // 2 + min(H_new, H) // 2,
    W_new // 2 - min(W_new, W) // 2: W_new // 2 + min(W_new, W) // 2] \
        = x[H // 2 - min(H_new, H) // 2: H // 2 + min(H_new, H) // 2,
          W // 2 - min(W_new, W) // 2: W // 2 + min(W_new, W) // 2]

    return x_rescale


def rgb2gray(rgb):
    """
    Args:
        rgb: 3d numpy [H, W, 3]

    Returns:
        gray scale image: [H, W]
    """
    H, W, d = rgb.shape
    if d == 3:
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        raise Exception("rgb  channel does not match with: " + str(rgb.shape))


# Defining a custom transformer class
class CustomRandomFlip(object):
    # random flip images with shape [H, W] or [C, H, W] transform function
    #
    def __init__(self, flip_prob_horizon=0.5, flip_prob_vertical=0.5):
        self.flip_prob_horizon = flip_prob_horizon
        self.flip_prob_vertical = flip_prob_vertical
        # Defining the transform method

    def __call__(self, image_pre, image_tg):
        # flip image, alwasy check horizon first then vertical

        if np.random.random() < self.flip_prob_horizon:
            if len(image_pre.shape) == 2:
                image_pre = np.flip(image_pre, axis=1)
            if len(image_pre.shape) == 3:
                image_pre = np.flip(image_pre, axis=2)
            # image_tg = np.flip(image_tg, axis=1)
            if len(image_tg.shape) == 2:
                image_tg = np.flip(image_tg, axis=1)
            if len(image_tg.shape) == 3:
                image_tg = np.flip(image_tg, axis=2)
        if np.random.random() < self.flip_prob_vertical:
            # image_pre = np.flip(image_pre, axis=0)
            if len(image_pre.shape) == 2:
                image_pre = np.flip(image_pre, axis=0)
            if len(image_pre.shape) == 3:
                image_pre = np.flip(image_pre, axis=1)
            # image_tg = np.flip(image_tg, axis=0)
            if len(image_tg.shape) == 2:
                image_tg = np.flip(image_tg, axis=0)
            if len(image_tg.shape) == 3:
                image_tg = np.flip(image_tg, axis=1)
            # Returning the two parts of the image
        return image_pre, image_tg


# Defining a custom transformer class
class CustomRandomRot(object):
    # random rot images clock or counter clock 90 degree with shape [H, W] or [C, H, W] transform function
    def __init__(self, rot_prob_clock=0.5, rot_prob_count=0.5):
        self.rot_prob_clock = rot_prob_clock
        self.rot_prob_count = rot_prob_count
        # Defining the transform method

    def __call__(self, image_pre, image_tg):
        # flip image, alwasy check horizon first then vertical
        if np.random.random() < self.rot_prob_clock:
            # image_pre = np.rot90(image_pre, k=1, axes=(1, 0))
            if len(image_pre.shape) == 2:
                image_pre = np.rot90(image_pre, k=1, axes=(1, 0))
            if len(image_pre.shape) == 3:
                image_pre = np.rot90(image_pre, k=1, axes=(2, 1))
            # image_tg = np.rot90(image_tg, k=1, axes=(1, 0))
            if len(image_tg.shape) == 2:
                image_tg = np.rot90(image_tg, k=1, axes=(1, 0))
            if len(image_tg.shape) == 3:
                image_tg = np.rot90(image_tg, k=1, axes=(2, 1))
        if np.random.random() < self.rot_prob_count:
            # image_pre = np.rot90(image_pre, k=1, axes=(0, 1))
            if len(image_pre.shape) == 2:
                image_pre = np.rot90(image_pre, k=1, axes=(0, 1))
            if len(image_pre.shape) == 3:
                image_pre = np.rot90(image_pre, k=1, axes=(1, 2))
            # image_tg = np.rot90(image_tg, k=1, axes=(0, 1))
            if len(image_tg.shape) == 2:
                image_tg = np.rot90(image_tg, k=1, axes=(0, 1))
            if len(image_tg.shape) == 3:
                image_tg = np.rot90(image_tg, k=1, axes=(1, 2))
            # Returning the two parts of the image
        return image_pre, image_tg


class CustomRandomRotDegree(object):
    # random rot images clock or counter clock 90 degree with shape [H, W] or [C, H, W] transform function
    def __init__(self, rot_prob_clock=0.5, rot_prob_count=0.5):
        self.rot_prob_clock = rot_prob_clock
        self.rot_prob_count = rot_prob_count
        # Defining the transform method

    def __call__(self, image_pre, image_tg):
        # flip image, alwasy check horizon first then vertical
        if np.random.random() < self.rot_prob_clock:
            # image_pre = np.rot90(image_pre, k=1, axes=(1, 0))
            if len(image_pre.shape) == 2:
                image_pre = np.rot90(image_pre, k=1, axes=(1, 0))
            if len(image_pre.shape) == 3:
                image_pre = np.rot90(image_pre, k=1, axes=(2, 1))
            # image_tg = np.rot90(image_tg, k=1, axes=(1, 0))
            if len(image_tg.shape) == 2:
                image_tg = np.rot90(image_tg, k=1, axes=(1, 0))
            if len(image_tg.shape) == 3:
                image_tg = np.rot90(image_tg, k=1, axes=(2, 1))
        if np.random.random() < self.rot_prob_count:
            # image_pre = np.rot90(image_pre, k=1, axes=(0, 1))
            if len(image_pre.shape) == 2:
                image_pre = np.rot90(image_pre, k=1, axes=(0, 1))
            if len(image_pre.shape) == 3:
                image_pre = np.rot90(image_pre, k=1, axes=(1, 2))
            # image_tg = np.rot90(image_tg, k=1, axes=(0, 1))
            if len(image_tg.shape) == 2:
                image_tg = np.rot90(image_tg, k=1, axes=(0, 1))
            if len(image_tg.shape) == 3:
                image_tg = np.rot90(image_tg, k=1, axes=(1, 2))
            # Returning the two parts of the image
        return image_pre, image_tg


def find_mask_bounds(mask):
    # Collapse the 3D mask into a 2D mask where any non-zero value across channels is considered
    if len(mask.shape) == 3:
        collapsed_mask = np.any(mask, axis=0)
    else:
        collapsed_mask = mask

    # Find rows and columns where mask has non-zero values
    rows = np.any(collapsed_mask, axis=1)
    cols = np.any(collapsed_mask, axis=0)

    # Get top and bottom boundaries
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1]) - 1

    # Get left and right boundaries
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1]) - 1

    return top, bottom, left, right





def resize_image_proportionally(image_array, target_size, interpolateion=cv2.INTER_NEAREST):
    shape = image_array.shape  # [H, W]

    if target_size[0] / shape[0] <= target_size[1] / shape[1]:  # H > W
        sizeH = target_size[0]
        sizeW = int(shape[1] * target_size[0] / shape[0])
    else:
        sizeW = target_size[1]
        sizeH = int(shape[0] * target_size[1] / shape[1])

    try:
        res = cv2.resize(image_array, dsize=(sizeW, sizeH), interpolation=interpolateion)
    except:
        raise Exception('resize_image_propotionally failed')
    return res


def check_if_only_0_1(image_array):
    # Check if any element is not 0 or 1
    has_non_binary = np.any((image_array != 0) & (image_array != 1))

    if has_non_binary:
        print("The array contains elements other than 0 or 1.")
        return False
    else:
        print("The array contains only 0 and 1 values.")
        return True


def crop_and_resize(image_array, crop_coords, target_size):
    """Crops and resizes a NumPy array representing an image.

    Args:
        image_array: The NumPy array representing the image.
        crop_coords: A tuple of (y1, y2, x1, x2) coordinates for cropping.
        target_size: A tuple of (width, height) for the target size.

    Returns:
        The cropped and resized image array.
    """

    # Crop the image
    y1, y2, x1, x2 = crop_coords
    if len(image_array.shape) == 2:
        cropped_image = image_array[y1:y2, x1:x2]
        resized_image = resize_image_proportionally(cropped_image, target_size)

        # pad the given size
        sizeH = resized_image.shape[0]
        sizeW =resized_image.shape[1]
        extra_left = int((target_size[1] - sizeW) / 2)
        extra_right = target_size[1] - sizeW - extra_left
        extra_top = int((target_size[0] - sizeH) / 2)
        extra_bottom = target_size[0] - sizeH - extra_top
        resized_image = np.pad(resized_image, ((extra_top, extra_bottom), (extra_left, extra_right)),
                         mode='constant', constant_values=0)

    elif len(image_array.shape) == 3:
        resized_image = np.zeros((image_array.shape[0], target_size[0], target_size[1]))
        for i in range(len(image_array)):
            cropped_image_i = image_array[i, y1:y2, x1:x2]
            resized_image_i = resize_image_proportionally(cropped_image_i, target_size)
            # print_var_detail(resized_image_i)

            # pad the given size
            sizeH = resized_image_i.shape[0]
            sizeW =resized_image_i.shape[1]
            extra_left = int((target_size[1] - sizeW) / 2)
            extra_right = target_size[1] - sizeW - extra_left
            extra_top = int((target_size[0] - sizeH) / 2)
            extra_bottom = target_size[0] - sizeH - extra_top
            resized_image_i = np.pad(resized_image_i, ((extra_top, extra_bottom), (extra_left, extra_right)),
                             mode='constant', constant_values=0)
            resized_image[i] = resized_image_i
    else:
        raise ValueError('image_array must be 2 or 3 dimensional')

    # Resize the image
    return resized_image


def random_crop_given_bounds(image_array, bounds, target_size, random_crop_margin_v = (0, 0), random_crop_margin_h = (0,0)):
    top, bottom, left, right = bounds

    # random extra margin around bounds
    top = np.clip(top - random_crop_margin_v[0], 0, 16384) # default max is 16K
    bottom = np.clip(bottom + random_crop_margin_v[1], 0, 16384)
    left = np.clip(left - random_crop_margin_h[0], 0, 16384)
    right = np.clip(right + random_crop_margin_h[1], 0, 16384)

    # set crop coords given random extra margin
    max_interval = max(bottom - top, right - left)
    max_interval = max_interval // 2
    crop_coords = (
    (top + bottom) // 2 - max_interval, (top + bottom) // 2 + max_interval, (right + left) // 2 - max_interval,
    (right + left) // 2 + max_interval)  # Example crop coordinates
    arr = np.array(crop_coords)
    # Clip the array
    clipped_arr = np.clip(arr, 0, 16384)
    # Convert back to tuple
    crop_coords = tuple(clipped_arr)

    image_array_crop = crop_and_resize(image_array, crop_coords, target_size)
    return image_array_crop





def rotate_around_axis_position(image, angle, axis, position, order=0):
    # Calculate the center of the array along each dimension
    center = np.array(image.shape) / 2

    # Calculate the shift to bring the specific position to the center
    shift_to_center = center - np.array(position)

    # Translate the array to bring the desired rotation position to the center
    translated_image = shift(image, shift=shift_to_center, order=order)

    # Rotate the translated mask around the specified axis
    if axis == 0:
        rotated_image = rotate(translated_image, angle=angle, axes=(1, 2), reshape=False, order=order)
    elif axis == 1:
        rotated_image = rotate(translated_image, angle=angle, axes=(0, 2), reshape=False, order=order)
    elif axis == 2:
        rotated_image = rotate(translated_image, angle=angle, axes=(0, 1), reshape=False, order=order)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")

    # Translate back to the original position
    final_image = shift(rotated_image, shift=-shift_to_center, order=order)

    # Threshold to ensure binary values
    if order == 0:
        final_image = (final_image > 0.5).astype(np.uint8)

    return final_image


def rotate_3d_slices_position(mask, angle, axis, position, flags):
    rotated_slices = []

    # Rotate each slice along the specified axis
    if axis == 0:  # Rotate each depth slice
        for i in range(mask.shape[0]):
            position_0 = (position[1], position[2])
            rotation_matrix = cv2.getRotationMatrix2D(position_0, angle, 1)
            rotated_slice = cv2.warpAffine(mask[i], rotation_matrix, (mask.shape[2], mask.shape[1]), flags=flags)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=0)

    elif axis == 1:  # Rotate each height slice
        for i in range(mask.shape[1]):
            position_1 = (position[0], position[2])
            rotation_matrix = cv2.getRotationMatrix2D(position_1, angle, 1)
            rotated_slice = cv2.warpAffine(mask[:, i, :], rotation_matrix, (mask.shape[2], mask.shape[0]), flags=flags)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=1)

    elif axis == 2:  # Rotate each width slice
        for i in range(mask.shape[2]):
            position_2 = (position[0], position[1])
            rotation_matrix = cv2.getRotationMatrix2D(position_2, angle, 1)
            rotated_slice = cv2.warpAffine(mask[:, :, i], rotation_matrix, (mask.shape[1], mask.shape[0]),
                                           flags=cv2.INTER_NEAREST)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=2)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")


def upscale_rotate_downscale_fast(image, angle, axis, position, upscale_factor=2):
    rotated_slices = []

    if axis == 0:  # Rotate each depth slice
        for i in range(image.shape[0]):
            # Upscale the slice to increase detail for smoother rotation
            upscaled_slice = cv2.resize(image[i], None, fx=upscale_factor, fy=upscale_factor,
                                        interpolation=cv2.INTER_LINEAR)
            upscaled_position = (int(position[0] * upscale_factor), int(position[1] * upscale_factor))

            # Rotate the upscaled slice around the upscaled position with bicubic interpolation
            rotation_matrix = cv2.getRotationMatrix2D(upscaled_position, angle, 1)
            rotated_upscaled_slice = cv2.warpAffine(upscaled_slice, rotation_matrix,
                                                    (upscaled_slice.shape[1], upscaled_slice.shape[0]),
                                                    flags=cv2.INTER_CUBIC)

            # Downscale back to original size for faster processing
            rotated_slice = cv2.resize(rotated_upscaled_slice, (image.shape[2], image.shape[1]),
                                       interpolation=cv2.INTER_AREA)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=0)

    elif axis == 1:  # Rotate each height slice
        for i in range(image.shape[1]):
            upscaled_slice = cv2.resize(image[:, i, :], None, fx=upscale_factor, fy=upscale_factor,
                                        interpolation=cv2.INTER_LINEAR)
            upscaled_position = (int(position[0] * upscale_factor), int(position[1] * upscale_factor))
            rotation_matrix = cv2.getRotationMatrix2D(upscaled_position, angle, 1)
            rotated_upscaled_slice = cv2.warpAffine(upscaled_slice, rotation_matrix,
                                                    (upscaled_slice.shape[1], upscaled_slice.shape[0]),
                                                    flags=cv2.INTER_CUBIC)
            rotated_slice = cv2.resize(rotated_upscaled_slice, (image.shape[2], image.shape[0]),
                                       interpolation=cv2.INTER_AREA)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=1)

    elif axis == 2:  # Rotate each width slice
        for i in range(image.shape[2]):
            upscaled_slice = cv2.resize(image[:, :, i], None, fx=upscale_factor, fy=upscale_factor,
                                        interpolation=cv2.INTER_LINEAR)
            upscaled_position = (int(position[0] * upscale_factor), int(position[1] * upscale_factor))
            rotation_matrix = cv2.getRotationMatrix2D(upscaled_position, angle, 1)
            rotated_upscaled_slice = cv2.warpAffine(upscaled_slice, rotation_matrix,
                                                    (upscaled_slice.shape[1], upscaled_slice.shape[0]),
                                                    flags=cv2.INTER_CUBIC)
            rotated_slice = cv2.resize(rotated_upscaled_slice, (image.shape[1], image.shape[0]),
                                       interpolation=cv2.INTER_AREA)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=2)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")


def upscale_rotate_downscale_binary(mask, angle, axis, position, upscale_factor=2):
    rotated_slices = []

    if axis == 0:  # Rotate each depth slice
        for i in range(mask.shape[0]):
            # Upscale the slice to increase detail for smoother rotation
            upscaled_slice = cv2.resize(mask[i], None, fx=upscale_factor, fy=upscale_factor,
                                        interpolation=cv2.INTER_NEAREST)
            upscaled_position = (int(position[0] * upscale_factor), int(position[1] * upscale_factor))

            # Rotate the upscaled slice around the upscaled position with bilinear interpolation
            rotation_matrix = cv2.getRotationMatrix2D(upscaled_position, angle, 1)
            rotated_upscaled_slice = cv2.warpAffine(upscaled_slice, rotation_matrix,
                                                    (upscaled_slice.shape[1], upscaled_slice.shape[0]),
                                                    flags=cv2.INTER_LINEAR)

            # Threshold to keep binary values after rotation
            rotated_upscaled_slice = (rotated_upscaled_slice > 0.5).astype(np.uint8)

            # Downscale back to original size, preserving binary values with nearest neighbor
            rotated_slice = cv2.resize(rotated_upscaled_slice, (mask.shape[2], mask.shape[1]),
                                       interpolation=cv2.INTER_NEAREST)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=0)

    elif axis == 1:  # Rotate each height slice
        for i in range(mask.shape[1]):
            upscaled_slice = cv2.resize(mask[:, i, :], None, fx=upscale_factor, fy=upscale_factor,
                                        interpolation=cv2.INTER_NEAREST)
            upscaled_position = (int(position[0] * upscale_factor), int(position[1] * upscale_factor))
            rotation_matrix = cv2.getRotationMatrix2D(upscaled_position, angle, 1)
            rotated_upscaled_slice = cv2.warpAffine(upscaled_slice, rotation_matrix,
                                                    (upscaled_slice.shape[1], upscaled_slice.shape[0]),
                                                    flags=cv2.INTER_LINEAR)
            rotated_upscaled_slice = (rotated_upscaled_slice > 0.5).astype(np.uint8)
            rotated_slice = cv2.resize(rotated_upscaled_slice, (mask.shape[2], mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=1)

    elif axis == 2:  # Rotate each width slice
        for i in range(mask.shape[2]):
            upscaled_slice = cv2.resize(mask[:, :, i], None, fx=upscale_factor, fy=upscale_factor,
                                        interpolation=cv2.INTER_NEAREST)
            upscaled_position = (int(position[0] * upscale_factor), int(position[1] * upscale_factor))
            rotation_matrix = cv2.getRotationMatrix2D(upscaled_position, angle, 1)
            rotated_upscaled_slice = cv2.warpAffine(upscaled_slice, rotation_matrix,
                                                    (upscaled_slice.shape[1], upscaled_slice.shape[0]),
                                                    flags=cv2.INTER_LINEAR)
            rotated_upscaled_slice = (rotated_upscaled_slice > 0.5).astype(np.uint8)
            rotated_slice = cv2.resize(rotated_upscaled_slice, (mask.shape[1], mask.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            rotated_slices.append(rotated_slice)
        return np.stack(rotated_slices, axis=2)
    else:
        raise ValueError("Axis must be 0, 1, or 2.")


import numpy as np
from scipy.ndimage import affine_transform

def sequential_rotate_volume(volume, angles, center, if_binary = False):
    """
    Rotate a 3D volume sequentially around X, Y, and Z spatial axes.

    Args:
        volume (numpy.ndarray): The 3D volume to rotate, shape (D, H, W).
        angles (tuple): Rotation angles in degrees for (D, H, W) spatial axes.
        center (tuple): Rotation center (D, H, W).

    Returns:
        numpy.ndarray: The rotated volume.
    """
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)  # (angle_x, angle_y, angle_z)

    # Rotation around X-axis (width): Affects (H, W)-plane
    cos_x, sin_x = np.cos(angles_rad[0]), np.sin(angles_rad[0])
    rotation_x = np.array([
        [1,      0,       0],
        [0, cos_x, -sin_x],
        [0, sin_x,  cos_x]
    ])

    # Rotation around Y-axis (height): Affects (D, W)-plane
    cos_y, sin_y = np.cos(angles_rad[1]), np.sin(angles_rad[1])
    rotation_y = np.array([
        [ cos_y, 0, sin_y],
        [ 0,     1,     0],
        [-sin_y, 0, cos_y]
    ])

    # Rotation around Z-axis (depth): Affects (D, H)-plane
    cos_z, sin_z = np.cos(angles_rad[2]), np.sin(angles_rad[2])
    rotation_z = np.array([
        [cos_z, -sin_z, 0],
        [sin_z,  cos_z, 0],
        [0,      0,     1]
    ])

    # Combined rotation matrix (X -> Y -> Z)
    combined_rotation = rotation_z @ rotation_y @ rotation_x

    # Calculate the offset to keep the center fixed
    center = np.array(center)
    offset = center - combined_rotation @ center

    # Apply the affine transformation
    rotated_volume = affine_transform(
        volume,
        combined_rotation,
        offset=offset,
        order=1,  # Linear interpolation
        mode='constant',  # Fill out-of-bound values with zeros
        cval=0.0
    )
    if if_binary:
        rotated_volume = (rotated_volume > 0.5).astype(np.uint8)

    return rotated_volume

import torch


def rotate_3d_array_pytorch(input_array, rotation_matrix, center_cor):
    """
    Rotates a 3D array using a given 3x3 rotation matrix with PyTorch for GPU acceleration.

    Parameters:
        input_array (np.ndarray): The 3D array (D, H, W) to rotate.
        rotation_matrix (np.ndarray): A 3x3 rotation matrix.

    Returns:
        np.ndarray: Rotated 3D array.
    """
    # Convert inputs to PyTorch tensors
    input_tensor = torch.tensor(input_array, dtype=torch.float32, device='cuda')
    rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float32, device='cuda')

    # Create a grid of coordinates
    D, H, W = input_tensor.shape
    z, y, x = torch.meshgrid(
        torch.arange(D, device='cuda', dtype=torch.float32),
        torch.arange(H, device='cuda', dtype=torch.float32),
        torch.arange(W, device='cuda', dtype=torch.float32),
        indexing='ij'
    )

    # Stack coordinates and apply the rotation
    coords = torch.stack([z, y, x], dim=0).reshape(3, -1)
    center = torch.tensor([[center_cor[0]], [center_cor[1]], [center_cor[2]]], device='cuda', dtype=torch.float32)
    coords = coords - center
    rotated_coords = torch.matmul(rotation_matrix, coords) + center

    # Normalize coordinates for grid_sample
    rotated_coords = rotated_coords.reshape(3, H, W, D).permute(3, 2, 1, 0)
    rotated_coords = 2.0 * (rotated_coords / torch.tensor([D - 1, H - 1, W - 1], device='cuda')) - 1.0

    # Prepare grid for grid_sample
    grid = rotated_coords.unsqueeze(0)
    rotated_tensor = torch.nn.functional.grid_sample(
        input_tensor.unsqueeze(0).unsqueeze(0),
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )

    # Remove extra dimensions and convert back to NumPy
    return rotated_tensor.squeeze().cpu().numpy()


def sequential_rotate_volume_pytorch(volume, angles, center, if_binary = False):
    """
    Rotate a 3D volume sequentially around X, Y, and Z spatial axes.

    Args:
        volume (numpy.ndarray): The 3D volume to rotate, shape (D, H, W).
        angles (tuple): Rotation angles in degrees for (D, H, W) spatial axes.
        center (tuple): Rotation center (D, H, W).

    Returns:
        numpy.ndarray: The rotated volume.
    """
    # Convert angles to radians
    angles_rad = np.deg2rad(angles)  # (angle_x, angle_y, angle_z)

    # Rotation around X-axis (width): Affects (H, W)-plane
    cos_x, sin_x = np.cos(angles_rad[0]), np.sin(angles_rad[0])
    rotation_x = np.array([
        [cos_x, -sin_x, 0],
        [sin_x, cos_x, 0],
        [0, 0, 1]
    ])

    # Rotation around Y-axis (height): Affects (D, W)-plane
    cos_y, sin_y = np.cos(angles_rad[1]), np.sin(angles_rad[1])
    rotation_y = np.array([
        [cos_y, 0, sin_y],
        [0,     1,     0],
        [-sin_y, 0, cos_y]
    ])

    # Rotation around Z-axis (depth): Affects (D, H)-plane
    cos_z, sin_z = np.cos(angles_rad[2]), np.sin(angles_rad[2])
    rotation_z = np.array([
        [1,      0,       0],
        [0, cos_z, -sin_z],
        [0, sin_z,  cos_z]
    ])

    # Combined rotation matrix (X -> Y -> Z)
    combined_rotation = rotation_z @ rotation_y @ rotation_x

    # Calculate the offset to keep the center fixed
    center = np.array(center)

    # Apply the affine transformation
    rotated_volume = rotate_3d_array_pytorch(volume, combined_rotation, center)
    if if_binary:
        rotated_volume = (rotated_volume > 0.5).astype(np.uint8)

    return rotated_volume