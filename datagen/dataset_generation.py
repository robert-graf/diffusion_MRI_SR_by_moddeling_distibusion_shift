import gc
import os
import random
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from numpy.typing import NDArray
from TPTBox import NII
from TPTBox.core.vert_constants import logging
from TPTBox.logger.log_file import Log_Type, Reflection_Logger

random.seed(6456)
logger = Reflection_Logger()
logger.override_prefix = "Data GEN"


def deformed_nii(
    arr_dic: dict[str, NII], sigma=None, points=None, deform_factor=1.0, deform_padding=10, verbose: logging = True
) -> dict[str, NII]:
    """
    Deform a dictionary of NII objects using random grid deformation.

    This function takes a dictionary of NII objects and applies random grid deformation to each object
    using specified deformation parameters or, if not provided, random parameters generated based on
    the `deform_factor`. The deformed objects are returned as a dictionary.

    Args:
        arr_dic (dict[str, NII]): A dictionary containing NII objects to be deformed.
        sigma (float, optional): The standard deviation of the deformation field. If not provided,
            it will be generated based on the `deform_factor`.
        points (int, optional): The number of control points for the deformation grid. If not provided,
            it will be generated based on the `deform_factor`.
        deform_factor (float, optional): A factor used to determine the deformation parameters if
            `sigma` and `points` are not specified. Larger values result in stronger deformations.
        deform_padding (int, optional): The padding added to the deformed objects to avoid edge artifacts.
        verbose (bool, optional): If True, enable verbose logging. Default is True.

    Returns:
        dict[str, NII]: A dictionary where keys correspond to the input dictionary keys, and values
        correspond to the deformed NII objects.

    Example:
        # Deform a dictionary of NII objects using default deformation parameters
        deformed_data = deformed_NII(arr_dic)

        # Deform a dictionary of NII objects with specific deformation parameters
        sigma = 1.0
        points = 20
        deformed_data = deformed_NII(arr_dic, sigma=sigma, points=points)
    """
    if sigma is None or points is None:
        sigma, points = get_random_deform_parameter(deform_factor=deform_factor)

    logger.print("deformation parameter sigma = ", round(sigma, 4), "; n_points = ", points, verbose=verbose)
    t = time.time()
    values = list(arr_dic.values())
    # Deform
    import elasticdeform

    assert sigma is not None
    p = deform_padding
    out: list[NDArray] = elasticdeform.deform_random_grid(
        [pad(v.get_array(), p=p) for v in values],
        sigma=sigma,
        points=points,
        order=[0 if v.seg else 3 for v in values],  # type: ignore
    )
    out2: dict[str, NII] = {}
    for (k, nii), arr in zip(arr_dic.items(), out, strict=True):
        out2[k] = nii.set_array(arr[p:-p, p:-p, p:-p])
    logger.print("Deformation took", round(time.time() - t, 1), "Seconds")
    return out2


def pad(arr, p=10):
    return np.pad(arr, p, mode="reflect")


def make_3d_np_to_2d(
    niis: dict[str, NII],
    filter_slice: Callable[[dict[str, NDArray]], bool] | None = None,
    deform=True,
    crop3D=None,
    crop2D: Callable[[dict[str, NDArray]], dict[str, NDArray]] | None = None,
    max_acc_digits=5,
    **kwargs,
):
    """
    Convert a dictionary of 3D NII objects into a list of 2D slices.

    This function takes a dictionary of NII objects, performs optional 3D deformation, and generates
    a list of 2D slices from each 3D NII object. Optionally, it can apply filtering to exclude
    specific slices based on the provided filter function.

    Args:
        niis (dict[str, NII]): A dictionary containing 3D NII objects to be converted to 2D slices.
        filter (Callable[[dict[str, NDArray]], bool] | None, optional): A filter function that
            determines whether to include or skip a slice. Default is None.
        deform (bool, optional): If True, apply 3D deformation to the NII objects. Default is True.
        crop3D (str | None, optional): If not None, specify the key for cropping all NII objects
            based on the computed crop slice of the specified NII object. Default is None.
        max_acc_digits (int, optional): The number of decimal places to round the slices if
            they are non-segmentation data. Default is 5.
        **kwargs: Additional keyword arguments to be passed to the deformation function (deformed_NII).

    Returns:
        list[tuple[int, dict[str, NDArray]]]: A list of 2D slices represented as dictionaries, where
        each tuple contains the slice index and the dictionary of NII objects' slices.

    Example:
        # Convert 3D NII objects to 2D slices with deformation
        slices = make_3d_np_to_2d(niis, deform=True)

        # Convert 3D NII objects to 2D slices without deformation and apply a custom filter
        def custom_filter(slice_dict):
            # Implement a custom filtering logic here
            return True  # Include the slice if the condition is met

        slices = make_3d_np_to_2d(niis, deform=False, filter=custom_filter)
    """
    # out = []
    if crop3D is not None:
        ex_slice = niis[crop3D].compute_crop(minimum=0.1)
        for key, nii in niis.items():
            nii.apply_crop_(ex_slice)
    # polynomial 3D deformation
    if deform:
        niis = deformed_nii(niis, **kwargs)
    for i in range(niis[next(iter(niis.keys()))].shape[0]):
        slice_dic: dict[str, NDArray] = {}
        seg_dic: dict[str, bool] = {}
        for key, nii in niis.items():
            slice = nii.get_array()[i]
            if nii.seg != 1:
                slice = np.clip(slice, a_min=0, a_max=1)
                slice = np.round(slice, decimals=max_acc_digits)
            seg_dic[key] = nii.seg
            slice_dic[key] = slice
        # filter slices
        if filter_slice is not None and not filter_slice(slice_dic):
            logger.print("skip slice", i, end="\r")
            continue
        if crop2D is not None:
            slice_dic = crop2D(slice_dic)

        yield (i, slice_dic, seg_dic)

        # out.append((i, slice_dic))

    # return out


def make_np_to_png(
    path: str | Path,
    prefix: str,
    arr_dic: dict[str, NII],
    filter_slice: Callable[[dict[str, NDArray]], bool] | None = None,
    single_png=False,
    # crop3d: str | None = None,
    deform: bool = False,
    sub_folders: dict[str, str] | None = None,
    **kwargs,
):
    """
    Convert a dictionary of NII objects to PNG images and save them to a specified directory.

    This function converts the slices of 3D NII objects (from the dictionary `arr_dic`) into PNG images
    and saves them to the specified directory. You can specify a filter function to include or exclude
    slices, control the output format (single PNG or separate PNGs for each key), and perform cropping
    and deformation of the NII objects if desired.

    Args:
        path (str | Path): The directory path where the PNG images will be saved.
        prefix (str): The prefix to be added to the image file names.
        arr_dic (dict[str, NII]): A dictionary containing NII objects to be converted to PNG images.
        filter (Callable[[dict[str, NDArray]], bool] | None, optional): A filter function that
            determines whether to include or skip a slice. Default is None.
        single_png (bool, optional): If True, save all NII slices in a single PNG image. If False,
            save separate PNG images for each key. Default is False.
        crop3D (str | None, optional): If not None, specify the key for cropping all NII objects
            based on the computed crop slice of the specified NII object. Default is None.
        deform (bool, optional): If True, apply 3D deformation to the NII objects. Default is False.
        **kwargs: Additional keyword arguments to be passed to the deformation function (make_3d_np_to_2d).

    Returns:
        None: The function saves the PNG images to the specified directory.

    Example:
        # Convert NII objects to separate PNG images with deformation and custom filter
        def custom_filter(slice_dict):
            # Implement a custom filtering logic here
            return True  # Include the slice if the condition is met

        make_np_to_PNG(path="output_directory", prefix="image", arr_dic=niis, deform=True, filter=custom_filter)

        # Convert NII objects to a single PNG image without deformation
        make_np_to_PNG(path="output_directory", prefix="image", arr_dic=niis, deform=False, single_png=True)
    """
    if sub_folders is None:
        sub_folders = {}
    path = str(path)
    Path(path).mkdir(exist_ok=True, parents=True)
    from PIL import Image

    for idx, slice_dic, seg_dic in make_3d_np_to_2d(arr_dic, filter_slice=filter_slice, deform=deform, **kwargs):
        if single_png:
            arr_list = list(slice_dic.values())
            value = np.concatenate(arr_list, axis=1)
            im = Image.fromarray(value * 255)
            im = im.convert("L")

            # sub_folders
            logger.print(Path(path, f"{prefix}_{idx}.png      "), end="\r")
            im.save(Path(path, f"{prefix}_{idx}.png"))

        else:
            for key, value in slice_dic.items():
                im = Image.fromarray(value * (1 if seg_dic[key] else 255))
                im = im.convert("L")

                out_path = Path(path, sub_folders.get(key, ""), f"{prefix}_{key}_{idx}.png")
                out_path.parent.mkdir(exist_ok=True)
                logger.print(out_path, "        ", end="\r")
                im.save(out_path)


def make_np_to_npz(path, image_name, arr_dic, **kwargs):
    """
    Convert a dictionary of NII objects to NumPy compressed (.npz) files and save them to a specified directory.

    This function converts the slices of 3D NII objects (from the dictionary `arr_dic`) into NumPy
    compressed (.npz) files and saves them to the specified directory. You can control the output format,
    perform cropping, deformation, and other operations on the NII objects if desired.

    Args:
        path (str | Path): The directory path where the .npz files will be saved.
        image_name (str): The base name for the .npz files.
        arr_dic (dict[str, NII]): A dictionary containing NII objects to be converted to .npz files.
        **kwargs: Additional keyword arguments to be passed to the conversion function (make_3d_np_to_2d).

    Returns:
        None: The function saves the .npz files to the specified directory.

    Example:
        # Convert NII objects to .npz files with deformation
        make_np_to_npz(path="output_directory", image_name="image", arr_dic=niis, deform=True)

        # Convert NII objects to .npz files without deformation
        make_np_to_npz(path="output_directory", image_name="image", arr_dic=niis, deform=False)
    """
    out_list = []
    Path(path).mkdir(exist_ok=True, parents=True)
    for idx, arr_dic_ in make_3d_np_to_2d(arr_dic, **kwargs):
        out = Path(path, f"{image_name}_{idx}.npz")
        # if not is_compressed(out):
        #    continue
        np.savez(out, **arr_dic_)
        out_list.append(out)
    return out_list


def is_compressed(npz_file):
    zip_infos = np.load(npz_file).zip.infolist()
    if len(zip_infos) == 0:
        raise RuntimeError("Did not find ZipInfos unexpectedly")
    compress_type = zip_infos[0].compress_type
    if compress_type == zipfile.ZIP_STORED:
        return False
    if compress_type == zipfile.ZIP_DEFLATED:
        return True

    raise ValueError("Unexpected compression type")


def make_np_to_fnp(path, prefix, arr_dic, sub_folders=None, **kwargs):
    if sub_folders is None:
        sub_folders = {}
    path = str(path)
    Path(path).mkdir(exist_ok=True, parents=True)
    from fastnumpyio import save

    for idx, slice_dic in make_3d_np_to_2d(arr_dic, **kwargs):
        for key, value in slice_dic.items():
            out_path = Path(path, sub_folders.get(key, ""), f"{prefix}_{key}_{idx}.fnp")
            out_path.parent.mkdir(exist_ok=True)
            logger.print(out_path, "    ", end="\r")
            save(out_path, value)


def make_nii_to_slice(
    png: bool, niis: dict[str, NII], out_path: Path, prefix: str, deform=True, sub_folders=None, fast_np: bool = False, **args
):
    """
    Convert a dictionary of NII objects to 2D slices (PNG images or .npz files) and save them to a specified directory.

    This function converts the slices of NII objects from the input dictionary into 2D slices (either PNG images or .npz
    files) and saves them to the specified directory. You can choose the output format, provide a prefix for the output
    files, and optionally apply 3D deformation to the NII objects.

    It is sliced along the first image dimension (use nii.reorient()) and expect images to be in [0,1]

    Args:
        png (bool): If True, convert the NII objects to PNG images. If False, convert to .npz files.
        niis (dict[str, NII]): A dictionary containing NII objects to be converted to 2D slices.
        out_path (Path): The directory path where the output slices will be saved.
        prefix (str): The prefix to be added to the output file names.
        deform (bool, optional): If True, apply 3D deformation to the NII objects. Default is True.

    Returns:
        None: The function saves the 2D slices to the specified directory.

    Example:
        # Convert NII objects to PNG images with deformation
        make_nii_to_slice(png=True, niis=niis, out_path=Path("output_directory"), prefix="image", deform=True)

        # Convert NII objects to .npz files without deformation
        make_nii_to_slice(png=False, niis=niis, out_path=Path("output_directory"), prefix="data", deform=False)
    """
    # key = list(niis.keys())[-1]
    # out_path = Path(out_path, sub_folders.get(key, ""), f"{prefix}_{key}_{0}.png")
    #
    # if out_path.exists():
    #    logger.print(out_path, "exists")
    #    return
    if sub_folders is None:
        sub_folders = {}
    fk = next(iter(niis.keys()))
    assert all((nii.seg and nii.min() >= 0) or (nii.max() <= 1 and nii.min() >= 0) for nii in niis.values()), [
        (str(n), n.min(), n.max()) for n in niis.values()
    ]
    assert all(niis[fk].orientation == nii.orientation for nii in niis.values())
    assert all(all(abs(a - b) <= 0.1 for (a, b) in zip(niis[fk].zoom, nii.zoom, strict=True)) for nii in niis.values())

    if png:
        return make_np_to_png(out_path, prefix, niis, deform=deform, sub_folders=sub_folders, **args)
    if fast_np:
        return make_np_to_fnp(out_path, prefix, niis, deform=deform, sub_folders=sub_folders, **args)

    return make_np_to_npz(out_path, prefix, niis, deform=deform, sub_folders=sub_folders, **args)


def get_random_deform_parameter(deform_factor: float = 1):
    """
    Generate random deformation parameters for use in 3D deformation.

    This function generates random values for the deformation parameters, including 'sigma' and 'points',
    based on the specified deformation factor. These parameters are used for 3D deformation operations.

    Args:
        deform_factor (float, optional): A factor to control the strength of deformation. Default is 1.

    Returns:
        tuple[float, int]: A tuple containing the generated 'sigma' (float) and 'points' (int) parameters.

    Example:
        # Generate random deformation parameters with a deformation factor of 1
        sigma, points = get_random_deform_parameter()

        # Generate random deformation parameters with a deformation factor of 2
        sigma, points = get_random_deform_parameter(deform_factor=2)
    """
    sigma = 2 + np.random.uniform() * 2.5  # 1,5 - 4.5
    min_points = 3
    max_points = 17
    if sigma < 2:
        max_points = 17
    elif sigma < 1.7:
        max_points = 16
    elif sigma < 2.1:
        max_points = 15
    elif sigma < 2.3:
        max_points = 14
    elif sigma < 2.5:
        max_points = 13
    elif sigma < 2.6:
        max_points = 12
    elif sigma < 2.7:
        max_points = 11
    elif sigma < 2.8:
        max_points = 10
    elif sigma < 3:
        max_points = 9
    elif sigma < 3.5:
        max_points = 8
    elif sigma < 4.0:
        max_points = 7
    elif sigma < 4.3:
        max_points = 6
    else:
        max_points = 5
    points = np.random.randint(max_points - min_points + 1) + min_points
    # Stronger
    sigma *= deform_factor
    points *= deform_factor
    points = round(points)
    return (sigma, points)


def random_skip_filter(x: dict[str, NDArray], p=0.2) -> bool:
    return random.random() <= p


def make_ds(
    acq="ax",
    orientation=("I", "P", "R"),  # first will be cut
    zoom=None,
    out="/media/data/robert/datasets/dataset-neuropoly/training_img/",
    dataset="/media/data/robert/datasets/dataset-neuropoly",
    crop3d=True,
    name_folder=None,
):
    """Here was a non-sens merge. make axial is a frankensteins monster for totalseg and neuropoly"""
    if name_folder is None:
        name_folder = acq
    import pickle

    from TPTBox import BIDS_Global_info

    files = {"train": [], "test": [], "val": []}
    train_phase_names = {0: "train", 1: "test", 2: "val"}
    out_pk = Path(__file__).parent / "subject_spit.pk"
    if isinstance(dataset, list):
        for file in dataset:
            split_dict = {}
            a = random.random() <= 0.8
            if a:
                split_dict[file] = 0
            elif random.random() <= 0.5:
                split_dict[file] = 1
            else:
                split_dict[file] = 2
            files[train_phase_names[split_dict[file]]].append(file)
    else:
        assert False
        bgi = BIDS_Global_info(datasets=[dataset], parents=["rawdata"])

        if out_pk.exists():
            with open(out_pk, "rb") as handle:
                split_dict = pickle.load(handle)
        else:
            split_dict = {}
            for name, _ in bgi.iter_subjects():
                a = random.random() <= 0.8
                if a:
                    split_dict[name] = 0
                elif random.random() <= 0.5:
                    split_dict[name] = 1
                else:
                    split_dict[name] = 2
            with open(out_pk, "wb") as handle:
                pickle.dump(split_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        for name, sub in bgi.iter_subjects():
            print(name)
            sub.new_query()
            q = sub.new_query(flatten=True)
            q.filter_format("T2w")
            q.filter("acq", acq)
            q.filter_non_existence("seg")
            q.filter_non_existence("lesions")
            q.filter_filetype("nii.gz")
            for file in q.loop_list():
                if "stitched" in str(file):
                    continue
                files[train_phase_names[split_dict[name]]].append(file.file["nii.gz"])

    def slice_total_seg_(i, nii_path, prefix_, phase, zoom, name_folder):
        gc.collect()
        try:
            mr = NII.load(nii_path, False).reorient_(orientation)
            if zoom is not None:
                mr.rescale_(zoom)
        except EOFError:
            return
        mr = mr.normalize(clamp_lower=0)
        prefix = f"{prefix_}_{i:05}"
        l = mr.shape[0]
        out_path = Path(out_pk, phase, name_folder, f"{prefix}_{name_folder}_{mr.shape[0]}.png")

        if out_path.exists():
            logger.print("Skip", out_path, "exists")
            return

        logger.print("Start", i, ltype=Log_Type.ITALICS)

        niis = {prefix_: mr}
        zoom = mr.zoom
        if name_folder == "ax":
            w = mr.zoom[-1]
            if w <= 0.4:
                r = random.random()
                z = 0.4571 * r + 0.4
                mr2 = mr.rescale((-1, z, z)).normalize(clamp_lower=0)
                niis = {prefix_: mr2}
                zoom = mr2.zoom
        sub_folders = {prefix_: prefix_}
        make_nii_to_slice(
            True,
            niis,
            out_path=Path(out, phase),
            prefix=prefix,
            deform=False,
            single_png=False,
            crop3D=prefix_ if crop3d else None,
            sub_folders=sub_folders,
        )

        # niis = {prefix_: mr.rescale_((-1, 0.8571, 0.8571)).normalize(clamp_lower=0)}
        # make_nii_to_slice(
        #    True,
        #    niis,
        #    out_path=Path(out, phase),
        #    prefix=prefix + "_08571",
        #    deform=False,
        #    single_png=False,
        #    crop3D=prefix_ if crop3D else None,
        #    sub_folders=sub_folders,
        # )
        logger.print(i, mr.zoom, zoom, f"finished{' ':90}", ltype=Log_Type.SAVE)
        gc.collect()

    tasks = []
    a = os.cpu_count() // 2
    print()
    print()
    for trainings_phase, list_nii in files.items():
        for i, ct_path in enumerate(list_nii):
            tasks.append(delayed(slice_total_seg_)(i, ct_path, name_folder, trainings_phase, zoom, name_folder))
    print("Start n =", len(tasks))
    Parallel(n_jobs=a)(tasks)


if __name__ == "__main__":
    # make_sublist(save="/DATA/NAS/ongoing_projects/robert/code/SynSeg-Net/sublist/sublist_TotalSeg_CT.txt")
    make_ds(acq="sag", name_folder="sa8571", orientation=("R", "I", "P"), zoom=(-1, 0.8571, 0.8571))
    # make_axial_ds(acq="ax", orientation=("I", "P", "R"))
    # make_axial_ds(acq="ax", name_folder="sag-of-ax", orientation=("R", "I", "P"), zoom=(0.8571, 0.8571, 0.8571))
