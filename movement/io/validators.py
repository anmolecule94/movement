"""`attrs` classes for validating file paths and data structures."""

import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Optional, Union

import h5py
import numpy as np
from attrs import converters, define, field, validators

from movement.logging import log_error, log_warning


@define
class ValidFile:
    """Class for validating file paths.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the file.
    expected_permission : {"r", "w", "rw"}
        Expected access permission(s) for the file. If "r", the file is
        expected to be readable. If "w", the file is expected to be writable.
        If "rw", the file is expected to be both readable and writable.
        Default: "r".
    expected_suffix : list of str
        Expected suffix(es) for the file. If an empty list (default), this
        check is skipped.

    Raises
    ------
    IsADirectoryError
        If the path points to a directory.
    PermissionError
        If the file does not have the expected access permission(s).
    FileNotFoundError
        If the file does not exist when `expected_permission` is "r" or "rw".
    FileExistsError
        If the file exists when `expected_permission` is "w".
    ValueError
        If the file does not have one of the expected suffix(es).

    """

    path: Path = field(converter=Path, validator=validators.instance_of(Path))
    expected_permission: Literal["r", "w", "rw"] = field(
        default="r", validator=validators.in_(["r", "w", "rw"]), kw_only=True
    )
    expected_suffix: list[str] = field(factory=list, kw_only=True)

    @path.validator
    def path_is_not_dir(self, attribute, value):
        """Ensure that the path does not point to a directory."""
        if value.is_dir():
            raise log_error(
                IsADirectoryError,
                f"Expected a file path but got a directory: {value}.",
            )

    @path.validator
    def file_exists_when_expected(self, attribute, value):
        """Ensure that the file exists (or not) as needed.

        This depends on the expected usage (read and/or write).
        """
        if "r" in self.expected_permission:
            if not value.exists():
                raise log_error(
                    FileNotFoundError, f"File {value} does not exist."
                )
        else:  # expected_permission is "w"
            if value.exists():
                raise log_error(
                    FileExistsError, f"File {value} already exists."
                )

    @path.validator
    def file_has_access_permissions(self, attribute, value):
        """Ensure that the file has the expected access permission(s).

        Raises a PermissionError if not.
        """
        file_is_readable = os.access(value, os.R_OK)
        parent_is_writeable = os.access(value.parent, os.W_OK)
        if ("r" in self.expected_permission) and (not file_is_readable):
            raise log_error(
                PermissionError,
                f"Unable to read file: {value}. "
                "Make sure that you have read permissions.",
            )
        if ("w" in self.expected_permission) and (not parent_is_writeable):
            raise log_error(
                PermissionError,
                f"Unable to write to file: {value}. "
                "Make sure that you have write permissions.",
            )

    @path.validator
    def file_has_expected_suffix(self, attribute, value):
        """Ensure that the file has one of the expected suffix(es)."""
        if self.expected_suffix and value.suffix not in self.expected_suffix:
            raise log_error(
                ValueError,
                f"Expected file with suffix(es) {self.expected_suffix} "
                f"but got suffix {value.suffix} instead.",
            )


@define
class ValidHDF5:
    """Class for validating HDF5 files.

    Parameters
    ----------
    path : pathlib.Path
        Path to the HDF5 file.
    expected_datasets : list of str or None
        List of names of the expected datasets in the HDF5 file. If an empty
        list (default), this check is skipped.

    Raises
    ------
    ValueError
        If the file is not in HDF5 format or if it does not contain the
        expected datasets.

    """

    path: Path = field(validator=validators.instance_of(Path))
    expected_datasets: list[str] = field(factory=list, kw_only=True)

    @path.validator
    def file_is_h5(self, attribute, value):
        """Ensure that the file is indeed in HDF5 format."""
        try:
            with h5py.File(value, "r") as f:
                f.close()
        except Exception as e:
            raise log_error(
                ValueError,
                f"File {value} does not seem to be in valid" "HDF5 format.",
            ) from e

    @path.validator
    def file_contains_expected_datasets(self, attribute, value):
        """Ensure that the HDF5 file contains the expected datasets."""
        if self.expected_datasets:
            with h5py.File(value, "r") as f:
                diff = set(self.expected_datasets).difference(set(f.keys()))
                if len(diff) > 0:
                    raise log_error(
                        ValueError,
                        f"Could not find the expected dataset(s) {diff} "
                        f"in file: {value}. ",
                    )


@define
class ValidDeepLabCutCSV:
    """Class for validating DeepLabCut-style .csv files.

    Parameters
    ----------
    path : pathlib.Path
        Path to the .csv file.

    Raises
    ------
    ValueError
        If the .csv file does not contain the expected DeepLabCut index column
        levels among its top rows.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def csv_file_contains_expected_levels(self, attribute, value):
        """Ensure that the .csv file contains the expected index column levels.

        These are to be found among the top 4 rows of the file.
        """
        expected_levels = ["scorer", "bodyparts", "coords"]

        with open(value) as f:
            top4_row_starts = [f.readline().split(",")[0] for _ in range(4)]

            if top4_row_starts[3].isdigit():
                # if 4th row starts with a digit, assume single-animal DLC file
                expected_levels.append(top4_row_starts[3])
            else:
                # otherwise, assume multi-animal DLC file
                expected_levels.insert(1, "individuals")

            if top4_row_starts != expected_levels:
                raise log_error(
                    ValueError,
                    ".csv header rows do not match the known format for "
                    "DeepLabCut pose estimation output files.",
                )


def _list_of_str(value: Union[str, Iterable[Any]]) -> list[str]:
    """Try to coerce the value into a list of strings."""
    if isinstance(value, str):
        log_warning(
            f"Invalid value ({value}). Expected a list of strings. "
            "Converting to a list of length 1."
        )
        return [value]
    elif isinstance(value, Iterable):
        return [str(item) for item in value]
    else:
        raise log_error(
            ValueError, f"Invalid value ({value}). Expected a list of strings."
        )


def _ensure_type_ndarray(value: Any) -> None:
    """Raise ValueError the value is a not numpy array."""
    if not isinstance(value, np.ndarray):
        raise log_error(
            ValueError, f"Expected a numpy array, but got {type(value)}."
        )


def _set_fps_to_none_if_invalid(fps: Optional[float]) -> Optional[float]:
    """Set fps to None if a non-positive float is passed."""
    if fps is not None and fps <= 0:
        log_warning(
            f"Invalid fps value ({fps}). Expected a positive number. "
            "Setting fps to None."
        )
        return None
    return fps


def _validate_list_length(
    attribute, value: Optional[list], expected_length: int
):
    """Raise a ValueError if the list does not have the expected length."""
    if (value is not None) and (len(value) != expected_length):
        raise log_error(
            ValueError,
            f"Expected `{attribute.name}` to have length {expected_length}, "
            f"but got {len(value)}.",
        )


@define(kw_only=True)
class ValidPosesDataset:
    """Class for validating data intended for a ``movement`` dataset.

    Attributes
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_individuals, n_keypoints, n_space)
        containing the poses.
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals, n_keypoints) containing
        the point-wise confidence scores.
        If None (default), the scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of unique names for the individuals in the video. If None
        (default), the individuals will be named "individual_0",
        "individual_1", etc.
    keypoint_names : list of str, optional
        List of unique names for the keypoints in the skeleton. If None
        (default), the keypoints will be named "keypoint_0", "keypoint_1",
        etc.
    fps : float, optional
        Frames per second of the video. Defaults to None.
    source_software : str, optional
        Name of the software from which the poses were loaded.
        Defaults to None.

    """

    # Define class attributes
    position_array: np.ndarray = field()
    confidence_array: Optional[np.ndarray] = field(default=None)
    individual_names: Optional[list[str]] = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    keypoint_names: Optional[list[str]] = field(
        default=None,
        converter=converters.optional(_list_of_str),
    )
    fps: Optional[float] = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _set_fps_to_none_if_invalid
        ),
    )
    source_software: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
    )

    # Add validators
    @position_array.validator
    def _validate_position_array(self, attribute, value):
        _ensure_type_ndarray(value)
        if value.ndim != 4:
            raise log_error(
                ValueError,
                f"Expected `{attribute.name}` to have 4 dimensions, "
                f"but got {value.ndim}.",
            )
        if value.shape[-1] not in [2, 3]:
            raise log_error(
                ValueError,
                f"Expected `{attribute.name}` to have 2 or 3 spatial "
                f"dimensions, but got {value.shape[-1]}.",
            )

    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        if value is not None:
            _ensure_type_ndarray(value)
            expected_shape = self.position_array.shape[:-1]
            if value.shape != expected_shape:
                raise log_error(
                    ValueError,
                    f"Expected `{attribute.name}` to have shape "
                    f"{expected_shape}, but got {value.shape}.",
                )

    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        if self.source_software == "LightningPose":
            # LightningPose only supports a single individual
            _validate_list_length(attribute, value, 1)
        else:
            _validate_list_length(
                attribute, value, self.position_array.shape[1]
            )

    @keypoint_names.validator
    def _validate_keypoint_names(self, attribute, value):
        _validate_list_length(attribute, value, self.position_array.shape[2])

    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        if self.confidence_array is None:
            self.confidence_array = np.full(
                (self.position_array.shape[:-1]), np.nan, dtype="float32"
            )
            log_warning(
                "Confidence array was not provided."
                "Setting to an array of NaNs."
            )
        if self.individual_names is None:
            self.individual_names = [
                f"individual_{i}" for i in range(self.position_array.shape[1])
            ]
            log_warning(
                "Individual names were not provided. "
                f"Setting to {self.individual_names}."
            )
        if self.keypoint_names is None:
            self.keypoint_names = [
                f"keypoint_{i}" for i in range(self.position_array.shape[2])
            ]
            log_warning(
                "Keypoint names were not provided. "
                f"Setting to {self.keypoint_names}."
            )


@define(kw_only=True)
class ValidBboxesDataset:
    """Class for validating bounding boxes data for a ``movement`` dataset.

    We consider 2D bounding boxes only.

    Attributes
    ----------
    position_array : np.ndarray
        Array of shape (n_frames, n_unique_individual_names, n_space)
        containing the bounding boxes' centroid positions. It will be
        converted to a `xarray.DataArray` object named "position".
    shape_array : np.ndarray
        Array of shape (n_frames, n_unique_individual_names, n_space)
        containing the bounding boxes' width (extension along the x-axis) and
        height (extension along the y-axis). It will be converted to a
        `xarray.DataArray` object named "shape".
    confidence_array : np.ndarray, optional
        Array of shape (n_frames, n_individuals, n_keypoints) containing
        the bounding boxes confidence scores. It will be converted to a
        `xarray.DataArray` object named "confidence". If None (default), the
        confidence scores will be set to an array of NaNs.
    individual_names : list of str, optional
        List of individual_names for the tracked bounding boxes in the video.
        If None (default), all bounding boxes will be labelled "id_1".
        ------------------------- ---> remove
        Each ID is a unique string in the format `id_<N>`, where <N> is an
        integer from 1 to Inf.
        ------------------------- ---> remove
    fps : float, optional
        Frames per second of the video. Defaults to None.
    source_software : str, optional
        Name of the software from which the bounding boxes were loaded.
        Defaults to None.

    """

    # Required attributes
    position_array: np.ndarray = field()
    shape_array: np.ndarray = field()

    # Optional attributes
    confidence_array: Optional[np.ndarray] = field(default=None)
    individual_names: Optional[list[str]] = field(
        default=None,
        converter=converters.optional(
            _list_of_str
        ),  # force into list of strings if not
    )
    fps: Optional[float] = field(
        default=None,
        converter=converters.pipe(  # type: ignore
            converters.optional(float), _set_fps_to_none_if_invalid
        ),
    )
    source_software: Optional[str] = field(
        default=None,
        validator=validators.optional(validators.instance_of(str)),
    )

    # validators for position_array and shape_array
    @position_array.validator
    @shape_array.validator
    def _validate_position_and_shape_arrays(self, attribute, value):
        # check value is a numpy array
        _ensure_type_ndarray(value)

        # # --------------------
        # # check number of dimensions
        # n_expected_dimensions = (
        #     3  # (n_frames, n_unique_individual_names, n_space)
        # )
        # if value.ndim != n_expected_dimensions:
        #     raise log_error(
        #         ValueError,
        #         f"Expected `{attribute.name}` to have "
        #         f"{n_expected_dimensions} dimensions, "
        #         f"but got {value.ndim}.",
        #     )
        # # --------------------

        # check last dimension (spatial) has 2 coordinates
        # - for the position_array the coordinates are x,y
        # - for the shape_array the coordinates are width, height
        n_expected_spatial_coordinates = 2
        if value.shape[-1] != n_expected_spatial_coordinates:
            raise log_error(
                ValueError,
                f"Expected `{attribute.name}` to have 2 spatial coordinates, "
                f"but got {value.shape[-1]}.",
            )

    # validator for bboxes individual_names
    @individual_names.validator
    def _validate_individual_names(self, attribute, value):
        # check the total number of unique individual_names matches those in
        # position_array
        _validate_list_length(attribute, value, self.position_array.shape[1])

        # check IDs are unique
        if len(value) != len(set(value)):
            raise log_error(
                ValueError,
                "individual_names passed to the dataset are not unique. "
                f"There are {len(value)} elements in the list, but "
                f"only {len(set(value))} are unique.",
            )

        # Check individual_names are strings of the expected format
        # `id_<integer>`) and extract the ID numbers from the strings
        list_IDs_as_integers = [
            self._check_ID_str_and_extract_int(value_i) for value_i in value
        ]

        # if None in list: some elements don't match the expected pattern
        if None in list_IDs_as_integers:
            raise log_error(
                ValueError,
                "At least one ID does not fit the expected "
                "format. Expected strings in the format 'id_<integer>' "
                f"but got: {value}\n",
            )

        # # --------------------
        # # check IDs are 1-based ----> this is a file validator requirement
        # if not all([ID_int >= 1 for ID_int in list_IDs_as_integers]):
        #     list_wrong_ID_str = [
        #         value_i
        #         for ID_int, value_i in zip(list_IDs_as_integers, value)
        #         if not (ID_int >= 1)
        #     ]
        #     raise log_error(
        #         ValueError,
        #         "Some of the individual_names provided are not 1-based: "
        #         f"{list_wrong_ID_str}. \n"
        #         "Please provide individual_names whose numbering starts "
        #         "from 1.",
        #     )
        # # --------------------

    # validator for confidence array
    @confidence_array.validator
    def _validate_confidence_array(self, attribute, value):
        # check only if a confidence array is passed
        if value is not None:
            # check value is a numpy array
            _ensure_type_ndarray(value)

            # check shape of confidence array matches position_array
            expected_shape = self.position_array.shape[:-1]
            if value.shape != expected_shape:
                raise log_error(
                    ValueError,
                    f"Expected `{attribute.name}` to have shape "
                    f"{expected_shape}, but got {value.shape}.",
                )

    # define default values
    def __attrs_post_init__(self):
        """Assign default values to optional attributes (if None)."""
        # if confidence_array is None, set it to an array of NaNs
        if self.confidence_array is None:
            self.confidence_array = np.full(
                (self.position_array.shape[:-1]),
                np.nan,
                dtype="float32",
            )
            log_warning(
                "Confidence array was not provided."
                "Setting to an array of NaNs."
            )
        # if no individual_names are provided for the tracked boxes:
        # assign them unique IDs per frame, starting with 1 ("id_1")
        # position_array.shape = (n_frames, n_unique_individual_names, n_space)
        if self.individual_names is None:
            self.individual_names = [
                f"id_{i+1}" for i in range(self.position_array.shape[1])
            ]
            log_warning(
                "Individual names for the bounding boxes "
                "were not provided. "
                f"Setting to {self.individual_names}.\n"
                "(1-based IDs that are unique per frame)"
            )

    def _check_ID_str_and_extract_int(self, ID_str: str) -> Optional[int]:
        """Check if the ID string.

        The function checks the ID string is of the expected format
        (`id_<integer>`). If there is a match, it casts the number to an
        integer. If there is no match, it returns None.
        """
        # check if there is a full-match of the pattern
        match = re.fullmatch(r"id_(\d+)$", ID_str)

        # if full match: cast to integer
        if match:
            return int(match.group(1))
            # Note: if there is a match, the group is always made of digits
            # which can always be cast to integer
        # if no match: return None
        else:
            return None
