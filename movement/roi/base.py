"""Class for representing 1- or 2-dimensional regions of interest (RoIs)."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import shapely
import xarray as xr
from numpy.typing import ArrayLike
from shapely.coords import CoordinateSequence

from movement.kinematics import compute_forward_vector_angle
from movement.utils.broadcasting import broadcastable_method
from movement.utils.logging import log_error
from movement.utils.vector import compute_signed_angle_2d

LineLike: TypeAlias = shapely.LinearRing | shapely.LineString
PointLike: TypeAlias = list[float] | tuple[float, ...]
PointLikeList: TypeAlias = Sequence[PointLike]
RegionLike: TypeAlias = shapely.Polygon
SupportedGeometry: TypeAlias = LineLike | RegionLike


class BaseRegionOfInterest:
    """Base class for regions of interest (RoIs).

    Regions of interest can be either 1 or 2 dimensional, and are represented
    by corresponding ``shapely.Geometry`` objects.

    To avoid the complexities of subclassing ``shapely`` objects (due to them
    relying on ``__new__()`` methods, see
    https://github.com/shapely/shapely/issues/1233), we simply wrap the
    relevant ``shapely`` object in the ``_shapely_geometry`` attribute of the
    class. This is accessible via the property ``region``. This also allows us
    to forbid certain operations and make the manipulation of ``shapely``
    objects more user friendly.

    Although this class can be instantiated directly, it is not designed for
    this. Its primary purpose is to reduce code duplication.

    Notes
    -----
    A region of interest includes the points that make up its boundary and the
    points contained in its interior. This convention means that points inside
    the region will be treated as having zero distance to the region, and the
    approach vector from these points to the region will be the null vector.

    This may be undesirable in certain situations, when we explicitly want to
    find the distance of a point to the boundary of a region, for example. To
    accommodate this, most methods of this class accept a keyword argument that
    forces the method to perform all computations using only the boundary of
    the region, rather than the region itself. For polygons, this will force
    the method in question to only consider distances or closest points on the
    segments that form the (interior and exterior) boundaries. For segments,
    the boundary is considered to be just the two endpoints of the segment.

    """

    __default_name: str = "Un-named region"

    _name: str | None
    _shapely_geometry: SupportedGeometry

    @property
    def coords(self) -> CoordinateSequence:
        """Coordinates of the points that define the region.

        These are the points passed to the constructor argument ``points``.

        Note that for Polygonal regions, these are the coordinates of the
        exterior boundary, interior boundaries must be accessed via
        ``self.region.interior.coords``.
        """
        return (
            self.region.coords
            if self.dimensions < 2
            else self.region.exterior.coords
        )

    @property
    def dimensions(self) -> int:
        """Dimensionality of the region."""
        return shapely.get_dimensions(self.region)

    @property
    def is_closed(self) -> bool:
        """Return True if the region is closed.

        A closed region is either:
        - A polygon (2D RoI).
        - A 1D LoI whose final point connects back to its first.
        """
        return self.dimensions > 1 or (
            self.dimensions == 1
            and self.region.coords[0] == self.region.coords[-1]
        )

    @property
    def name(self) -> str:
        """Name of the instance."""
        return self._name if self._name else self.__default_name

    @property
    def region(self) -> SupportedGeometry:
        """``shapely.Geometry`` representation of the region."""
        return self._shapely_geometry

    def __init__(
        self,
        points: PointLikeList,
        dimensions: Literal[1, 2] = 2,
        closed: bool = False,
        holes: Sequence[PointLikeList] | None = None,
        name: str | None = None,
    ) -> None:
        """Initialise a region of interest.

        Parameters
        ----------
        points : Sequence of (x, y) values
            Sequence of (x, y) coordinate pairs that will form the region.
        dimensions : Literal[1, 2], default 2
            The dimensionality of the region to construct.
            '1' creates a sequence of joined line segments,
            '2' creates a polygon whose boundary is defined by ``points``.
        closed : bool, default False
            Whether the line to be created should be closed. That is, whether
            the final point should also link to the first point.
            Ignored if ``dimensions`` is 2.
        holes : sequence of sequences of (x, y) pairs, default None
            A sequence of items, where each item will be interpreted like
            ``points``. These items will be used to construct internal holes
            within the region. See the ``holes`` argument to
            ``shapely.Polygon`` for details. Ignored if ``dimensions`` is 1.
        name : str, default None
            Human-readable name to assign to the given region, for
            user-friendliness. Default name given is 'Un-named region' if no
            explicit name is provided.

        """
        self._name = name
        if len(points) < dimensions + 1:
            raise log_error(
                ValueError,
                f"Need at least {dimensions + 1} points to define a "
                f"{dimensions}D region (got {len(points)}).",
            )
        elif dimensions < 1 or dimensions > 2:
            raise log_error(
                ValueError,
                "Only regions of interest of dimension 1 or 2 are supported "
                f"(requested {dimensions})",
            )
        elif dimensions == 1 and len(points) < 3 and closed:
            raise log_error(
                ValueError,
                "Cannot create a loop from a single line segment.",
            )
        if dimensions == 2:
            self._shapely_geometry = shapely.Polygon(shell=points, holes=holes)
        else:
            self._shapely_geometry = (
                shapely.LinearRing(coordinates=points)
                if closed
                else shapely.LineString(coordinates=points)
            )

    def __repr__(self) -> str:  # noqa: D105
        return str(self)

    def __str__(self) -> str:  # noqa: D105
        display_type = "-gon" if self.dimensions > 1 else " line segment(s)"
        n_points = len(self.coords) - 1
        return (
            f"{self.__class__.__name__} {self.name} "
            f"({n_points}{display_type})\n"
        ) + " -> ".join(f"({c[0]}, {c[1]})" for c in self.coords)

    def _vector_from_centroid_of_keypoints(
        self,
        data: xr.DataArray,
        position_keypoint: Hashable | Sequence[Hashable],
        renamed_dimension: str = "vector to",
        which_method: str = "compute_approach_vector",
        **method_args: Any,
    ) -> xr.DataArray:
        """Compute a vector from the centroid of some keypoints to a target.

        Intended for internal use when calculating ego- and allocentric
        boundary angles. First, the position of the centroid of the given
        keypoints is computed. Then, this value, along with the other keyword
        arguments passed, is given to the method specified in ``which_method``
        in order to compute the necessary vectors.

        Parameters
        ----------
        data : xarray.DataArray
            DataArray of position data.
        position_keypoint : Hashable | Sequence[Hashable]
            Keypoints to compute centroid of, then compute vectors to/from.
        renamed_dimension : str
            The name of the new dimension created by ``which_method`` that
            contains the corresponding vectors. This dimension will be renamed
            to 'space' and given coordinates x, y [, z].
        which_method : str
            Name of a class method, which will be used to compute the
            appropriate vectors.
        method_args : Any
            Additional keyword arguments needed by the specified class method.

        """
        position_data = data.sel(keypoints=position_keypoint, drop=True)
        if "keypoints" in position_data.dims:
            position_data = position_data.mean(dim="keypoints")

        method = getattr(self, which_method)
        return (
            method(
                position_data,
                **method_args,
            )
            .rename({renamed_dimension: "space"})
            .assign_coords(
                {
                    "space": ["x", "y"]
                    if len(data["space"]) == 2
                    else ["x", "y", "z"]
                }
            )
        )

    @broadcastable_method(only_broadcastable_along="space")
    def contains_point(
        self,
        /,
        position: ArrayLike,
        include_boundary: bool = True,
    ) -> bool:
        """Determine if a position is inside the region of interest.

        Parameters
        ----------
        position : ArrayLike
            Spatial coordinates [x, y, [z]] to check as being inside the
            region.
        include_boundary : bool
            Whether to treat a position on the region's boundary as inside the
            region (True) or outside the region (False). Default True.

        Returns
        -------
        bool
            True if the ``position`` provided is within the region of interest.
            False otherwise.

        """
        point = shapely.Point(position)

        current_region = self.region
        point_is_inside = current_region.contains(point)

        if include_boundary:
            # 2D objects have 1D object boundaries,
            # which in turn have point-boundaries.
            while not current_region.boundary.is_empty:
                current_region = current_region.boundary
                point_is_inside = point_is_inside or current_region.contains(
                    point
                )
        return point_is_inside

    @broadcastable_method(only_broadcastable_along="space")
    def compute_distance_to(
        self, point: ArrayLike, boundary_only: bool = False
    ) -> float:
        """Compute the distance from the region to a point.

        Parameters
        ----------
        point : ArrayLike
            Coordinates of a point, from which to find the nearest point in the
            region defined by ``self``.
        boundary_only : bool, optional
            If ``True``, compute the distance from ``point`` to the boundary of
            the region, rather than the closest point belonging to the region.
            Default ``False``.

        Returns
        -------
        float
            Euclidean distance from the ``point`` to the (closest point on the)
            region.

        See Also
        --------
        shapely.distance : Underlying used to compute the nearest point.

        """
        from_where = self.region.boundary if boundary_only else self.region
        return shapely.distance(from_where, shapely.Point(point))

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="nearest point"
    )
    def compute_nearest_point_to(
        self, /, position: ArrayLike, boundary_only: bool = False
    ) -> np.ndarray:
        """Compute a nearest point in the region to the ``position``.

        position : ArrayLike
            Coordinates of a point, from which to find the nearest point in the
            region defined by ``self``.
        boundary_only : bool, optional
            If ``True``, compute the nearest point to ``position`` that is on
            the  boundary of ``self``. Default ``False``.

        Returns
        -------
        np.ndarray
            Coordinates of the point on ``self`` that is closest to
            ``position``.

        See Also
        --------
        shapely.shortest_line : Underlying used to compute the nearest point.

        """
        from_where = self.region.boundary if boundary_only else self.region
        # shortest_line returns a line from 1st arg to 2nd arg,
        # therefore the point on self is the 0th coordinate
        return np.array(
            shapely.shortest_line(from_where, shapely.Point(position)).coords[
                0
            ]
        )

    @broadcastable_method(
        only_broadcastable_along="space", new_dimension_name="vector to"
    )
    def compute_approach_vector(
        self,
        point: ArrayLike,
        boundary_only: bool = False,
        unit: bool = False,
    ) -> np.ndarray:
        """Compute the approach vector from a ``point`` to the region.

        The approach vector is defined as the vector directed from the
        ``point`` provided, to the closest point that belongs to the region.

        Parameters
        ----------
        point : ArrayLike
            Coordinates of a point to compute the vector to (or from) the
            region.
        boundary_only : bool
            If ``True``, the approach vector to the boundary of the region is
            computed. Default ``False``.
        unit : bool
            If ``True``, the approach vector is returned normalised, otherwise
            it is not normalised. Default is ``True``.

        Returns
        -------
        np.ndarray
            Vector directed between the point and the region.

        """
        from_where = self.region.boundary if boundary_only else self.region

        # "point to region" by virtue of order of arguments to shapely call
        directed_line = shapely.shortest_line(shapely.Point(point), from_where)

        displacement_vector = np.array(directed_line.coords[1]) - np.array(
            directed_line.coords[0]
        )
        if unit:
            norm = np.sqrt(np.sum(displacement_vector**2))
            # Cannot normalise the 0 vector
            if not np.isclose(norm, 0.0):
                displacement_vector /= norm
        return displacement_vector

    def compute_allocentric_angle(
        self,
        data: xr.DataArray,
        position_keypoint: Hashable | Sequence[Hashable],
        angle_rotates: Literal[
            "approach to ref", "ref to approach"
        ] = "approach to ref",
        boundary_only: bool = False,
        in_radians: bool = False,
        reference_vector: np.ndarray | xr.DataArray = None,
    ) -> float:
        """Compute the allocentric angle to the region.

        The allocentric angle is the :func:`signed angle\
        <movement.utils.vector.compute_signed_angle_2d>` between the approach
        vector (directed from a point to the region) and a given reference
        vector. `angle_rotates`` can be used to reverse the sign convention of
        the returned angle.

        The approach vector is the vector from ``position_keypoints`` to the
        closest point within the region (or the closest point on the boundary
        of the region if ``boundary`` is set to ``True``), as determined by
        :func:`compute_approach_vector`.

        Parameters
        ----------
        data : xarray.DataArray
            `DataArray` of positions that has at least 3 dimensions; "time",
            "space", and ``keypoints_dimension``.
        position_keypoint : Hashable | Sequence[Hashable]
            The keypoint defining the origin of the approach vector. If
            provided as a sequence, the average of all provided keypoints is
            used.
        angle_rotates : Literal["approach to ref", "ref to approach"]
            Direction of the signed angle returned. Default is
            ``"approach to ref"``.
        boundary_only : bool
            Passed to ``compute_approach_vector``. Default ``False``.
        in_radians : bool
            If ``True``, angles are returned in radians. Otherwise angles are
            returned in degrees. Default ``False``.
        reference_vector : ArrayLike | xr.DataArray
            The reference vector to be used. Dimensions must be compatible with
            the argument of the same name that is passed to
            :func:`compute_signed_angle_2d`. Default ``(1., 0.)``.

        See Also
        --------
        ``compute_signed_angle_2d`` : The underlying function used to compute
        the signed angle between the approach vector and the reference vector.
        ``compute_egocentric_angle`` : Related class method for computing the
        egocentric angle to the region.

        """
        if reference_vector is None:
            reference_vector = np.array([[1.0, 0.0]])
        # Translate the more explicit convention used here into the convention
        # used by our backend functions.
        if angle_rotates == "ref to approach":
            ref_as_left_operand = True
        elif angle_rotates == "approach to ref":
            ref_as_left_operand = False
        else:
            raise ValueError(f"Unknown angle convention: {angle_rotates}")

        # Determine the approach vector, for all time-points.
        approach_vector = self._vector_from_centroid_of_keypoints(
            data,
            position_keypoint=position_keypoint,
            boundary_only=boundary_only,
            unit=False,
        )

        # Then, compute signed angles at all time-points
        angles = compute_signed_angle_2d(
            approach_vector,
            reference_vector,
            v_as_left_operand=ref_as_left_operand,
        )
        if not in_radians:
            angles *= 180.0 / np.pi
        return angles

    def compute_egocentric_angle(
        self,
        data: xr.DataArray,
        left_keypoint: Hashable,
        right_keypoint: Hashable,
        angle_rotates: Literal[
            "approach to forward", "forward to approach"
        ] = "approach to forward",
        boundary_only: bool = False,
        camera_view: Literal["top_down", "bottom_up"] = "top_down",
        in_radians: bool = False,
        position_keypoint: Hashable | Sequence[Hashable] | None = None,
    ) -> xr.DataArray:
        """Compute the egocentric angle to the region.

        The egocentric angle is the signed angle between the approach vector
        (directed from a point towards the region) a forward direction
        (typically of a given individual or keypoint). ``angle_rotates`` can
        be used to reverse the sign convention of the returned angle.

        The forward vector is determined by ``left_keypoint``,
        ``right_keypoint``, and ``camera_view`` as per :func:`forward vector\
        <movement.kinematics.compute_forward_vector>`.

        The approach vector is the vector from ``position_keypoints`` to the
        closest point within the region (or the closest point on the boundary
        of the region if ``boundary`` is set to ``True``), as determined by
        :func:`compute_approach_vector`.

        Parameters
        ----------
        data : xarray.DataArray
            `DataArray` of positions that has at least 3 dimensions; "time",
            "space", and ``keypoints_dimension``.
        left_keypoint : Hashable
            The left keypoint defining the forward vector, as passed to
            func:``compute_forward_vector_angle``.
        right_keypoint : Hashable
            The right keypoint defining the forward vector, as passed to
            func:``compute_forward_vector_angle``.
        angle_rotates : Literal["approach to forward", "forward to approach"]
            Direction of the signed angle returned. Default is
            ``"approach to forward"``.
        boundary_only : bool
            Passed to ``compute_approach_vector`` (see Notes). Default
            ``False``.
        camera_view : Literal["top_down", "bottom_up"]
            Passed to func:`compute_forward_vector_angle`. Default
            ``"top_down"``.
        in_radians : bool
            If ``True``, angles are returned in radians. Otherwise angles are
            returned in degrees. Default ``False``.
        position_keypoint : Hashable | Sequence[Hashable], optional
            The keypoint defining the origin of the approach vector. If
            provided as a sequence, the average of all provided keypoints is
            used. By default, the centroid of ``left_keypoint`` and
            ``right_keypoint`` is used.

        See Also
        --------
        ``compute_forward_vector_angle`` : The underlying function used
            to compute the signed angle between the forward vector and the
            approach vector.

        """
        # Default to centre of left and right keypoints for position,
        # if not provided.
        if position_keypoint is None:
            position_keypoint = [left_keypoint, right_keypoint]
        # Translate the more explicit convention used here into the convention
        # used by our backend functions.
        rotation_angle: Literal["ref to forward", "forward to ref"] = (
            angle_rotates.replace("approach", "ref")  # type: ignore
        )
        if rotation_angle not in ["ref to forward", "forward to ref"]:
            raise ValueError(f"Unknown angle convention: {angle_rotates}")

        # Determine the approach vector, for all time-points.
        approach_vector = self._vector_from_centroid_of_keypoints(
            data,
            position_keypoint=position_keypoint,
            boundary_only=boundary_only,
            unit=False,
        )

        # Then, compute signed angles at all time-points
        return compute_forward_vector_angle(
            data,
            left_keypoint=left_keypoint,
            right_keypoint=right_keypoint,
            reference_vector=approach_vector,
            camera_view=camera_view,
            in_radians=in_radians,
            angle_rotates=rotation_angle,
        )
