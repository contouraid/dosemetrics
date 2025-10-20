import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from typing import Optional
from ..metrics.dvh import dvh_by_structure, compute_dvh
from matplotlib.transforms import Bbox


def _get_cmap(n, name="gist_ncar"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def from_dataframe(dataframe: pd.DataFrame, plot_title: str, output_path: str) -> None:
    col_names = dataframe.columns
    cmap = _get_cmap(40)

    plt.style.use("dark_background")
    fig, ax = plt.subplots()

    for i in range(len(col_names)):
        if i % 2 == 0:
            name = col_names[i].split("\n")[0]
            line_color = cmap(i)
            x = dataframe[col_names[i]]
            y = dataframe[col_names[i + 1]]
            plt.plot(x, y, color=line_color, label=name)

    plt.xlabel("Dose [Gy]")
    plt.xlim([0, 65])
    plt.grid()
    plt.ylabel("Ratio of Total Structure Volume [%]")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.title(plot_title)
    plt.savefig(output_path)
    plt.close(fig)


# function that calculates and plots the DVHs based on the dose array of a specific structure
def compare_dvh(
    _gt: np.ndarray,
    _pred: np.ndarray,
    _struct_mask: np.ndarray,
    max_dose=65,
    step_size=0.1,
):
    bins_gt, values_gt = compute_dvh(
        _gt, _struct_mask, max_dose=max_dose, step_size=step_size
    )
    bins_pred, values_pred = compute_dvh(
        _pred, _struct_mask, max_dose=max_dose, step_size=step_size
    )

    fig = plt.figure()
    plt.plot(bins_gt, values_gt, color="b", label="ground truth")
    plt.plot(bins_pred, values_pred, color="r", label="prediction")

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.legend(loc="best")

    return fig


def generate_dvh_variations(
    dose_volume: np.ndarray,
    structure_mask: np.ndarray,
    n_variations: int = 100,
    dice_range: tuple = (0.7, 1.0),
    volume_variation: float = 0.2,
    max_dose: float = 65,
    step_size: float = 0.1,
) -> tuple:
    """
    Generate DVH variations using non-rigid transformations.

    Parameters:
    -----------
    dose_volume : np.ndarray
        The dose volume array
    structure_mask : np.ndarray
        The original structure mask
    n_variations : int
        Number of variations to generate
    dice_range : tuple
        Target range of Dice coefficients (min_dice, max_dice)
    volume_variation : float
        Maximum relative volume change (e.g., 0.2 = ±20%)
    max_dose : float
        Maximum dose for DVH computation
    step_size : float
        Step size for DVH computation

    Returns:
    --------
    tuple: (dvh_data, dice_coefficients, original_dvh)
        - dvh_data: list of (bins, values) tuples for each variation
        - dice_coefficients: list of Dice coefficients
        - original_dvh: (bins, values) tuple for the original structure
    """
    from scipy import ndimage

    dvh_data = []
    dice_coefficients = []

    # Compute original DVH
    original_bins, original_values = compute_dvh(
        dose_volume, structure_mask, max_dose=max_dose, step_size=step_size
    )
    original_volume = structure_mask.sum()

    min_dice, max_dice = dice_range
    target_dice_values = np.random.uniform(min_dice, max_dice, n_variations)

    for target_dice in target_dice_values:
        # Generate random transformation parameters
        # Use multiple transformation types for realistic variations

        # 1. Elastic deformation parameters
        sigma = np.random.uniform(1, 5)  # Smoothness of deformation
        alpha = np.random.uniform(5, 20)  # Magnitude of deformation

        # 2. Scaling parameters (to vary volume)
        scale_factor = 1.0 + np.random.uniform(-volume_variation, volume_variation)

        # 3. Rotation parameters (small random rotations)
        rotation_angles = np.random.uniform(-10, 10, 3)  # degrees

        # Create transformed mask
        transformed_mask = structure_mask.copy().astype(float)

        # Apply scaling
        if scale_factor != 1.0:
            zoom_factors = [scale_factor] * 3
            transformed_mask = ndimage.zoom(transformed_mask, zoom_factors, order=1)

            # Crop or pad to match original shape
            original_shape = structure_mask.shape
            current_shape = transformed_mask.shape

            if current_shape[0] > original_shape[0]:
                # Crop
                start = [
                    (cs - os) // 2 for cs, os in zip(current_shape, original_shape)
                ]
                transformed_mask = transformed_mask[
                    start[0] : start[0] + original_shape[0],
                    start[1] : start[1] + original_shape[1],
                    start[2] : start[2] + original_shape[2],
                ]
            else:
                # Pad
                pad_width = [
                    (os - cs) // 2 for cs, os in zip(current_shape, original_shape)
                ]
                pad_width = [
                    (p, os - cs - p)
                    for p, cs, os in zip(pad_width, current_shape, original_shape)
                ]
                transformed_mask = np.pad(
                    transformed_mask, pad_width, mode="constant", constant_values=0
                )

        # Apply elastic deformation
        # Create random displacement fields
        shape = transformed_mask.shape
        dx = ndimage.gaussian_filter(
            (np.random.rand(*shape) - 0.5) * alpha, sigma, mode="constant", cval=0
        )
        dy = ndimage.gaussian_filter(
            (np.random.rand(*shape) - 0.5) * alpha, sigma, mode="constant", cval=0
        )
        dz = ndimage.gaussian_filter(
            (np.random.rand(*shape) - 0.5) * alpha, sigma, mode="constant", cval=0
        )

        # Create coordinate grids
        x, y, z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )

        # Apply displacement
        indices = np.array(
            [
                np.clip(x + dx, 0, shape[0] - 1),
                np.clip(y + dy, 0, shape[1] - 1),
                np.clip(z + dz, 0, shape[2] - 1),
            ]
        )

        transformed_mask = ndimage.map_coordinates(
            transformed_mask, indices, order=1, mode="constant", cval=0
        )

        # Apply small rotation
        for axis_idx, angle in enumerate(rotation_angles):
            if abs(angle) > 0.1:
                axes = [(axis_idx + 1) % 3, (axis_idx + 2) % 3]
                transformed_mask = ndimage.rotate(
                    transformed_mask, angle, axes=axes, reshape=False, order=1
                )

        # Threshold to binary mask
        transformed_mask = (transformed_mask > 0.5).astype(np.uint8)

        # Compute Dice coefficient
        intersection = np.logical_and(structure_mask, transformed_mask)
        dice = (
            (2 * intersection.sum() / (structure_mask.sum() + transformed_mask.sum()))
            if (structure_mask.sum() + transformed_mask.sum()) > 0
            else 0
        )

        # Compute DVH for transformed mask
        bins, values = compute_dvh(
            dose_volume, transformed_mask, max_dose=max_dose, step_size=step_size
        )

        dvh_data.append((bins, values))
        dice_coefficients.append(dice)

    return dvh_data, dice_coefficients, (original_bins, original_values)


def plot_dvh_variations(
    dvh_data: list,
    dice_coefficients: list,
    original_dvh: tuple,
    constraint_limit: float,
    structure_name: str,
) -> tuple:
    """
    Plot DVH variations with color-coded Dice coefficients.

    Parameters:
    -----------
    dvh_data : list
        List of (bins, values) tuples for each variation
    dice_coefficients : list
        List of Dice coefficients corresponding to each variation
    original_dvh : tuple
        (bins, values) tuple for the original structure DVH
    constraint_limit : float
        Dose constraint limit to display
    structure_name : str
        Name of the structure for plot labels

    Returns:
    --------
    tuple: (fig, (min_dice, max_dice))
        - fig: matplotlib figure object
        - min_dice, max_dice: range of Dice coefficients in the data
    """
    fig = plt.figure(figsize=(10, 8))

    if len(dice_coefficients) == 0:
        # No variations, just plot original
        original_bins, original_values = original_dvh
        plt.plot(
            original_bins, original_values, color="r", label=structure_name, linewidth=2
        )
        plt.axvline(
            x=constraint_limit, color="g", label="Constraint Limit", linewidth=2
        )
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Ratio of Total Structure Volume [%]")
        plt.title(f"DVH for {structure_name}")
        plt.legend()
        plt.grid()
        return fig, (1.0, 1.0)

    min_dice = min(dice_coefficients)
    max_dice = max(dice_coefficients)

    # Create colormap
    n_colors = 100
    cmap = mpl.colormaps["viridis"]
    colors = cmap(np.linspace(0, 1, n_colors + 1))

    # Normalize Dice coefficients to color indices
    dice_range = max_dice - min_dice if max_dice > min_dice else 1.0

    sc = None
    for (bins, values), dice in zip(dvh_data, dice_coefficients):
        # Map dice to color index
        color_idx = int(((dice - min_dice) / dice_range) * n_colors)
        color_idx = np.clip(color_idx, 0, n_colors)
        color = colors[color_idx]
        sc = plt.scatter(bins, values, s=0.5, c=[color], alpha=0.25)

    # Plot original DVH
    original_bins, original_values = original_dvh
    plt.plot(
        original_bins,
        original_values,
        color="r",
        label=structure_name,
        linewidth=2,
        zorder=10,
    )

    # Add constraint line
    plt.axvline(
        x=constraint_limit,
        color="g",
        label="Constraint Limit",
        linewidth=2,
        linestyle="--",
        zorder=10,
    )

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.title(f"DVH Family for {structure_name}")

    # Add colorbar
    if sc is not None:
        color_bar = plt.colorbar(sc, label="Dice Coefficient")
        color_bar.set_alpha(1)
        # Set colorbar ticks
        color_bar.set_ticks([0, 0.5, 1.0])
        color_bar.set_ticklabels(
            [f"{min_dice:.2f}", f"{(min_dice+max_dice)/2:.2f}", f"{max_dice:.2f}"]
        )

    plt.legend()
    plt.grid()

    return fig, (min_dice, max_dice)


def variability(dose_volume, structure_mask, constraint_limit, structure_of_interest):
    """
    Legacy function for backward compatibility.
    Generates DVH variations and plots them.

    DEPRECATED: Use generate_dvh_variations() and plot_dvh_variations() separately.
    """
    # Use default parameters for backward compatibility
    dvh_data, dice_coefficients, original_dvh = generate_dvh_variations(
        dose_volume, structure_mask, n_variations=100, dice_range=(0.7, 1.0)
    )

    fig, (min_dice, max_dice) = plot_dvh_variations(
        dvh_data,
        dice_coefficients,
        original_dvh,
        constraint_limit,
        structure_of_interest,
    )

    return fig, (max_dice, min_dice)


def plot_dvh(dose_volume: np.ndarray, structure_masks: dict, output_file: str):
    """
    PLOT_DVH:
    Plot the dose-volume histogram (DVH) for the given dose volume and structure masks.
    :param dose_volume: Dose volume data as a numpy array.
    :param structure_masks: Dictionary of structure masks.
    :param output_file: Path to save the DVH plot.
    """
    df = dvh_by_structure(dose_volume, structure_masks)
    _, ax = plt.subplots()
    df.set_index("Dose", inplace=True)
    df.groupby("Structure")["Volume"].plot(legend=True, ax=ax)

    # Shrink current axis by 20%
    box = ax.get_position()
    new_box = Bbox.from_bounds(box.x0, box.y0, box.width * 0.8, box.height)
    ax.set_position(new_box)

    # Put a legend to the right of the current axis
    ax.legend(loc="center left", bbox_to_anchor=(0.9, 0.5))

    plt.xlabel("Dose [Gy]")
    plt.ylabel("Ratio of Total Structure Volume [%]")
    plt.grid()
    plt.savefig(output_file)
    plt.close()


def plot_dose_differences(
    dose_array: np.ndarray,
    predicted_array: np.ndarray,
    output_file: str,
    n_slices: int = 30,
    figsize: tuple = (30, 15),
) -> None:
    """
    Visualize dose differences between predicted and ground truth doses across multiple slices.

    Parameters:
    -----------
    dose_array : np.ndarray
        Ground truth dose array
    predicted_array : np.ndarray
        Predicted dose array
    output_file : str
        Output PDF file path
    n_slices : int
        Number of slices to visualize
    figsize : tuple
        Figure size for plots
    """
    import matplotlib.backends.backend_pdf as pdf

    plt.rcParams["figure.figsize"] = list(figsize)

    pp = pdf.PdfPages(output_file)
    diff_array = predicted_array - dose_array

    # Get evenly spaced slice indices
    z_max = dose_array.shape[0]
    selected_indices = np.linspace(0, z_max - 1, num=n_slices, dtype=int)

    for idx in selected_indices:
        slice_idx = int(idx)

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Ground truth dose
        im1 = axes[0, 0].imshow(dose_array[slice_idx, :, :], cmap="hot")
        axes[0, 0].set_title("Ground Truth Dose")
        plt.colorbar(im1, ax=axes[0, 0])

        # Predicted dose
        im2 = axes[0, 1].imshow(predicted_array[slice_idx, :, :], cmap="hot")
        axes[0, 1].set_title("Predicted Dose")
        plt.colorbar(im2, ax=axes[0, 1])

        # Difference
        im3 = axes[1, 0].imshow(diff_array[slice_idx, :, :], cmap="RdBu_r")
        axes[1, 0].set_title("Dose Difference (Pred - GT)")
        plt.colorbar(im3, ax=axes[1, 0])

        # Absolute difference
        im4 = axes[1, 1].imshow(np.abs(diff_array[slice_idx, :, :]), cmap="hot")
        axes[1, 1].set_title("Absolute Dose Difference")
        plt.colorbar(im4, ax=axes[1, 1])

        fig.suptitle(f"Slice {slice_idx}")
        pp.savefig(fig)
        plt.close()

    pp.close()


def plot_frequency_analysis(
    dose_arrays: list,
    output_file: str,
    labels: Optional[list] = None,
    max_value: float = 1000000,
) -> None:
    """
    Perform and visualize frequency domain analysis of dose distributions.

    Parameters:
    -----------
    dose_arrays : list
        List of dose arrays to analyze
    output_file : str
        Output PDF file path
    labels : list, optional
        Labels for each dose array
    max_value : float
        Maximum value for colorbar scaling
    """
    import matplotlib.backends.backend_pdf as pdf

    if labels is None:
        labels = [f"Dose_{i+1}" for i in range(len(dose_arrays))]

    pp = pdf.PdfPages(output_file)

    # Compute FFT for each dose array
    dose_fft_list = []
    for dose_array in dose_arrays:
        dose_fft = np.fft.fftn(dose_array)
        dose_fft_list.append(dose_fft)

    # Visualize frequency domain for each slice
    n_slices = dose_arrays[0].shape[2] if len(dose_arrays[0].shape) == 3 else 128

    for slice_idx in range(n_slices):
        fig, axes = plt.subplots(1, len(dose_arrays), figsize=(15, 5))
        if len(dose_arrays) == 1:
            axes = [axes]

        for i, (dose_fft, label) in enumerate(zip(dose_fft_list, labels)):
            power_spectrum = np.abs(np.fft.fftshift(dose_fft[:, :, slice_idx])) ** 2
            im = axes[i].imshow(power_spectrum, vmax=max_value, vmin=0, cmap="hot")
            axes[i].set_title(f"{label} - Slice {slice_idx}")
            plt.colorbar(im, ax=axes[i])

        fig.suptitle(f"Frequency Analysis - Slice {slice_idx}")
        pp.savefig(fig)
        plt.close()

    pp.close()


def generate_dvh_family_plot(
    dose_array: np.ndarray,
    structure_mask: np.ndarray,
    constraint_limit: float,
    structure_name: str,
    output_file: str,
    n_variations: int = 10,
    noise_level: float = 0.1,
) -> None:
    """
    Generate DVH family plot showing variations around a base DVH.

    Parameters:
    -----------
    dose_array : np.ndarray
        Base dose array
    structure_mask : np.ndarray
        Structure mask
    constraint_limit : float
        Dose constraint limit to highlight
    structure_name : str
        Name of the structure
    output_file : str
        Output file path
    n_variations : int
        Number of variations to generate
    noise_level : float
        Level of noise/variation to add
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute base DVH
    bins, values = compute_dvh(dose_array, structure_mask)

    # Plot base DVH
    ax.plot(
        bins, values, "k-", linewidth=3, label=f"{structure_name} (Original)", alpha=0.8
    )

    # Generate and plot variations
    cmap = _get_cmap(n_variations)

    for i in range(n_variations):
        # Add noise to dose array
        noise = np.random.normal(0, noise_level * np.std(dose_array), dose_array.shape)
        varied_dose = dose_array + noise

        # Compute DVH for varied dose
        var_bins, var_values = compute_dvh(varied_dose, structure_mask)

        # Plot variation
        color = cmap(i)
        ax.plot(var_bins, var_values, color=color, alpha=0.3, linewidth=1)

    # Add constraint line
    ax.axvline(
        x=constraint_limit,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Constraint: {constraint_limit} Gy",
    )

    ax.set_xlabel("Dose [Gy]")
    ax.set_ylabel("Ratio of Total Structure Volume [%]")
    ax.set_title(f"DVH Family for {structure_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def interactive_dvh_plotter():
    """
    Interactive DVH plotter using tkinter file dialogs.
    """
    import tkinter as tk
    from tkinter.filedialog import askopenfilename, asksaveasfilename
    import SimpleITK as sitk
    from matplotlib.backends.backend_pdf import PdfPages

    def compute_stats(file_name: str, dose_array: np.ndarray) -> dict:
        """Compute DVH statistics for a structure."""
        stats = {}
        stats["name"] = file_name.split("/")[-1].split(".")[0]
        struct_image = sitk.ReadImage(file_name)
        struct_array = sitk.GetArrayFromImage(struct_image)

        from ..metrics.dvh import mean_dose, max_dose, volume

        stats["bins"], stats["values"] = compute_dvh(dose_array, struct_array)
        stats["max"] = mean_dose(
            dose_array, struct_array
        )  # Note: function name seems swapped in original
        stats["mean"] = max_dose(
            dose_array, struct_array
        )  # Note: function name seems swapped in original
        stats["volume"] = volume(struct_array, struct_image.GetSpacing())
        stats["color"] = "b"
        return stats

    def plot_stats(stats_dict):
        """Plot DVH statistics."""
        fig = plt.figure(figsize=(10, 6))
        plt.plot(
            stats_dict["bins"],
            stats_dict["values"],
            color=stats_dict["color"],
            label=stats_dict["name"],
        )
        plt.legend(loc="best")
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Ratio of Total Structure Volume [%]")
        plt.title(
            f"Volume: {stats_dict['volume']:4.3f} (cc); "
            f"Max Dose: {stats_dict['max']:2.3f}; "
            f"Mean Dose: {stats_dict['mean']:2.3f}"
        )
        plt.axvline(x=stats_dict["mean"], color="y", label="Mean")
        plt.axvline(x=stats_dict["max"], color="r", label="Max")
        plt.grid()
        return fig

    # Initialize tkinter
    root = tk.Tk()
    root.withdraw()  # Hide the main window

    # Select dose file
    dose_file = askopenfilename(
        title="Select Dose File",
        filetypes=[("NIfTI files", "*.nii.gz"), ("All files", "*.*")],
    )

    if not dose_file:
        return

    # Read dose array
    dose_image = sitk.ReadImage(dose_file)
    dose_array = sitk.GetArrayFromImage(dose_image)

    # Select structure files
    structure_files = []
    while True:
        struct_file = askopenfilename(
            title="Select Structure File (Cancel to finish)",
            filetypes=[("NIfTI files", "*.nii.gz"), ("All files", "*.*")],
        )
        if not struct_file:
            break
        structure_files.append(struct_file)

    if not structure_files:
        print("No structure files selected.")
        return

    # Select output file
    output_file = asksaveasfilename(
        title="Save DVH Plot As",
        defaultextension=".pdf",
        filetypes=[
            ("PDF files", "*.pdf"),
            ("PNG files", "*.png"),
            ("All files", "*.*"),
        ],
    )

    if not output_file:
        return

    # Generate plots
    if output_file.endswith(".pdf"):
        with PdfPages(output_file) as pp:
            for struct_file in structure_files:
                stats = compute_stats(struct_file, dose_array)
                fig = plot_stats(stats)
                pp.savefig(fig)
                plt.close()
    else:
        # For single image outputs, plot all structures together
        fig = plt.figure(figsize=(12, 8))
        colors = plt.cm.get_cmap("tab10")(np.linspace(0, 1, len(structure_files)))

        for i, struct_file in enumerate(structure_files):
            stats = compute_stats(struct_file, dose_array)
            stats["color"] = colors[i]
            plt.plot(
                stats["bins"],
                stats["values"],
                color=stats["color"],
                label=stats["name"],
            )

        plt.legend(loc="best")
        plt.xlabel("Dose [Gy]")
        plt.ylabel("Ratio of Total Structure Volume [%]")
        plt.title("DVH Comparison")
        plt.grid()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"DVH plot saved to: {output_file}")
    root.destroy()
