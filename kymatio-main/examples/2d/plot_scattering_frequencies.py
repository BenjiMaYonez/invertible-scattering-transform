import numpy as np
import matplotlib.pyplot as plt
from kymatio import Scattering2D
import torch
from torchvision import datasets, transforms
from PIL import Image
import os

def visualize_frequencies(image, depth, coeff_idx, image_idx):
    """
    Takes a grayscale image as a numpy array and visualizes its frequency components
    using the Fourier Transform. Adds a scale and axes emphasizing the center point.
    Saves the plot to a directory.
    """
    # Ensure the image is a 2D numpy array
    if len(image.shape) != 2:
        raise ValueError("Input image must be a 2D numpy array representing a grayscale image.")

    # Compute the 2D Fourier Transform of the image
    f_transform = np.fft.fft2(image)

    # Shift the zero frequency component to the center
    f_shift = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magnitude_spectrum = np.abs(f_shift)
    magnitude_spectrum_log = np.log1p(magnitude_spectrum)

    # Prepare directory for saving plots
    output_dir = "coefficients_plot"
    os.makedirs(output_dir, exist_ok=True)

    # Define the file name
    file_name = f"image_{image_idx}_coefficient_{coeff_idx}_depth_{depth}.png"
    file_path = os.path.join(output_dir, file_name)

    # Plot the results
    plt.figure(figsize=(15, 5))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', interpolation='none')
    plt.title(f'Original Coefficient : image index : {image_idx} depth {depth} index : {coeff_idx}')
    plt.axis('off')

    # Plot the magnitude spectrum with a colorbar and emphasized axes
    ax = plt.subplot(1, 2, 2)
    im = ax.imshow(magnitude_spectrum_log, cmap='gray', interpolation='none', extent=(-image.shape[1]//2, image.shape[1]//2, -image.shape[0]//2, image.shape[0]//2))
    ax.set_title('Magnitude Spectrum')
    ax.set_xlabel("Frequency X-axis")
    ax.set_ylabel("Frequency Y-axis")
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8)  # Emphasize Y=0
    ax.axvline(0, color='red', linestyle='--', linewidth=0.8)  # Emphasize X=0
    plt.colorbar(im, ax=ax, orientation='vertical', label='Log Magnitude')  # Add color scale

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {file_path}")


if __name__ == "__main__":

    img_name = os.path.join(os.getcwd(), "kymatio-main\examples\\2d\images\digit.png")

    ####################################################################
    # Scattering computations
    #-------------------------------------------------------------------
    # First, we read the input digit:
    image_dim = 64
    image = Image.open(img_name).convert('L').resize((image_dim, image_dim))
    #image = np.array(image)

    # Step 1: Load an MNIST image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    src_img_tensor = transform(image)[0]


    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    rnd = np.random.default_rng()
    rnd_idxes = set((rnd.random(10)*100).round().tolist())
    rnd_idxes = [int(idx) for idx in rnd_idxes]
    images = [mnist_dataset[idx][0][0] for idx in rnd_idxes]


    # Display the original MNIST image
    # plt.figure(figsize=(2, 2))
    # plt.imshow(images[0], cmap='gray')
    # plt.title('Original MNIST Image')
    # plt.axis('off')
    # plt.show()

    # Step 2: Compute scattering coefficients
    L = 8
    J = 3
    max_order = 3
    scattering = Scattering2D(J=J, shape=images[0].shape, L=L, max_order=max_order, frontend='torch', out_type="list", model_kind='invertible_scattering')

    ####################################################################
    # We now compute the scattering coefficients:




    #src_img_tensor = image.astype(np.float32) / 255.

    selected_indices = [0,1,2]
    for image_idx , image in enumerate(images):
        coefficients = []
        scat_coeffs = scattering(image)
        i = 0
        for coeff in scat_coeffs:
            if coeff['depth'] == max_order:
                i += 1
                visualize_frequencies(coeff['coef'].numpy(), max_order, i, image_idx)
                if i == 3:
                    break
                #coefficients.append(coeff['coef'].numpy())

     # print("coeffs shape: ", scat_coeffs.shape)


        ####################################################################
        # There are 127 scattering coefficients, among which 1 is low-pass, $JL=18$ are of first-order and $L^2(J(J-1)/2)=108$
        # are of second-order. Due to the subsampling by $2^J=8$, the final spatial grid is of size $4\times4$.
        # We now retrieve first-order and second-order coefficients for the display.
        # len_order_1 = J*L
        # scat_coeffs_order_1 = scat_coeffs[1:1+len_order_1, :, :]

        # len_order_2 = (J*(J-1)//2)*(L**2)
        # scat_coeffs_order_2 = scat_coeffs[1+len_order_1:, :, :]

        # coefficients = scat_coeffs_order_2
        #coefficients = scat_coeffs

        # window_rows, window_columns = coefficients[0].shape

        # num_coefficients = len(coefficients)
        # print("Number of coefficients (channels):", num_coefficients)

        # # For demonstration, let's select a few coefficients
        # selected_indices = [0, num_coefficients // 2, num_coefficients - 1]  # First, middle, last coefficients

        # # Step 4: Analyze each selected coefficient
        # for idx in selected_indices:
        #     coeff = coefficients[idx]
        #     visualize_frequencies(coeff, max_order, idx)

