import numpy as np
import matplotlib.pyplot as plt
from kymatio import Scattering2D
import torch
from torchvision import datasets, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from  plot_scattering_frequencies import visualize_frequencies




if __name__ == "__main__":

    img_name = os.path.join(os.getcwd(),"kymatio-main","examples","2d","images","digit.png")

    ####################################################################
    # Scattering computations
    #-------------------------------------------------------------------
    # First, we read the input digit:
    image_dim = 64
    image = Image.open(img_name).convert('L').resize((image_dim, image_dim))
    #image = np.array(image)

    # Step 1: Load an MNIST image
    transform = transforms.Compose([
        transforms.Resize((image_dim, image_dim)),
        transforms.ToTensor()
    ])
    src_img_tensor = transform(image)[0]


    mnist_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    rnd = np.random.default_rng()
    rnd_idxes = set((rnd.random(10)*100).round().tolist())
    rnd_idxes = [int(idx) for idx in rnd_idxes]
    images = [mnist_dataset[idx][0][0] for idx in rnd_idxes]

####### unit testing ###########################################
    # M , N = 10, 10
    # images = [torch.zeros(M,N) for _ in range(3)]
    # images[0] = torch.full((10, 10), float(0.0))
    # #initing images[1] so all values are 1.0
    # images[1][0:M, 0:N] = 1.0
    # #initing images[2] to have ones in the middle  and zeros elsewhere
    # images[2] = torch.zeros(M,N)
    # images[2][M//4:M//4*3, N//4:N//4*3] = 1.0
###########################################

    # Step 2: Compute scattering coefficients
    J = 6
    L = 8
    max_order = 1
    print(f"images[0].shape = {images[0].shape}")
    filter_type = 'meyer'
    tighten = False
    scattering = Scattering2D(J=J, shape=images[0].shape, L=L, max_order=max_order, 
                              frontend='torch', out_type="list", model_kind='invertible_scattering',
                              downsample=False, dilation_optimization=False, tighten=tighten, filter_type=filter_type)
    print("finished creating scattering object")
    ####################################################################
    # We now compute the scattering coefficients:




    #src_img_tensor = image.astype(np.float32) / 255.



    selected_indices = [0,1,2]
    for image_idx , image in enumerate(images):
        # plt.figure()
        # plt.title(f"Input Image {image_idx}")
        # plt.imshow(image.cpu().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()

        coefficients = []
        scat_coeffs = scattering(image, downsample=False) 
        print("finished computing scattering coefficients")
        # for i in range(max_order+1):
        #     coeff = [coeff for coeff in scat_coeffs if coeff['depth'] == i][0]  # Get the first coefficient of depth i
        #     visualize_frequencies(coeff['coef'].squeeze().numpy(), coeff['depth'], coeff['n'], image_idx)
        
        # zeroing all coefficients except the first one
        # for i in range (1,len(scat_coeffs)):
        #     if scat_coeffs[i]['depth'] == 0:
        #         print("oops, depth 0 should not be here")
        #     scat_coeffs[i]['coef'] = torch.zeros_like(scat_coeffs[i]['coef'])

        first_coeff = scat_coeffs[0].copy()
        # zeroing first coefficient
        last_layer = scattering.build_actuall_last_layer(scat_coeffs)

        reconstructed_image_last_layer_zeros = scattering.inverse_scattering(scat_coeffs,last_layer=None)[...,0]
        reconstructed_image_last_layer_actual = scattering.inverse_scattering(scat_coeffs,last_layer=last_layer)[...,0]

        scat_coeffs[0]['coef'] = torch.zeros_like(scat_coeffs[0]['coef'])

        reconstructed_image_last_layer_zeros_first_coeff_zero = scattering.inverse_scattering(scat_coeffs,last_layer=None)[...,0]
        reconstructed_image_last_layer_actual_first_coeff_zero = scattering.inverse_scattering(scat_coeffs,last_layer=last_layer)[...,0]

        #saving each reconstructed image and beside it the original image instead of plotting it
        output_dir = os.path.join(os.getcwd(),"kymatio-main","examples","2d","reconstructed_images_new")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Save the original image
        plt.imsave(os.path.join(output_dir, f"original_image_{image_idx}.png"), image.cpu().detach().numpy(), cmap='gray')
        plt.imsave(os.path.join(output_dir, f"reconstructed_image_{image_idx}_{filter_type}_{J}_{L}_last_layer_none.png"), reconstructed_image_last_layer_zeros.cpu().detach().numpy(), cmap='gray')
        plt.imsave(os.path.join(output_dir, f"reconstructed_image_{image_idx}_{filter_type}_{J}_{L}_last_layer_actual.png"), reconstructed_image_last_layer_actual.cpu().detach().numpy(), cmap='gray')
        plt.imsave(os.path.join(output_dir, f"reconstructed_image_{image_idx}_{filter_type}_{J}_{L}_last_layer_none_first_coeff_zero.png"), reconstructed_image_last_layer_zeros_first_coeff_zero.cpu().detach().numpy(), cmap='gray')
        plt.imsave(os.path.join(output_dir, f"reconstructed_image_{image_idx}_{filter_type}_{J}_{L}_last_layer_actual_first_coeff_zero.png"), reconstructed_image_last_layer_actual_first_coeff_zero.cpu().detach().numpy(), cmap='gray')

        # Save a side-by-side comparison of original and reconstructed images with a title
        reconstructions = [
            ("last_layer_none", reconstructed_image_last_layer_zeros),
            ("last_layer_actual", reconstructed_image_last_layer_actual),
            ("last_layer_none_first_coeff_zero", reconstructed_image_last_layer_zeros_first_coeff_zero),
            ("last_layer_actual_first_coeff_zero", reconstructed_image_last_layer_actual_first_coeff_zero)
        ]
        for suffix, recon_img in reconstructions:
            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            fig.suptitle(suffix, fontsize=12)
            axes[0].imshow(image.cpu().detach().numpy(), cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis('off')
            axes[1].imshow(recon_img.cpu().detach().numpy(), cmap='gray')
            axes[1].set_title(f"Reconstructed at depth {max_order}")
            axes[1].axis('off')
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f"comparison_{image_idx}_{suffix}_{filter_type}_{J}_{L}.png"))
            plt.close(fig)

        print(f"saved reconstructed images for image {image_idx} in {output_dir}")
        if image_idx > 0:
            break
        # # Plot the reconstructed image
        # plt.figure()
        # plt.title(f"Reconstructed Image {image_idx}")
        # # If reconstructed_image is a torch tensor, convert to numpy
        # img_to_plot = reconstructed_image.cpu().detach().numpy()
        # plt.imshow(img_to_plot, cmap='gray')
        # plt.colorbar()
        # plt.axis('off')
        # plt.show()


# try to sum all of the squers of the norms of the fiolters and see if its close to 1 (theoretically should be equal to 1)
