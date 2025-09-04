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

    J=6
    L=8
    max_order = 1
    scattering = Scattering2D(J=J, shape=(256,256), L=L, max_order=max_order,
                               frontend='torch', out_type="list", model_kind='invertible_scattering',
                               downsample=False, dilation_optimization=False, tighten=False, filter_type='meyer')
                            #    sigma0=1.2, theta0=0.0, xi0=0.9, slant0=0.7)

    output_dir = os.path.join(os.getcwd(),"kymatio-main","examples","2d","filters_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    phi,psi = scattering.get_filters()
    phi0 =  phi['levels'][0] 
    print("phi0 level min/max =", float(phi0.real.min()), float(phi0.real.max()))

    phi0_np = phi0.cpu().detach().numpy() if hasattr(phi0, 'cpu') else np.array(phi0)
    LP = np.abs(phi0_np)**2
    for w in psi:
        if len(w['levels']) > 0:
            w0 = w['levels'][0]
            w0_np = w0.cpu().detach().numpy() if hasattr(w0, 'cpu') else np.array(w0)
            LP += np.abs(w0_np)**2

    print("LP min/max/mean abs dev from 1:",
        LP.min(), LP.max(), float(np.mean(np.abs(LP-1))))


    sum_squers = np.abs(phi['levels'][0]).squeeze()**2
    filters_to_plot = [("low pass filter", phi['levels'][0].real.squeeze())]
    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        theta1 = psi[n1]['theta']
        filter = psi[n1]['levels'][0].squeeze()
        sum_squers += np.abs(filter)**2
        filters_to_plot.append((f"j={j1}, θ={theta1}", filter.real))
    filters_to_plot.append(("sum of squares", sum_squers.real))

    max_per_plot = 10
    num_batches = int(np.ceil(len(filters_to_plot) / max_per_plot))

    for batch_idx in range(num_batches):
        start = batch_idx * max_per_plot
        end = min((batch_idx + 1) * max_per_plot, len(filters_to_plot))
        batch = filters_to_plot[start:end]
        ncols = 4
        nrows = int(np.ceil(len(batch) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        axes = axes.flatten()
        for i, (title, filt) in enumerate(batch):
            im = axes[i].imshow(np.fft.fftshift(filt.cpu().detach().numpy()), cmap='gray', vmin=0, vmax=1)
            axes[i].set_title(title)
            axes[i].axis('on')
            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        # Hide unused subplots
        for i in range(len(batch), len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"filters_grid_batch{batch_idx+1}.png"))
        plt.show()

    # plotting sum of squares only
    plt.figure(figsize=(6, 6))
    im_sum2 = plt.imshow(np.fft.fftshift(sum_squers.real.cpu().numpy()), cmap='gray', vmin=0, vmax=sum_squers.real.max().cpu().detach().numpy())
    plt.title("Sum of Squares of All Filters")
    plt.axis('on')
    plt.colorbar(im_sum2, fraction=0.046, pad=0.04)
    plt.show()

