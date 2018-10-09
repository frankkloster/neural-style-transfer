import argparse

from optimization import run_style_transfer
from images import imsave

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--content_path", type=str, help="Path to desired content image.")
    parser.add_argument("-s", "--style_path", type=str, help="Path to desired style image.")
    parser.add_argument("-a", "--alpha", type=float, help="Desired content weight.", default=1e3)
    parser.add_argument("-b", "--beta", type=float, help="Desired style weight.", default=1e-2)
    parser.add_argument("-f", "--final", type=str, help="Desired final image to save result.")

    args=parser.parse_args()

    content_path = args.content_path
    style_path = args.style_path
    content_weight = args.alpha
    style_weight = args.beta
    final_path = args.final

    best, best_loss = run_style_transfer(content_path=content_path, 
                                     style_path=style_path, num_iterations=1000,
                                     content_weight=content_weight, style_weight=style_weight)

    imsave(best, final)