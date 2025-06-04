import os
import cv2 as cv
from .augment import *
from .prep import *

def procesiraj_in_augmetiraj(input_dir, output_dir):
    augmentacije = ['original', 'rotacija', 'sum', 'kontrast', 'senca', 'kombinirano']

    for aug in augmentacije:
        os.makedirs(os.path.join(output_dir, aug), exist_ok=True)

    slike = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for slika_ime in slike:
        pot = os.path.join(input_dir, slika_ime)

        slika_rgb = cv.imread(pot)
        slika_gray = cv.cvtColor(slika_rgb, cv.COLOR_BGR2GRAY)

        filtrirana = filtriraj_z_gaussovim_jedrom(slika_gray, sigma=1)
        filtrirana = cv.normalize(filtrirana, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        linearizirana = linearizacija_sivin(filtrirana)

        # out_path = os.path.join(output_dir, 'original', slika_ime)
        # cv.imwrite(out_path, linearizirana)
        #
        # rotirana = rotiraj_random(linearizirana, max_kot=15)
        # out_path = os.path.join(output_dir, 'rotacija', slika_ime)
        # cv.imwrite(out_path, rotirana)
        #
        # suma_slika = dodaj_gaussov_sum(linearizirana, sigma=25)
        # out_path = os.path.join(output_dir, 'sum', slika_ime)
        # cv.imwrite(out_path, suma_slika)
        #
        # kontrastirana = spremeni_kontrast(linearizirana, 1.5)
        # out_path = os.path.join(output_dir, 'kontrast', slika_ime)
        # cv.imwrite(out_path, kontrastirana)
        #
        # senca = add_shadow_simple(linearizirana)
        # out_path = os.path.join(output_dir, 'senca', slika_ime)
        # cv.imwrite(out_path, senca)

        kombinirana = add_shadow_simple(
            spremeni_kontrast(
                dodaj_gaussov_sum(
                    rotiraj_random(linearizirana, max_kot=15), sigma=25), 1.5))
        kombinirana_bgr = cv.cvtColor(kombinirana, cv.COLOR_GRAY2BGR)
        out_path = os.path.join(output_dir, 'kombinirano', slika_ime)
        cv.imwrite(out_path, kombinirana_bgr)

    if __name__ == '__main__':
        input_dir = "../data/raw"
        output_dir = "../data/processed"

        procesiraj_in_augmetiraj(input_dir, output_dir)