import math
import cv2 as cv
import numpy as np

def konvolucija(slika, jedro):
    visina_slika, sirina_slika = slika.shape
    visina_jedro, sirina_jedro = jedro.shape

    x_odmik = visina_jedro // 2
    y_odmik = sirina_jedro // 2

    razsirjena = np.pad(slika, ((x_odmik, x_odmik), (y_odmik, y_odmik)), mode='median')
    rezultat = np.zeros_like(slika, dtype=np.float32)

    for i in range(visina_slika):
        for j in range(sirina_slika):
            izsek = razsirjena[i:i + visina_jedro, j:j + visina_jedro]
            rezultat[i, j] = np.sum(izsek * jedro)

    return rezultat

def filtriraj_z_gaussovim_jedrom(slika, sigma):
    slika = slika.copy()

    velikost_jedra = (2 * sigma) * 2 + 1
    k = velikost_jedra / 2 - 0.5

    jedro = np.zeros((math.ceil(velikost_jedra), math.ceil(velikost_jedra)))
    for i in range(math.ceil(velikost_jedra)):
        for j in range(math.ceil(velikost_jedra)):
            jedro[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-((i - k) ** 2 + (j - k) ** 2) / (2 * sigma ** 2))

    jedro /= np.sum(jedro)
    return konvolucija(slika, jedro)

def linearizacija_sivin(slika):
    min_val = np.min(slika)
    max_val = np.max(slika)
    if max_val - min_val == 0:
        return slika.copy()

    raztegnjena = (slika - min_val) * (255.0 / (max_val - min_val))

    return raztegnjena.astype(np.uint8)

if __name__ == '__main__':
    slika_rgb = cv.imread("../data/raw/test_face.png")

    slika = cv.cvtColor(slika_rgb, cv.COLOR_BGR2GRAY)
    slika_hsv = cv.cvtColor(slika_rgb, cv.COLOR_BGR2HSV)

    gauss = filtriraj_z_gaussovim_jedrom(slika, 1)
    gauss = cv.normalize(gauss, None, 0, 255, cv.NORM_MINMAX)
    gauss = gauss.astype(np.uint8)

    linearizirana = linearizacija_sivin(gauss)

    hsv_to_bgr = cv.cvtColor(slika_hsv, cv.COLOR_HSV2BGR)

    cv.imshow('Original Gray', slika)
    cv.imshow('Po Gaussu', gauss)
    cv.imshow('Po Linearizaciji', linearizirana)
    cv.imshow('HSV (v BGR)', hsv_to_bgr)

    cv.waitKey(0)
    cv.destroyAllWindows()