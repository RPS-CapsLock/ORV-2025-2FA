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
