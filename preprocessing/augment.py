import cv2 as cv
import numpy as np
import random
import math

def rotiraj_random(slika, max_kot=15):
    kot = random.uniform(-max_kot, max_kot)
    visina, sirina = slika.shape[:2]
    center = (sirina // 2, visina // 2)

    matrika_rotacije = cv.getRotationMatrix2D(center, kot, 1.0)
    rotirana = cv.warpAffine(slika, matrika_rotacije, (sirina, visina),
                             flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT)

    return rotirana

def dodaj_gaussov_sum(slika, sigma=25):
    slika = slika.astype(np.float32)
    gauss = np.random.normal(0, sigma, slika.shape).astype(np.float32)
    suma_slika = slika + gauss
    suma_slika = np.clip(suma_slika, 0, 255)

    return suma_slika.astype(np.uint8)