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

def spremeni_kontrast(slika, faktor):
    slika = slika.astype(np.float32)
    sredina = 127
    nova_slika = (slika - sredina) * faktor + sredina
    nova_slika = np.clip(nova_slika, 0, 255)

    return nova_slika.astype(np.uint8)

def add_shadow_simple(img):
    img_shadow = img.copy()
    h, w = img.shape[:2]

    if random.choice(['horizontal', 'vertical']) == 'horizontal':
        y1 = random.randint(0, h // 2)
        y2 = y1 + random.randint(h // 8, h // 3)
        if len(img.shape) == 2:
            img_shadow[y1:y2, :] = (img_shadow[y1:y2, :] * 0.5).astype(np.uint8)
        else:
            img_shadow[y1:y2, :, :] = (img_shadow[y1:y2, :, :] * 0.5).astype(np.uint8)
    else:
        x1 = random.randint(0, w // 2)
        x2 = x1 + random.randint(w // 8, w // 3)
        if len(img.shape) == 2:
            img_shadow[:, x1:x2] = (img_shadow[:, x1:x2] * 0.5).astype(np.uint8)
        else:
            img_shadow[:, x1:x2, :] = (img_shadow[:, x1:x2, :] * 0.5).astype(np.uint8)

    return img_shadow

if __name__ == '__main__':
    slika = cv.imread("../data/raw/test_face.png")

    rotirana = rotiraj_random(slika, max_kot=15)
    z_sumom = dodaj_gaussov_sum(slika, sigma=25)
    kontrastirana = spremeni_kontrast(slika, 1.5)
    senca = add_shadow_simple(slika)

    cv.imshow('Original', slika)
    cv.imshow('Rotirana', rotirana)
    cv.imshow('Gaussov sum', z_sumom)
    cv.imshow('Kontrast', kontrastirana)
    cv.imshow('Senca', senca)

    cv.waitKey(0)
    cv.destroyAllWindows()