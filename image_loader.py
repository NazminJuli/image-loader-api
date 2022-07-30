import base64

import aiohttp
import asyncio
import uvicorn

# from starlette.applications import Starlette
# from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
# from starlette.routing import Route


import os
import sys
import cv2
import numpy as np
import fillfront
import priorities
import bestpatch
import update
from skimage import io

pixel_extension = 18
working_image = None
working_mask = None
nx = None
ny = None
nw = None
nh = None
#
# from fastai.vision.all import *
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from typing import List

app = FastAPI()


def generate_mask_patch(original_image):

    img_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    #detect red color
    lower_red = np.array([0, 240, 240])  # modified red color range for 3 pixels from marking using Adobe's path selection
    upper_red = np.array([180, 255, 255])

    # # # detect green color
    # lower_red = np.array([40, 240, 240])  # modified red color range for 3 pixels from marking using Adobe's path selection
    # upper_red = np.array([70, 255, 255])
    # Merge the mask and crop the red regions
    red_mask = cv2.inRange(img_hsv, lower_red, upper_red)
    # red = cv2.bitwise_and(original_image, original_image, mask=red_mask)
    count_c, _ = cv2.findContours(red_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_mask = np.zeros((original_image.shape[0], original_image.shape[1]), dtype='uint8')
    print("mask contour len initial: ", len(count_c))

    for c in count_c:
        cv2.drawContours(img_mask, [c], -1, (255, 255, 255), thickness = cv2.FILLED)

    invert = cv2.bitwise_not(img_mask)
    mask = invert

    kernel = np.ones((3, 3), 'uint8')
    mask = cv2.erode(mask, kernel, iterations = 3)
    img_mask = cv2.dilate(img_mask, kernel, iterations=3)
    # cv2.imwrite('invert_mask.jpg', mask)
    # cv2.imwrite('original_mask.jpg', img_mask)
    return mask, img_mask


def source_region(i, contour,working_image,working_mask):
    image_copy = working_image
    # print(image_copy.shape, 'working.....')
    mask_copy = working_mask
    (nx, ny, nw, nh) = cv2.boundingRect(contour)
    print("show box:",nx, ny, nw, nh)
    new_source_patch = image_copy[ny - pixel_extension:ny + nh + pixel_extension,
                       nx - pixel_extension:nx + nw + pixel_extension]

    new_mask_patch = mask_copy[ny - pixel_extension:ny + nh + pixel_extension,
                     nx - pixel_extension:nx + nw + pixel_extension]
    new_mask_patch = cv2.merge([new_mask_patch, new_mask_patch, new_mask_patch])  # 3 mode generation of mask
    # on basis of mask, generate original patch from bitwise_OR
    masked_and = cv2.bitwise_or(new_source_patch, new_mask_patch)
    # new_source_patch = masked_and
    new_img = 'new_search_' + str(i) + '.jpg'
    new_mask = 'new_mask_' + str(i) + '.jpg'
    cv2.imwrite(new_img, new_source_patch)
    return new_source_patch, new_mask_patch, nx, ny, nw, nh


def image_process(masque,working_image,working_mask):

    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("length of total contours....", len(contours))
    i = 0
    lk = 0

    for contour in contours:
        new_source_patch, new_mask_patch, nx, ny, nw, nh = source_region(i, contour,working_image,working_mask)
        i = i + 1
        tau = 170
        omega = []

        nxsize, nysize, channels = new_source_patch.shape
        if channels == 4:
            print("its a RGBA image")
            taillecadre = 4
        else:
            taillecadre = 3

        new_mask_patch = cv2.cvtColor(new_mask_patch, cv2.COLOR_RGB2GRAY)

        confiance = np.copy(new_mask_patch)
        masque = np.copy(new_mask_patch)

        for x in range(nxsize):
            for y in range(nysize):
                v = masque[x, y]
                if v < tau:
                    omega.append([x, y])
                    new_source_patch[x, y] = [255, 255, 255]
                    masque[x, y] = 1
                    confiance[x, y] = 0.
                else:
                    masque[x, y] = 0
                    confiance[x, y] = 1.


        source = np.copy(confiance)
        original = np.copy(confiance)
        dOmega = []
        normale = []

        im = np.copy(new_source_patch)

        result = np.ndarray(shape=new_source_patch.shape)

        data = np.ndarray(shape=new_source_patch.shape[:2])

        bool = True  # pour le while
        # print("Algorithme en fonctionnement")
        k = 0

        while bool:
            # print(k)
            k += 1
            xsize, ysize = source.shape

            niveau_de_gris = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

            gradientX = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 1, 0)))

            gradientY = np.float32(cv2.convertScaleAbs(cv2.Scharr(niveau_de_gris, cv2.CV_32F, 0, 1)))

            for x in range(xsize):
                for y in range(ysize):
                    if masque[x][y] == 1:
                        gradientX[x][y] = 0
                        gradientY[x][y] = 0
            gradienX, gradientY = gradientX / 255, gradientY / 255

            dOmega, normale = fillfront.IdentifyTheFillFront(masque, source)

            # print("shapes:", im.shape, masque.shape, confiance.shape, original.shape, dOmega)
            confiance, data, index = priorities.calculPriority(im, taillecadre, masque, dOmega, normale, data,
                                                               gradientX, gradientY, confiance)
            # print("index:", index)

            list, pp = bestpatch.calculPatch(dOmega, index, im, original, masque, taillecadre)

            im, gradientX, gradientY, confiance, source, masque = update.update(im, gradientX, gradientY, confiance,
                                                                                source, masque, dOmega, pp, list, index,
                                                                                taillecadre)

            # on verifie si on a fini
            bool = False
            for x in range(xsize):
                for y in range(ysize):
                    if source[x, y] == 0:
                        bool = True

            #     # on enregistre a chaque fois pour voir l'avancée
            # print("indexes:", ny,nh,nx,nw)
            working_image[ny - pixel_extension: ny + pixel_extension + nh,
            nx - pixel_extension: nx + nw + pixel_extension] = im

            name = 'F:\\PyCharm Codes\\pythonProject\\retouch\\api\\test\\' + str(lk) + "_result.jpg"
            cv2.imwrite(name, im)
            lk = lk + 1
            # cv2.imwrite("F:\\PyCharm Codes\\pythonProject\\retouch\\Inpainting-master\\Inpainting-master\\tests\\out.jpg", im)

    cv2.imwrite("_resultat_new.png", working_image)
    return working_image

@app.post("/upload/")
async def upload(files: List[UploadFile] = File(...)):

      for file in files:
        # data = await request.form()
        # filename = data['file'].filename
        bytes = await file.read()
        # n = len(data)
        nparr = np.fromstring(bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_dimensions = str(image.shape)

        # generate mask from the marking area
        invert_masque, masque = generate_mask_patch(image)
        xsize, ysize, channels = image.shape
        #
        x, y = invert_masque.shape
        #
        # if x != xsize or y != ysize:
        #     print("error!!!!")
        #     exit()
        #
        working_image = image
        working_mask = invert_masque

        value, thresh = cv2.threshold(masque, 60, 255, cv2.THRESH_BINARY_INV)

        processed_image = image_process(masque,working_image,working_mask)
        # # line that fixed it
        _, encoded_img = cv2.imencode('.PNG', processed_image)
        #
        encoded_img = base64.b64encode(encoded_img)
        return {'DIMENSIONS': encoded_img}
      return {'done!!!'}



@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload/" method = "post" enctype = "multipart/form-data">
            <u> Select picture to upload: </u> <br> <p>
            1. <input type="file" name="files" multiple><br><p>
            2. <input type="submit" value="Upload">
        </form>
        """)


if __name__ == "__main__":

        uvicorn.run(app, host = "0.0.0.0", port = 8000)