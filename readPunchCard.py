# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:28:16 2022

@author: Joachim
https://scipy-lectures.org/advanced/image_processing/index.html
"""

from scipy import ndimage
import numpy as np
from PIL import Image
from PIL import ImageOps
from PIL import ImageChops
from PIL import ImageStat
from PIL import ImageEnhance

if False:
    import matplotlib.pyplot as plt
from PIL import ImageDraw
import time


def run(file=None, contrast=1.0, debug=False):
    if file is None:
        file = "card2.jpg"

    #############################
    # https://github.com/digitaltrails/punchedcardreader/blob/master/punchedCardReader.py
    IBM_MODEL_029_KEYPUNCH = """
        /&-0123456789ABCDEFGHIJKLMNOPQR/STUVWXYZ:#@'="`.<(+|!$*);^~,%_>? |
    12 / O           OOOOOOOOO                        OOOOOO             |
    11|   O                   OOOOOOOOO                     OOOOOO       |
     0|    O                           OOOOOOOOO                  OOOOOO |
     1|     O        O        O        O                                 |
     2|      O        O        O        O       O     O     O     O      |
     3|       O        O        O        O       O     O     O     O     |
     4|        O        O        O        O       O     O     O     O    |
     5|         O        O        O        O       O     O     O     O   |
     6|          O        O        O        O       O     O     O     O  |
     7|           O        O        O        O       O     O     O     O |
     8|            O        O        O        O OOOOOOOOOOOOOOOOOOOOOOOO |
     9|             O        O        O        O                         | 
      |__________________________________________________________________|"""

    translate = None
    if translate is None:
        translate = {}
        # Turn the ASCII art sideways and build a hash look up for
        # column values, for example:
        #   (O, , ,O, , , , , , , , ):A
        #   (O, , , ,O, , , , , , , ):B
        #   (O, , , , ,O, , , , , , ):C
        rows = IBM_MODEL_029_KEYPUNCH[1:].split('\n');
        rotated = [[r[i] for r in rows[0:13]] for i in range(5, len(rows[0]) - 1)]
        for v in rotated:
            translate[tuple(v[1:])] = v[0]
        # print(translate)

    #############################
    # PIL/pillow operations:
    card = Image.open(file).convert("L")  # read image and convert to gray
    width, height = card.size
    stat = ImageStat.Stat(card)
    print(f"orig picture: mean={stat.mean[0]}  median={stat.median[0]}  stddev={stat.stddev[0]}  min,max={stat.extrema[0]}")
    if stat.median[0] < 80:
        return "[color=ff3333]Error: picture is too dark[/color]"
    if stat.median[0] > 230:
        return "[color=ff3333]Error: picture is too bright[/color]"
    if stat.stddev[0] < 30:
        return "[color=ff3333]Error: picture has low contrast[/color]"

    cornerBrightness = (card.getpixel((0, 0)) +
                        card.getpixel((width-1, 0)) +
                        card.getpixel((0, height-1)) +
                        card.getpixel((width-1, height-1))) / 4
    edgeBrightness = (card.getpixel((0, int(height/2))) +
                      card.getpixel((int(width/2), 0)) +
                      card.getpixel((int(width/2), height-1)) +
                      card.getpixel((width-1, int(height/2)))) / 4
    borderBrightness = int((cornerBrightness + edgeBrightness) / 2)
    print(f"mean border pixel={borderBrightness}")
    if borderBrightness > 50:
        return "[color=ff3333]Error: no black/dark background[/color]"

    if height > width:  # convert a portrait orientation to landscape
        card = card.rotate(90)
    width, height = card.size
    print(f"height={height}")

    if contrast != 1.0:
        enh = ImageEnhance.Contrast(card)
        card = enh.enhance(contrast)
    else:
        contrast = "auto"
        card = ImageOps.autocontrast(card)

    stat = ImageStat.Stat(card)
    print(f"enhanced picture, contrast = {contrast}: mean={stat.mean[0]}  median={stat.median[0]}  stddev={stat.stddev[0]}  min,max={stat.extrema[0]}")
    threshold = int(stat.mean[0])

    if False:
        card.show()

    ########################
    # scipy operations:
    # https://stackoverflow.com/questions/384759/how-to-convert-a-pil-image-into-a-numpy-array
    im2arr = np.array(card)  # im2arr.shape: height x width x channel
    # print(f"shape={im2arr.shape} dtype={im2arr.dtype} dimensions={im2arr.ndim}")

    # https://stackoverflow.com/questions/3823752/display-image-as-grayscale-using-matplotlib
    if False:
        plt.gray()  # all plt.show are gray now
        plt.imshow(im2arr)
        plt.show()

    # threshold = 80
    bw_img = im2arr > threshold  # convert to black and white
    if False:
        plt.imshow(bw_img)
        plt.show()

    # Remove small white regions
    open_img = ndimage.binary_opening(bw_img, iterations=3)
    # Remove small black holes
    close_img = ndimage.binary_closing(open_img, iterations=3)
    if False:
        plt.imshow(close_img)
        plt.show()

    # fill_img: gets the unpunched card (white)
    fill_img = ndimage.binary_fill_holes(close_img)
    my_img = Image.fromarray(fill_img)
    # my_img.show()
    # https://stackoverflow.com/questions/54134295/what-is-the-method-for-edge-detection-of-a-binary-image-in-python
    # edges are black on white
    # edges_img = ndimage.binary_dilation(~fill_img) ^ fill_img
    # plt.imshow(edges_img)
    # plt.show()

    #####################
    # get the crop coordinates of the original (possibly rotated/perspective) image
    y_nonzero, x_nonzero = np.nonzero(my_img.point(lambda i: i))
    # print(f"{x_nonzero} {y_nonzero}")
    width, height = my_img.size
    print(f"Unpunched card width, height: {width}, {height}")
    bBox = [[np.min(x_nonzero), np.min(y_nonzero)], [np.max(x_nonzero), np.max(y_nonzero)]]
    croppedWidth = np.max(x_nonzero) - np.min(x_nonzero)
    croppedHeight = np.max(y_nonzero) - np.min(y_nonzero)
    print(f"Unpunched card BBox: {bBox} width={croppedWidth} height={croppedHeight}")
    if False:
        plt.imshow(fill_img)
        plt.show()  # fill_img = unpunched card (white on black)

    # crop both images at bBox fill_img ^ close_img
    fill_imgc = fill_img[np.min(y_nonzero): np.max(y_nonzero), np.min(x_nonzero): np.max(x_nonzero)]
    if False:
        plt.imshow(fill_imgc)
        plt.show()  # fill_imgc = cropped unpunched card (white on black)
    close_imgc = close_img[np.min(y_nonzero): np.max(y_nonzero), np.min(x_nonzero): np.max(x_nonzero)]
    # only the white holes are left in holes_img
    holes_imgc = fill_imgc ^ close_imgc
    if False:
        plt.imshow(holes_imgc)
        plt.show()

    my_imgc = Image.fromarray(fill_imgc)
    y_nonzero, x_nonzero = np.nonzero(my_imgc.point(lambda i: i))
    bBox = [[np.min(x_nonzero), np.min(y_nonzero)], [np.max(x_nonzero), np.max(y_nonzero)]]
    croppedWidth = np.max(x_nonzero) - np.min(x_nonzero)
    croppedHeight = np.max(y_nonzero) - np.min(y_nonzero)
    print(f"BBox: {bBox} width={croppedWidth} height={croppedHeight}")

    ####################
    # try to rectify the image by QUAD translation: find the card borders
    # horizontal scan at y=25% and y=75% to avoid bevel and black print on left side
    xLow = [-1, -1]
    xHigh = [-1, -1]
    horizScan = [int(0.25 * croppedHeight), int(0.75 * croppedHeight)]
    for idx, y in enumerate(horizScan):
        for x in range(croppedWidth):
            if fill_imgc[y, x] > 0:
                # print(f"xLow: {x}")
                xLow[idx] = x
                break
        for x in reversed(range(croppedWidth)):
            if fill_imgc[y, x] > 0:
                # print(f"xHigh: {x}")
                xHigh[idx] = x
                break
    delta = (xLow[1] - xLow[0] + xHigh[1] - xHigh[0]) / 0.5 / 2

    # vertical scan at x=10% and x=90% to find tilt of card
    yLow = [-1, -1]
    yHigh = [-1, -1]
    vertScan = [int(0.1 * croppedWidth), int(0.9 * croppedWidth)]
    for idx, x in enumerate(vertScan):
        for y in range(croppedHeight):
            if fill_imgc[y, x] > 0:
                # print(f"yLow: {y}")
                yLow[idx] = y
                break
        for y in reversed(range(croppedHeight)):
            if fill_imgc[y, x] > 0:
                # print(f"yHigh: {y}")
                yHigh[idx] = y
                break
    delta = (yLow[1] - yLow[0] + yHigh[1] - yHigh[0]) / 0.8 / 2

    ####################
    # find the corners
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    # print get_intersect((0, 1), (1, 2), (0, 10), (1, 9))  # another line for fun
    def get_intersect(a1, a2, b1, b2):
        """
        Returns the point of intersection of the lines iterationing through a2,a1 and b2,b1.
        a1: [x, y] a point on the first line
        a2: [x, y] another point on the first line
        b1: [x, y] a point on the second line
        b2: [x, y] another point on the second line
        """
        s = np.vstack([a1, a2, b1, b2])  # s for stacked
        h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
        l1 = np.cross(h[0], h[1])  # get first line
        l2 = np.cross(h[2], h[3])  # get second line
        x, y, z = np.cross(l1, l2)  # point of intersection
        if z == 0:  # lines are parallel
            return int('inf'), int('inf')
        return int(x / z), int(y / z)

    def flatten(lst):
        return [item for sublist in lst for item in sublist]

    # coordinates are nw = left upper corner (low x and low y)
    nw = get_intersect((vertScan[0], yLow[0]),
                       (vertScan[1], yLow[1]),
                       (xLow[0], horizScan[0]),
                       (xLow[1], horizScan[1])
                       )
    sw = get_intersect((vertScan[0], yHigh[0]),
                       (vertScan[1], yHigh[1]),
                       (xLow[0], horizScan[0]),
                       (xLow[1], horizScan[1])
                       )
    ne = get_intersect((vertScan[0], yLow[0]),
                       (vertScan[1], yLow[1]),
                       (xHigh[0], horizScan[0]),
                       (xHigh[1], horizScan[1])
                       )
    se = get_intersect((vertScan[0], yHigh[0]),
                       (vertScan[1], yHigh[1]),
                       (xHigh[0], horizScan[0]),
                       (xHigh[1], horizScan[1])
                       )
    print(f"nw:{nw}  sw:{sw}  ne:{ne}  se:{se}")

    #############################
    # PIL/pillow operations:

    # transform perspective to rectangle
    # https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/Image__class_Image.html
    '''
    https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.transform
    https://pillow.readthedocs.io/en/stable/_modules/PIL/Image.html#Transform            
                ne = data[6:8]            # given as NW, SW, SE, and NE.
                nw = data[:2]
                sw = data[2:4]
                se = data[4:6]
    '''

    holesimc = Image.fromarray(holes_imgc)
    draw = ImageDraw.Draw(holesimc)
    draw.line([nw, ne, se, sw, nw], fill=256, width=5)
    if False:
        holesimc.show()

    # QUAD transform should turn perspective to rectangle
    size = (int((ne[0] - nw[0] + se[0] - sw[0]) / 2), int((sw[1] - nw[1] + se[1] - ne[1]) / 2))  # mean
    # size = ( max(ne[0]-nw[0], se[0]-sw[0]), max(sw[1]-nw[1], se[1]-ne[1]) )            # max
    # size = ( int((math.dist(ne,nw) + math.dist(se,sw))/2), int((math.dist(nw,sw) + math.dist(ne,se))/2) )  # dist
    # size = (max(ne[0],se[0]) - min(nw[0],sw[0]), max(se[1],sw[1]) - min(ne[1],nw[1]))  # bbox
    print(f"size: {size}")
    imgholes = holesimc.transform(size, Image.QUAD,  # Image.Transform.QUAD  9.2 vs 8.4
                                  flatten([nw, sw, se, ne]),
                                  resample=Image.BILINEAR)  # Image.Resampling.BILINEAR
    if False:
        imgholes.show()

    ###################################################################
    # also transform the blank card image to find out position of bevel
    imgcard = my_imgc.transform(size, Image.QUAD,  # Image.Transform.QUAD
                                flatten([nw, sw, se, ne]),
                                resample=Image.BILINEAR)  # Image.Resampling.BILINEAR
    if False:
        imgcard.show()
    bevelx = 0.04
    bevely = 0.13
    bbox_ul = (0, 0, int(imgcard.size[0] * bevelx), int(imgcard.size[1] * bevely))
    bbox_ur = (int(imgcard.size[0] * (1 - bevelx)), 0, imgcard.size[0], int(imgcard.size[1] * bevely))
    bbox_ll = (0, int(imgcard.size[1] * (1 - bevely)), int(imgcard.size[0] * bevelx), imgcard.size[1])
    bbox_lr = (
    int(imgcard.size[0] * (1 - bevelx)), int(imgcard.size[1] * (1 - bevely)), imgcard.size[0], imgcard.size[1])

    imgcard_ul = imgcard.crop(bbox_ul)
    stat_ul = ImageStat.Stat(imgcard_ul).mean
    # print(stat)
    # imgcard_ul.show()
    imgcard_ur = imgcard.crop(bbox_ur)
    stat_ur = ImageStat.Stat(imgcard_ur).mean
    # print(stat)
    # imgcard_ur.show()
    imgcard_ll = imgcard.crop(bbox_ll)
    stat_ll = ImageStat.Stat(imgcard_ll).mean
    # print(stat)
    # imgcard_ll.show()
    imgcard_lr = imgcard.crop(bbox_lr)
    stat_lr = ImageStat.Stat(imgcard_lr).mean
    # print(stat)
    # imgcard_lr.show()

    print(f"Image Mode={imgholes.mode}")  # image Mode is 1, expected L or RGB

    stat = [("ul", stat_ul), ("ur", stat_ur), ("ll", stat_ll), ("lr", stat_lr)]
    corner = min(stat, key=lambda t: t[1])
    print(corner)
    bevelCorner = corner[0]
    print(f"bevelCorner:{bevelCorner}")
    if bevelCorner == "ul":
        # frontside, correct orientation nothing to do
        pass
    elif bevelCorner == "ur":
        # backside, flip horizontally
        imgholes = ImageOps.mirror(imgholes)
    elif bevelCorner == "ll":
        # backside, flip vertically
        imgholes = ImageOps.flip(imgholes)
    elif bevelCorner == "lr":
        # frontside, rotate 180°
        imgholes = imgholes.rotate(180)
    else:
        pass

    def invertImage(image):  # only in PIL 9.2.0 we can call ImageOps.invert
        lut = []
        for i in range(256):
            lut.append(255 - i)
        return image.point(lut)

    print(f"Image Mode={imgholes.mode}")
    if imgholes.mode == "1":
        imgci = invertImage(imgholes)  # invert, now card is white, holes black
    else:
        imgci = ImageOps.invert(imgholes)  # invert, now card is white, holes black
    # imgci.show()

    width, height = imgci.size
    idealRatio = 187.325 / 82.55
    print(f"cropped size: {width} {height}  ratio: {width / height * 100:.1f}%  (ideal: {idealRatio * 100:.1f})")

    # Find all holes
    # https://scipy-lectures.org/advanced/image_processing/auto_examples/plot_find_object.html#sphx-glr-advanced-image-processing-auto-examples-plot-find-object-py
    imgci_np = np.array(imgci)
    mask = ~imgci_np
    # print(f"imgci={imgci}")
    # print(f"imgci={np.array(imgci)}")
    # print(f"mask={mask}")
    label_im, nb_labels = ndimage.label(mask)
    # print(label_im, nb_labels)
    print(f"{nb_labels} holes detected")

    labels = np.unique(label_im)
    label_im = np.searchsorted(labels, label_im)
    # plt.imshow(label_im)
    # plt.show()
    slices = ndimage.find_objects(label_im)
    # print(result)
    printSlices = False
    if printSlices:
        for myslice in slices:
            # print(slice)
            slice_y, slice_x = myslice
            ul = slice_x.start, slice_y.start
            lr = slice_x.stop - 1, slice_y.stop - 1
            bbox = [ul, lr]
            print(bbox)

    ############################################################
    # "intersect of union": calculate "percentage" of overlap
    # https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
    def iou(boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0][0], boxB[0][0])
        yA = max(boxA[0][1], boxB[0][1])
        xB = min(boxA[1][0], boxB[1][0])
        yB = min(boxA[1][1], boxB[1][1])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[1][0] - boxA[0][0] + 1) * (boxA[1][1] - boxA[0][1] + 1)
        boxBArea = (boxB[1][0] - boxB[0][0] + 1) * (boxB[1][1] - boxB[0][1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        # print(f"\n{boxA} vs {boxB}")
        # print(f"{boxA[0][0]} {boxA[0][1]}  {boxA[1][0]} {boxA[1][1]}")
        # print(f"{boxB[0][0]} {boxB[0][1]}  {boxB[1][0]} {boxB[1][1]}")
        # print(f"({xA}, {yA}) ({xB}, {yB})")
        # print(f"inter:{interArea}  boxA:{boxAArea}  boxB:{boxBArea}")
        IoU = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return IoU

    ############################################################
    # return true if bounding boxes overlap
    def intersect(bbox1, bbox2):  # (xl, yl), (xh, yh)
        return False if bbox1[0][0] > bbox2[1][0] \
                        or bbox2[0][0] > bbox1[1][0] \
                        or bbox1[0][1] > bbox2[1][1] \
                        or bbox2[0][1] > bbox1[1][1] else True

    ############################################################
    # return xCoord if a hole matches the predefined pattern, else None
    def holeMatch(bbox, slices):
        for slice in slices:
            slice_y, slice_x = slice
            ul = slice_x.start, slice_y.start
            lr = slice_x.stop - 1, slice_y.stop - 1
            hole = [ul, lr]
            if intersect(bbox, hole):
                IoU = int(iou(bbox, hole) * 100)
                # print(f"Intersection: {bbox} vs {hole} {IoU:.1f}")
                holeArea = (hole[1][0] - hole[0][0]) * (hole[1][1] - hole[0][1])
                midX = (hole[1][0] + hole[0][0]) / 2
                holePerc = int(holeArea / chadArea * 100)
                # print(f"holeArea:{holeArea}  chadArea:{chadArea}  holePerc:{holePerc}%  IoU:{IoU}%")
                if holePerc > 20 and IoU > 10:
                    return midX
                # else:
                # print(f"Intersection too small: {bbox} vs {hole} {IoU}%")
        return None

    ######################################################
    # create punch card image as reference/mask
    # and find overlapping holes
    # dimensions: http://www.quadibloc.com/comp/cardint.htm
    # card size = 187,325 mm × 82,55 mm  (7 3/8" wide and 3 1/4" high)
    # Reihenraster  1/4 Zoll = 6,35  mm                        0,07692
    # Lochhöhe      1/8 Zoll = 3,175 mm                        0,03846
    #
    # Spaltenraster 0,087 Zoll = 2,2098 mm                     0,01180
    # Lochbreite    0,055 Zoll = 1,397  mm                     0,007458
    #
    # Rand oben und unten jeweils     3/16 Zoll = 4,7625 mm    0,05770
    # Rand links und rechts jeweils 0,2235 Zoll = 5,6769 mm    0,03030

    holes = np.zeros((80, 12))

    horizBorder = width * 0.03030
    vertBorder = height * 0.05770
    crd = Image.new("L", (width, height))
    drw = ImageDraw.Draw(crd)
    drw.rectangle([0, 0, width, height], fill="white")
    chadWidth = width * 0.007458
    chadHeight = height * 0.03846
    chadArea = int(chadWidth * chadHeight)
    columnPeriod = width * 0.01180

    holeMatches = 0
    xStart = horizBorder + int(chadWidth / 2)  # start in the middle of the chad
    xl = xStart
    for x in range(1, 81):  # columns
        # xl = horizBorder + (x-1) * columnPeriod
        passes = ['align', 'read']  # 2 passes: calc mean xDeviation and then read
        xDevMean = 0
        for iteration in passes:
            if iteration == 'read':  # first iteration: calc Deviation
                xl = int(xl + xDevMean)
            xh = int(xl + chadWidth)
            xl = int(xl)
            xMidChad = (xh + xl) / 2
            numHoles = 0
            for y in range(1, 13):  # rows
                yl = vertBorder + (y - 1) * height * 0.07692
                yh = int(yl + chadHeight)
                yl = int(yl)
                drw.rectangle([xl, yl, xh, yh], fill="black")
                bbox = [(xl, yl), (xh, yh)]
                xMid = holeMatch(bbox, slices)
                if xMid is not None:
                    xDev = xMid - xMidChad
                    if iteration == 'read':
                        holeMatches += 1
                        holes[x - 1, y - 1] = 1
                    else:
                        numHoles += 1
                        xDevMean += xDev
                    if (debug):
                        print(f"{iteration:5s} : {x} {y} {bbox}  xDeviation: {xMid} - {xMidChad} = {xDev}")
            if numHoles > 0 and iteration == 'align':
                xDevMean = xDevMean / numHoles
        xl += columnPeriod  # next column
        xlIdeal = xStart + x * columnPeriod
        deviation = xl - xlIdeal
        if abs(deviation) > columnPeriod:
            print(f"WARNING: column {x} deviation {deviation:.1f} > columnPeriod {columnPeriod:.1f}")

    missingColumns = int(deviation / columnPeriod)
    print(f"{holeMatches} matched holes")
    print(f"WARNING: {missingColumns} missing columns")

    #########################################################
    # translate holes to text
    text = ""
    for x in range(1, 81):  # column
        col_pattern = []
        rows = np.zeros(12)
        for y in range(1, 13):  # rows
            if holes[x - 1, y - 1] > 0:
                col_pattern.append('O')
            else:
                col_pattern.append(' ')
        column_hole_pattern = tuple(col_pattern)
        if column_hole_pattern in translate:
            # A translation exists - append it to the text result
            text += translate[column_hole_pattern]
        else:
            text += "\u00BF"            # unknown (upside down question mark ?)

    print("")
    print("         1         2         3         4         5         6         7         8")
    print("1........0.........0.........0.........0.........0.........0.........0.........0")
    print(f"{text}\n")
    #      0242313KEINERT,JOACHIM   PRAKTIKUM IMPULS-UND DIGITALTECHNIK     7884041705 0 13

    #######################################################
    # show the difference
    # ImageOps.invert(ImageChops.difference(imgci, crd)).show()
    imgdiff = ImageChops.difference(imgci, crd)

    if file is None:
        fn = "/sdcard/CARD_{}_READ.png".format(time.strftime("%Y%m%d_%H%M%S"))
    else:
        fn = file.replace(".png", "_DIFF.png")
    imgdiff.save(fn)

    return (f"[color=0080ff]Text: '[/color]{text}[color=0080ff]'[/color]", text)


if __name__ == "__main__":
    file = "card2.jpg"
    file = "CARD_20220922_094644.png"
    file = "CARD_20220922_102704.png"
    debug = True
    res = run(file=file, contrast=1.0, debug=debug)
    print(res)
