from functions import *

input_dir = "input"
output_dir = "output"


def tie():
    output = os.path.join(output_dir, "tie")
    if not os.path.exists(output):
        os.makedirs(output)
    near = cv2.imread(os.path.join(input_dir, "near.jpg"))
    middle = cv2.imread(os.path.join(input_dir, "middle.jpg"))
    far = cv2.imread(os.path.join(input_dir, "far.jpg"))
    images = [near, middle, far]
    new_image, mags, blurs, masks = allInFocus(images)
    writeWithIndex(mags, "magnitude", output)
    writeWithIndex(blurs, "blur", output)
    writeWithIndex(masks, "mask", output)
    tie = os.path.join(output, "tie.jpg")
    cv2.imwrite(tie, new_image)

def boardgame():
    output = os.path.join(output_dir, "boardgame")
    if not os.path.exists(output):
        os.makedirs(output)
    one = cv2.imread(os.path.join(input_dir, "boardgame_1.jpg"))
    two = cv2.imread(os.path.join(input_dir, "boardgame_2.jpg"))
    three = cv2.imread(os.path.join(input_dir, "boardgame_3.jpg"))
    four = cv2.imread(os.path.join(input_dir, "boardgame_4.jpg"))
    images = [one, two, three, four]
    new_image, mags, blurs, masks = allInFocus(images)
    writeWithIndex(mags, "magnitude", output)
    writeWithIndex(blurs, "blur", output)
    writeWithIndex(masks, "mask", output)
    boardgame = os.path.join(output, "boardgame.jpg")
    cv2.imwrite(boardgame, new_image)


def flowers():
    output = os.path.join(output_dir, "flowers")
    if not os.path.exists(output):
        os.makedirs(output)
    far = cv2.imread(os.path.join(input_dir, "flowers_far.jpg"))
    near = cv2.imread(os.path.join(input_dir, "flowers_near.jpg"))
    images = [far, near]
    new_image, mags, blurs, masks = allInFocus(images, lambda x: boxBlur(x, 70))
    writeWithIndex(mags, "magnitude", output)
    writeWithIndex(blurs, "blur", output)
    writeWithIndex(masks, "mask", output)
    flowers = os.path.join(output, "flowers.jpg")
    cv2.imwrite(flowers, new_image)


def flowers_aligned():
    output = os.path.join(output_dir, "flowers_aligned")
    if not os.path.exists(output):
        os.makedirs(output)
    far = cv2.imread(os.path.join(input_dir, "flowers_far.jpg"))
    near = cv2.imread(os.path.join(input_dir, "flowers_near.jpg"))
    far_warped, img1_kp, img2_kp, matches = align(far, near, 6)
    warp = os.path.join(output, "warp.jpg")
    matchez = os.path.join(output, "matches.jpg")
    cv2.imwrite(matchez, drawMatches(far, img1_kp, near, img2_kp, matches))
    cv2.imwrite(warp, far_warped)
    images = [far_warped, near]
    new_image, mags, blurs, masks = allInFocus(images, lambda x: boxBlur(x, 70))
    writeWithIndex(mags, "magnitude", output)
    writeWithIndex(blurs, "blur", output)
    writeWithIndex(masks, "mask", output)
    flowers = os.path.join(output, "flowers.jpg")
    cv2.imwrite(flowers, new_image)


def kitchen():
    output = os.path.join(output_dir, "kitchen")
    if not os.path.exists(output):
        os.makedirs(output)
    far = cv2.imread(os.path.join(input_dir, "kitchen_far.jpg"))
    near = cv2.imread(os.path.join(input_dir, "kitchen_near.jpg"))
    middle = cv2.imread(os.path.join(input_dir, "kitchen_middle.jpg"))
    images = [far, near, middle]
    new_image, mags, blurs, masks = allInFocus(images)
    writeWithIndex(mags, "magnitude", output)
    writeWithIndex(blurs, "blur", output)
    writeWithIndex(masks, "mask", output)
    kitchen = os.path.join(output, "kitchen.jpg")
    cv2.imwrite(kitchen, new_image)


def kitchen_aligned():
    output = os.path.join(output_dir, "kitchen_aligned")
    if not os.path.exists(output):
        os.makedirs(output)
    far = cv2.imread(os.path.join(input_dir, "kitchen_far.jpg"))
    near = cv2.imread(os.path.join(input_dir, "kitchen_near.jpg"))
    middle = cv2.imread(os.path.join(input_dir, "kitchen_middle.jpg"))
    aImages = align(far, middle, 10)
    matchez = os.path.join(output, "far_matches.jpg")
    warpFar, img1_kp, img2_kp, matches = aImages
    cv2.imwrite(matchez, drawMatches(far, img1_kp, middle, img2_kp, matches))

    warpNear, img1_kp, img2_kp, matches = align(near, middle, 4)
    matchez = os.path.join(output, "near_matches.jpg")
    cv2.imwrite(matchez, drawMatches(near, img1_kp, middle, img2_kp, matches))


    images = [warpNear, middle, warpFar]
    new_image, mags, blurs, masks = allInFocus(images)
    writeWithIndex(mags, "magnitude", output)
    writeWithIndex(blurs, "blur", output)
    writeWithIndex(masks, "mask", output)
    writeWithIndex(images, "warp", output)
    kitchen = os.path.join(output, "kitchen.jpg")
    cv2.imwrite(kitchen, new_image)






if __name__ == "__main__":
    tie()
    boardgame()
    flowers()
    flowers_aligned()
    kitchen()
    kitchen_aligned()

