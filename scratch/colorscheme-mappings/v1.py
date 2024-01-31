from itertools import product
import json

from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


# img = Image.open("/home/isaac/Pictures/wallpaper/mountain_lake.jpg")
# img.show()
# I,J,K = np.shape(img)
# pix = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)

# colors = np.reshape(pix, (I * J, K))
# unique_colors = np.unique(colors, axis=0)
# print(np.mean(unique_colors, axis=0))


def get_three_closest(unseen: np.ndarray, colorscheme: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.reshape(np.concatenate(tuple(map(np.transpose, sorted(colorscheme, key=lambda x: np.sum(np.abs(unseen - x)))[:3]))), (3, 3)).T


# newcolor = np.array([100, 100, 100])

# three_closest = get_three_closest(newcolor, unique_colors)
# print(three_closest)


def transform(color, newbase):
    inverse = np.linalg.inv(newbase)
    newcolor_vec = np.reshape(color, (3, 1))
    new_representation = inverse.dot(newcolor_vec)
    return new_representation
    

# inverse = np.linalg.inv(three_closest)
# newcolor_vec = np.reshape(newcolor, (3, 1))
# new_representation = inverse.dot(newcolor_vec)

# approx = three_closest.dot(new_representation)


def cluster_top_k(img_array: np.ndarray, k=16) -> list[np.ndarray]:
    km = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(img_array)
    labels = km.labels_
    print(len(labels))
    cluster_dict = {i: [] for i in range(k)}
    for cluster, pixel in zip(labels, img_array):
        cluster_dict[cluster].append(pixel)
    return {k: np.array(v) for k, v in cluster_dict.items()}


# cluster_dict = cluster_top_k(colors)
# cluster_means = {k: np.mean(v, axis=0) for k, v in cluster_dict.items()}
# cluster_medians = {k: np.median(v, axis=0) for k, v in cluster_dict.items()}
# cluster_sizes = {k: len(v) for k, v in cluster_dict.items()}


def newbase_from_equivalence(base, equivalence):
    newmatrix = np.zeros((3, 3))
    for i, vec in enumerate(base.T):
        tup = tuple(vec)
        newvec = equivalence[tup]
        newmatrix[:,i] = newvec
    return newmatrix


def remap_color(color, origin_colors, equivalence):
    base = get_three_closest(color, origin_colors)
    # print(base, np.mean(base))
    newrep = transform(color, base)
    newrep = newrep / np.sum(np.abs(newrep))
    # print(newrep)
    # print(base.dot(newrep))
    newbase = newbase_from_equivalence(base, equivalence)
    # print(newbase)
    newcolor = newbase.dot(newrep)
    # print(newcolor)
    proportions = np.array([np.linalg.norm(color - vec) for vec in newbase.T])
    # print(proportions)
    proportions = proportions / np.sum(proportions)
    # print(proportions)
    targetlength = newbase.dot(proportions)
    # print(targetlength)
    targetlength = np.linalg.norm(targetlength)
    # print(targetlength)
    newarray = newcolor * targetlength / np.linalg.norm(newcolor)
    newarray = np.array(np.reshape(np.minimum(np.maximum(newcolor, 0), 255), (3,)))
    # print(newarray)
    ret = newarray.astype("uint8")
    # print(ret)
    return ret

# remap_color(np.array([100, 100, 100]), orig_colors, equiv)

# for i in range(16):
#     print(len(cluster_dict[i]))
#     print()


def parse_color_codes(codes_list):
    print(codes_list)
    def to_int(hexstr):
        return int(hexstr, base=16)
    def parse_code(code):
        print(code)
        rgb = np.array(list(map(to_int, [code[:2], code[2:4], code[4:6]])))
        print(rgb)
        return rgb
    return tuple(map(parse_code, codes_list))


def compute_equivalence(orig_colors, new_colors):
    new_colors = [list(x) for x in new_colors]
    equivalence = {tuple(orig_colors[0]): new_colors[0]}
    for color in orig_colors:
        best_match = min(new_colors, key=lambda x: np.linalg.norm(np.array(x) - color))
        equivalence.update({tuple(color): best_match})
        new_colors.remove(best_match)
    print(equivalence)
    return equivalence
#-----------------------------------------------------------------------------------------------------------------------

def repaint_image(img_path: Path, colorscheme: list[tuple[int, int, int]], k=7) -> Image.Image:
    img = Image.open(img_path)
    I,J,K = np.shape(img)
    print(I, J, K)
    pixellist = np.array(img.getdata())
    # pixellist = imgarray.reshape((img.size[0] * img.size[1], 3))
    print(pixellist.shape)
    unique_colors = np.unique(pixellist, axis=0)
    cluster_dict = cluster_top_k(pixellist, k=k)
    cluster_means = {k: np.mean(v, axis=0) for k, v in cluster_dict.items()}
    for kv in cluster_means.items():
        print(kv)
    # cluster_medians = {k: np.median(v, axis=0) for k, v in cluster_dict.items()}
    cluster_sizes = {k: len(v) for k, v in cluster_dict.items()}
    for kv in cluster_sizes.items():
        print(kv)
    orig_colors = [cluster_means[i] for i in sorted(cluster_sizes, key=lambda x: cluster_sizes[x], reverse=True)]
    print(orig_colors)
    equiv = compute_equivalence(orig_colors, colorscheme) #{tuple(orig): np.array(newcol) for orig, newcol in zip(orig_colors, colorscheme)}
    print(equiv)
    mapping = {tuple(p): remap_color(p, orig_colors, equiv) for p in unique_colors}
    newimg = np.zeros_like(pixellist)
    for i, rgb in enumerate(pixellist): #[100000:100100], start=100000):
        # print(mapping[tuple(rgb)])
        newimg[i] = mapping[tuple(rgb)]
    reshaped = np.reshape(newimg, np.shape(img))
    reshaped = np.array(reshaped, dtype="uint8")
    return Image.fromarray(reshaped)


#-----------------------------------------------------------------------------------------------------------------------
apathy = parse_color_codes([
    "031A16",
    "0B342D",
    "184E45",
    "2B685E",
    "5F9C92",
    "81B5AC",
    "A7CEC8",
    "D2E7E4",
    "3E9688",
    "3E7996",
    "3E4C96",
    "883E96",
    "963E4C",
    "96883E",
    "4C963E",
    "3E965B",
])

dracula = parse_color_codes([
    "282936", #background
    "3a3c4e",
    "4d4f68",
    "626483",
    "62d6e8",
    "e9e9f4", #foreground
    "f1f2f8",
    "f7f7fb",
    "ea51b2",
    "b45bcf",
    "00f769",
    "ebff87",
    "a1efe4",
    "62d6e8",
    "b45bcf",
    "00f769",
])

nature_img_path = Path("/home/isaac/Pictures/wallpaper/mountain_lake.jpg")
leaves_path = Path("/home/isaac/Pictures/wallpaper/leaves.jpg")
newpath =  Path("/home/isaac/Pictures/wallpaper/mountain_lake_dracula2.jpg")
apath = Path("/home/isaac/Pictures/wallpaper/mountain_lake_apathy.jpg")
newleavespath = Path("/home/isaac/Pictures/wallpaper/leaves_apathy.jpg")
      
# new_img = repaint_image(nature_img_path, dracula)
# new_img.save(newpath)

# new_img = repaint_image(nature_img_path, apathy)
# new_img.save(apath)

new_img = repaint_image(leaves_path, apathy, k=3)
new_img.save(newleavespath)


#---------------
# https://stackoverflow.com/questions/26571199/vectorize-multiplying-rgb-array-by-color-transform-matrix-for-image-processing

X = np.arange(120).reshape(5,8,3)
W = np.array(
    [
        [1. , 0. , 0.5],
        [0. , 1. , 0. ],
        [0.5, 0.5, 1. ]
    ]
)

transformed = np.tensordot(X, W, axes=([2], [1]))
print(transformed.shape)


# another possibility: https://stackoverflow.com/questions/58823918/how-to-parallelize-model-prediction-from-a-pytorch-model