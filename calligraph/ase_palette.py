# Adapted from GIMP plugin
# Author: Chris Mohler, Roy Curtis
# Copyright 2008 Chris Mohler
# License: GPL v3
# Portions of this code were taken from easyrgb.com
# GIMP plugin to convert Adobe Swatch Exchange (ASE) palettes to GIMP (GPL) palettes
# Updated for GIMP 2.8.x and later ASE format by Roy Curtis

# ASE file format references:
# * http://carl.camera/default.aspx?id=109
# * http://www.selapa.net/swatches/colors/fileformats.php#adobe_ase
# * https://bazaar.launchpad.net/~olivier-berten/swatchbooker/trunk/view/head:/src/swatchbook/codecs/adobe_ase.py
#
# ... and their archives, if they become lost to history:
#
# * http://archive.is/C2Fe6
# * http://archive.is/jFiTU
# * http://archive.is/AEB9m

from struct import unpack_from, unpack
import os, traceback
from io import BytesIO
import numpy as np

def load_act_palette(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()

    # Remove the extra 2 bytes if the file is 770 bytes long
    if len(data) == 770:
        data = data[:-2]

    # ACT palettes are fixed at 256 colors max
    num_colors = len(data) // 3

    colors = []
    for i in range(num_colors):
        r = data[i * 3]
        g = data[i * 3 + 1]
        b = data[i * 3 + 2]
        colors.append((r, g, b))

    colors = [clr for clr in colors if clr != (0, 0, 0)]

    return [np.array(clr)/255 for clr in colors]

# Constants
PAL_START = b"\xC0\x01"
PAL_ENTRY = b"\x00\x01"
PAL_END   = b"\xC0\x02"
STR_TERM  = b"\x00\x00"

def ase_converter(p, ase_path):
    gimp.progress_init()

    if ase_path.startswith('file:///'):
        ase_path = ase_path.replace('file:///', '/')

    try:
        with open(ase_path, 'rb') as ase_file:
            do_convert(ase_file)
    except Exception as ex:
        pdb.gimp_message(
            'Error: %s\n' % traceback.format_exc() +
            "Try starting GIMP from a console to see what went wrong."
        )
        if isinstance(ex, SystemExit):
            raise

def load(path):
    with open(path, 'rb') as ase_file:
        return do_convert(ase_file)

def do_convert(ase_file):
    inaccuracy_warning = False

    ase_header = ase_file.read(4)
    if ase_header != b"ASEF":
        raise Exception("\"" + ase_file.name  + "\" is not an ASE file.")

    ase_version_major = unpack('>H', ase_file.read(2))[0]
    ase_version_minor = unpack('>H', ase_file.read(2))[0]

    if ase_version_major != 1:
        raise Exception("Major version of given file is not 1; not compatible with script.")

    if ase_version_minor != 0:
        print("Warning: Minor version is not 0; might not work.")

    ase_nbblocks = unpack('>I', ase_file.read(4))[0]

    if ase_nbblocks == 0:
        raise Exception("ASE file has no blocks")

    palettes = [[]]
    pal_title = ""
    pal_ncols = 0

    for block in range(ase_nbblocks):
        block_type = ase_file.read(2)
        block_len = unpack('>I', ase_file.read(4))[0]
        block_data = BytesIO(ase_file.read(block_len))

        if block_type == PAL_START:
            if palettes and palettes[-1]:
                raise ValueError('Unexpected start of palette')
            palettes.append([])

        elif block_type == PAL_ENTRY:
            if not palettes:
                raise ValueError("Unexpected palette entry before palette start")

            col_name = read_ase_string(block_data)
            col_model = block_data.read(4)

            if col_model in [b"LAB ", b"CMYK"]:
                if not inaccuracy_warning:
                    print("Warning: Converting from LAB or CMYK colors is inaccurate.")
                    inaccuracy_warning = True

            if col_model == b"RGB ":
                red   = unpack_from('>f', block_data.read(4))[0]
                green = unpack_from('>f', block_data.read(4))[0]
                blue  = unpack_from('>f', block_data.read(4))[0]
                palettes[-1].append(np.array([red, green, blue]))
            elif col_model == b"LAB ":
                lab_L = unpack_from('>f', block_data.read(4))[0]
                lab_A = unpack_from('>f', block_data.read(4))[0]
                lab_B = unpack_from('>f', block_data.read(4))[0]

                lab_L = lab_L * 100
                var_Y = (lab_L + 16) / 116
                var_X = lab_A / 500 + var_Y
                var_Z = var_Y - lab_B / 200

                var_Y = var_Y**3 if var_Y**3 > 0.008856 else (var_Y - 16 / 116) / 7.787
                var_X = var_X**3 if var_X**3 > 0.008856 else (var_X - 16 / 116) / 7.787
                var_Z = var_Z**3 if var_Z**3 > 0.008856 else (var_Z - 16 / 116) / 7.787

                ref_X, ref_Y, ref_Z = 95.047, 100.000, 108.883
                X = ref_X * var_X
                Y = ref_Y * var_Y
                Z = ref_Z * var_Z

                var_X = X / 100
                var_Y = Y / 100
                var_Z = Z / 100

                var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986
                var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415
                var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570

                var_R = 1.055 * (var_R ** (1 / 2.4)) - 0.055 if var_R > 0.0031308 else 12.92 * var_R
                var_G = 1.055 * (var_G ** (1 / 2.4)) - 0.055 if var_G > 0.0031308 else 12.92 * var_G
                var_B = 1.055 * (var_B ** (1 / 2.4)) - 0.055 if var_B > 0.0031308 else 12.92 * var_B

                palettes[-1].append(np.array([var_R, var_G, var_B]))

            elif col_model == b"CMYK":
                cmyk_C = unpack_from('>f', block_data.read(4))[0]
                cmyk_M = unpack_from('>f', block_data.read(4))[0]
                cmyk_Y = unpack_from('>f', block_data.read(4))[0]
                cmyk_K = unpack_from('>f', block_data.read(4))[0]

                C = cmyk_C * (1 - cmyk_K) + cmyk_K
                M = cmyk_M * (1 - cmyk_K) + cmyk_K
                Y = cmyk_Y * (1 - cmyk_K) + cmyk_K

                R = 1 - C
                G = 1 - M
                B = 1 - Y

                palettes[-1].append(np.array([R, G, B]))

            else:
                print("Warning: Unknown color model \"" + str(col_model) + "\", skipped")

        elif block_type == PAL_END:
            if not palettes[-1]:
                raise ValueError("Unexpected palette end before palette start")


        else:
            raise ValueError("Error: Unexpected block type " + str(block_type))

    return [pal for pal in palettes if pal]

def read_ase_string(data):
    length = unpack('>H', data.read(2))[0] - 1
    raw_string = data.read(length * 2)
    terminator = data.read(2)

    if terminator != STR_TERM:
        raise ValueError("Expected double-NUL terminated string")

    return str(raw_string, "utf_16_be")
