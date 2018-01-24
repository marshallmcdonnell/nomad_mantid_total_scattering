#!/usr/bin/env python
import numpy as np
import argparse
from diagnostics import io
from diagnostics import grouping


parser = argparse.ArgumentParser()
parser.add_argument("-nh", "--num_groups_h", type=int, default=8)
parser.add_argument("-nw", "--num_groups_w", type=int, default=2)
parser.add_argument("--mask-ids", type=str, default=None, dest="mask_ids",
                    help="List of pixels to mask")
parser.add_argument("--mask-ids-file", type=str, default=None,dest="mask_ids_file",
                    help="Filename with list of pixels to mask")
args = parser.parse_args()

######################################################################
# parameters for generating groups
######################################################################

bank_total = 99 # number in mantid geometry

pix_per_bank_h = 128 # == 128
pix_per_bank_w = 8   # == 8
pix_per_bank = pix_per_bank_h * pix_per_bank_w

pix_total = bank_total * pix_per_bank_h *pix_per_bank_w

num_groups_h = args.num_groups_h
num_groups_w = args.num_groups_w

pix_per_group_h = pix_per_bank_h / num_groups_h
pix_per_group_w = pix_per_bank_w / num_groups_w
pix_per_group = pix_per_group_h * pix_per_group_w


# print diagnositic information

num_dashes=35
print "-"*num_dashes
print "tot %4d  pixels per group" % (pix_per_group)
print "w   %4d  pixels per group" % (pix_per_group_w)
print "h   %4d  pixels per group" % (pix_per_group_h)
print "-"*num_dashes
print "tot %4d / %4d = %4d groups per bank" % (pix_per_bank , pix_per_group, pix_per_bank/pix_per_group)
print "h   %4d / %4d = %4d groups per bank" % (pix_per_bank_h , pix_per_group_h, pix_per_bank_h/pix_per_group_h)
print "w   %4d / %4d = %4d groups per bank" % (pix_per_bank_w , pix_per_group_w, pix_per_bank_w/pix_per_group_w)
print "-"*num_dashes
print "total pixels:", pix_total
print "total groups:", pix_total / pix_per_group
print "-"*num_dashes

#-----------------------------------------------------
# Create the unique pixel ids (i=0 -> N, N=total pixels)
pixels = np.arange(0,pix_total,dtype=int)

#-----------------------------------------------------
# Initialize grouping (with None)
grouper = np.full_like(pixels, -1)

#-----------------------------------------------------
# Create grouper
group_num = 0
for i in xrange(bank_total):

    for j in xrange(pix_per_bank_w/pix_per_group_w):
        # left bounds are created fresh for each iteration
        left  = np.arange(0, pix_per_group_w,dtype=int) * pix_per_bank_h
        left += i*pix_per_bank + j*(pix_per_bank_h/pix_per_group_h)*pix_per_group

        # loop over the length of the tube
        for k in xrange(pix_per_bank_h/pix_per_group_h):
            right = left + (pix_per_group_h - 1)

            group_pixel_ids = list()
            for (l,r) in zip(left, right):
                group_pixel_ids += list(io.utils.expand_ints("%d-%d" % (l,r)))
            print("group: %d l-r: %s" % (group_num,io.utils.compress_ints(group_pixel_ids)))
            grouper[group_pixel_ids] = group_num

            left += pix_per_group_h
            group_num += 1
io.utils.print_array("Pixels:", pixels)
io.utils.print_array("Grouping:", grouper)
# Check that the calculation ended with the total number
# of pixels. One is added because of zero-indexing
if (right[-1] + 1) != (bank_total * pix_per_bank_h *pix_per_bank_w):
    raise RuntimeError("Number of pixels didn't work")

#-----------------------------------------------------
# Create the mask to apply to both data and grouping

mask_ids=str()
if args.mask_ids:
    mask_ids += args.mask_ids
if args.mask_ids_file:
    with open(args.mask_ids_file, 'r') as f:
        mask_ids_from_file = ",".join(line.strip() for line in f)
        if mask_ids:
            mask_ids += ','+mask_ids_from_file
        else:
            mask_ids = mask_ids_from_file
mask = grouping.utils.create_mask(pixels,mask_ids)


#-----------------------------------------------------
# Mask and Group
masked_pixels, masked_grouper, groups = grouping.utils.mask_and_group(pixels, grouper, mask)

#-----------------------------------------------------
# Print results
io.utils.print_array("Masked Data", pixels[mask])
io.utils.print_array("Masked Group", grouper[mask])
io.utils.print_array("New Masked Group", masked_grouper)

for i, group in enumerate(groups):
    print("Group #: {} Data: {}".format(i, io.utils.compress_ints(group)))


######################################################################
# create the file
######################################################################
# filename based off of parameters
if args.mask_ids or args.mask_ids_file:
    filename = "nomad_group_%d_%d_masked.xml" % (pix_per_group_h, pix_per_group_w)
else:
    filename = "nomad_group_%d_%d.xml" % (pix_per_group_h, pix_per_group_w)
print "writing out to %s" % filename
handle = file(filename, 'w')
handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
handle.write('<detector-grouping instrument="NOMAD">\n')

for groupnum, group in enumerate(groups):
            handle.write('<group ID="%d">\n' % (groupnum+1))
            handle.write('<detids>%s</detids>\n' % (io.utils.compress_ints(group)))
            handle.write('</group>\n')
handle.write('</detector-grouping>\n')


