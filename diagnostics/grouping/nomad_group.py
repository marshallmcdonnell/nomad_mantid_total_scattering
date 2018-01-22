#!/usr/bin/env python
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-nh", "--num_groups_h", type=int, default=8)
parser.add_argument("-nw", "--num_groups_w", type=int, default=2)
args = parser.parse_args()

######################################################################
# parameters for generating groups
######################################################################

bank_total = 99 # number in mantid geometry

pix_per_bank_h = 128
pix_per_bank_w = 8
pix_per_bank = pix_per_bank_h * pix_per_bank_w

pix_total = bank_total * pix_per_bank_h *pix_per_bank_w

num_groups_h = args.num_groups_h
num_groups_w = args.num_groups_w

pix_per_group_h = pix_per_bank_h / num_groups_h
pix_per_group_w = pix_per_bank_w / num_groups_w
pix_per_group = pix_per_group_h * pix_per_group_w


# filename based off of parameters
filename = "nomad_group_%d_%d.xml" % (pix_per_group_h, pix_per_group_w)

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
print "writing out to %s" % filename

######################################################################
# create the file
######################################################################
handle = file(filename, 'w')

handle.write('<?xml version="1.0" encoding="UTF-8" ?>\n')
handle.write('<detector-grouping instrument="NOMAD">\n')
groupnum = 1
for i in xrange(bank_total):

    for j in xrange(pix_per_bank_w/pix_per_group_w):
        # left bounds are created fresh for each iteration
        left  = np.arange(0, pix_per_group_w,dtype=int) * pix_per_bank_h
        left += i*pix_per_bank + j*(pix_per_bank_h/pix_per_group_h)*pix_per_group

        # loop over the length of the tube
        for k in xrange(pix_per_bank_h/pix_per_group_h):
            right = left + (pix_per_group_h - 1)

            groups = ["%d-%d" % (l,r) for (l,r) in zip(left, right)]
            handle.write('<group ID="%d">\n' % groupnum)
            handle.write('<detids>%s</detids>\n' % ', '.join(groups))
            handle.write('</group>\n')

            left += pix_per_group_h
            groupnum += 1
handle.write('</detector-grouping>\n')

# Check that the calculation ended with the total number
# of pixels. One is added because of zero-indexing
if (right[-1] + 1) != (bank_total * pix_per_bank_h *pix_per_bank_w):
    raise RuntimeError("Number of pixels didn't work")

