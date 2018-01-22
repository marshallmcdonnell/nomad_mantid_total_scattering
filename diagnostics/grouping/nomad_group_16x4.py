#!/usr/bin/env python
group=list()
numpacks=50
coldiv=16
rowdiv=4
colwidth=128
rowhieght=8
pack=0
colsteps=colwidth/coldiv
print "The column steps are", colsteps
rowsteps=rowhieght/rowdiv
print "The row steps are", rowsteps

numbanks=(numpacks*colwidth*rowhieght)/(coldiv*rowdiv)
print "The numbers of final spectrum is", numbanks


for i in range(1,numbanks+2):
    group.append([])
    
for nopack in range(1,numpacks+1):
    pack=pack+1
    for jj in range (1,rowsteps+1):
        for i in range (1,colsteps+1):
            for ii in range (1,rowdiv+1):
                for j in range(1,coldiv+1):
                    bank=i+(jj-1)*rowdiv+((pack-1)*(colsteps*rowsteps))
                    if pack==1 : print bank,j
                    group[bank].append(j+coldiv*(i-1)+((ii-1)*colwidth)+(pack-1)*colwidth*rowhieght)
    

#for i in range (1,numbanks+1):
#   print i,group[i]

print 1, group[1]
print 8, group[8]
