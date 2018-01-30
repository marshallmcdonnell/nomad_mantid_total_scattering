pro read_bm,tof,lambda,bm,bmeff,bmpp,bmpc,fluxonsample,filename=file,calcfos=calcfos,inter=inter,help=help

; ##################################################################################
; ###################### API for IDL (event Data) ##################################
; ##################################################################################

if not var_defined(help) then help=0
if not var_defined(inter) then inter=0
if help then begin
print,'USAGE:read_bm,tof,lambda,bm,bmeff,bmpp,bmpc,fluxonsample,filename=file,calcfos=calcfos,help=help'
print,'tof (output) time of flight vector'
print,'lambda (output) wavelength vector'
print,'bm: raw beam monitor count / micro s TOF bin'
print,'bmeff: corrected for detection efficiency '
print,'bmpp: beam monitor per pulse'
print,'bmpc: beam monitor per proton charge'
print,'fluxonsample flux on sample using simulated neutron transport for 6mm sample'
print,'filename (optional) beam monito file'
print,'calcfos=1 try to calculate the flux on sample'
return
end

; INPUT 
; bmeff is total number of neutrons per bin over entire time
; bmpp is per pulse and 1 A 
; fluxonsample is normalized to the sample position on a spot .6cm wide and 2 cm high over the entire time

modtomon=18.62

path = "/SNS/NOM"  ;path to event file
if not var_defined(calcfos) then calcfos=1

; ******* Pickup Dialog Box **************

sublength=-1
if not var_defined(file) then begin
while sublength lt 0 do begin
 file = dialog_pickfile(/must_exist, $
 title = 'Select a beam monitor file', $
 filter = ['*.dat'], $
 path = path,$
 get_path = Path)
; ****************************************

; to get only the last part of the name
file_list = strsplit(file,'\',/extract, count=length)
file_name = file_list[length-1]
sublength=strpos(file,'bmon')
if sublength lt 0 then print, 'Wrong file?'
end
end

sublength=strpos(file,'bmon')
file_pid=strmid(file,0,sublength)+'pulseid.dat'
read_pulsid,plow,phigh,evid,char,filename=file_pid

openr,1,file
n=16000l
tof=long(findgen(16000))
bm=tof
bmeff=findgen(16000)
readu,1,bm
close,1
lambda=.003955*tof/modtomon
p=0.000794807
; gives 1.03 10^-5 effiency at 1.8 A
e0=5333d0*lambda/1.8*2.43e-5*p
bmeff(1:*)=bm(1:*)/(1.d0-exp(-e0(1:*)*.1))
w=where(char gt 0,n1)
; n1 is the number of pulse with non-zero charge
bmpp=bmeff/float(n1)/(lambda(1)-lambda(0))
; devide by delta lambda
tchar=total(char(w))*1e-6
bmpc=bmeff/tchar/(lambda(1)-lambda(0))
if calcfos then begin
rstd,l,ios,'/SNS/NOM/shared/D10_lamda_I_sample_0015.txt',nz=6
rstd,l,iomon,'/SNS/NOM/shared/D10_l_i_exitguide_0036.txt',nz=6
fluxonsample=bmpp/interpol(iomon(*,3),iomon(*,0),lambda)*interpol(ios(*,3),ios(*,0),lambda)*float(n1)*(lambda(1)-lambda(0))
fluxonsample(where(lambda gt 3.1 or lambda lt 0.05))=0
endif
end
