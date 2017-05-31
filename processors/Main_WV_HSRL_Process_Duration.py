# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:10:46 2017

@author: mhayman
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 08:45:42 2017

@author: mhayman
"""

import numpy as np
import matplotlib.pyplot as plt
import LidarProfileFunctions as lp
import WVProfileFunctions as wv
import scipy.interpolate

import FourierOpticsLib as FO

import datetime
import json

#import glob

"""
USER INPUTS
"""


#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,5,12,startHr=15,duration=20)
Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,5,31,startHr=0,duration=24)


plotAsDays = False
getMLE_extinction = False
run_MLE = False
runKlett = False

save_as_nc = False
save_figs = True

run_geo_cal = False

model_atm = True

nctag = ''

use_diff_geo = False   # no diff geo correction after April ???
use_geo = True

use_mask = False
SNRmask = 2.0  #SNR level used to decide what data points we keep in the final data product
countLim = 2.0

MaxAlt = 12e3 #12e3
WV_Min_Alt = 350  # mask data below this altitude

KlettAlt = 14e3  # altitude where Klett inversion starts

# set bin sizes
tres_hsrl = 1.0*60.0  # HSRL bin time resolution in seconds (2 sec typical base)
tres_wv = 1.0*60.0    # HSRL bin time resolution in seconds (2 sec typical base)
zres = 37.5  # bin range resolution in m (37.5 m typical base)

# parameters for WV Channels smoothing
tsmooth_wv = 5*60 # convolution kernal time (HW sigma) in seconds
zsmooth_wv = 1 #150  # convolution kernal range (HW sigma) in meters
zsmooth2_wv = np.sqrt(150**2+150**2)  # 75 # second range smoothing conducted on the actual WV retrieval


"""
Paths
"""
# path to data
basepath = '/scr/eldora1/MSU_h2o_data/'

# path for saving data
save_data_path = '/h/eol/mhayman/DIAL/Processed_Data/'
save_fig_path = '/h/eol/mhayman/DIAL/Processed_Data/Plots/'

# path to sonde data
sonde_path = '/scr/eldora1/HSRL_data/'

# path to calibration files
cal_path = '/h/eol/mhayman/PythonScripts/NCAR-LidarProcessing/calibrations/'
cal_file = cal_path+'dlb_calvals_msu.json'


# field labels
FieldLabel_WV = 'FF'
ON_FileBase = 'Online_Raw_Data.dat'
OFF_FileBase = 'Offline_Raw_Data.dat'

FieldLabel_HSRL = 'NF'
MolFileBase = 'Online_Raw_Data.dat'
CombFileBase = 'Offline_Raw_Data.dat'



"""
Begin Processing
"""

ProcStart = datetime.datetime(Years[0],Months[0],Days[0],np.int(Hours[0][0]),np.int(np.remainder(Hours[0][0],1.0)*60))

DateLabel = ProcStart.strftime("%A %B %d, %Y")

with open(cal_file,"r") as f:
    cal_jdata = json.loads(f.read())
f.close()

MCSbins = lp.get_calval(ProcStart,cal_jdata,'MCS bins')[0]
BinWidth = lp.get_calval(ProcStart,cal_jdata,'Bin Width')[0]
dt = lp.get_calval(ProcStart,cal_jdata,'Data Time Resolution')[0]
LaserPulseWidth = lp.get_calval(ProcStart,cal_jdata,'Laser Pulse Width')[0]

if use_diff_geo:
    cal_value = lp.get_calval(ProcStart,cal_jdata,'Molecular Gain',cond=['diff_geo','!=','none'],returnlist=['value','diff_geo'])
    diff_geo_file = cal_path+cal_value[1]
else:
    cal_value = lp.get_calval(ProcStart,cal_jdata,'Molecular Gain',cond=['diff_geo','=','none'])
MolGain = cal_value[0]

if use_geo:
    cal_value = lp.get_calval(ProcStart,cal_jdata,'Geo File Record',returnlist=['filename'])
    geo_file = cal_path+cal_value[0]

dR = BinWidth*lp.c/2  # profile range resolution (500e-9*c/2)-typical became 100e-9*c/2 on 2/22/2017
Roffset = ((1.25+0.5)-0.5/2)*lp.c*LaserPulseWidth/2  # offset in range

zres = np.max([np.round(zres/dR),1.0])*dR  #only allow z resolution to be integer increments of the MCS range

BGIndex = -50; # negative number provides an index from the end of the array
Cam = 0.00 # Cross talk of aerosols into the molecular channel - 0.005 on Dec 21 2016 after 18.5UTC
            # 0.033 found for 4/18/2017 11UTC extinction test case




if save_as_nc or save_figs:
    ncfilename0 = lp.create_ncfilename('MSU_WVDIAL_DLBHSRL',Years,Months,Days,Hours,tag=nctag)
    ncfilename = save_data_path+ncfilename0
    figfilename = save_fig_path + ncfilename0[:-3]

if use_geo:
    geo_data = np.load(geo_file)
    geo_corr = geo_data['geo_prof']
    geo_corr0 = geo_corr[100,1]  # normalize to bin 100
    geo_corr[:,1] = geo_corr[:,1]/geo_corr0
    if any('sonde_scale' in s for s in geo_data.keys()):
        sonde_scale=geo_data['sonde_scale']/geo_corr0
    else:
        sonde_scale=1.0/geo_corr0
else:
    sonde_scale=1.0




# define time grid for lidar signal processing to occur
MasterTimeHSRL = np.arange(Hours[0,0]*3600,Days.size*24*3600-(24-Hours[-1,-1])*3600,tres_hsrl)
MasterTimeWV = np.arange(Hours[0,0]*3600,Days.size*24*3600-(24-Hours[-1,-1])*3600,tres_wv)


[Molecular,CombHi],[lambda_hsrl,[surf_temp],[surf_pres],[surf_humid]],HourLim = wv.Load_DLB_Data(basepath,FieldLabel_HSRL,[MolFileBase,CombFileBase],MasterTimeHSRL,Years,Months,Days,Hours,MCSbins,lidar='DLB-HSRL',dt=dt,Roffset=Roffset,BinWidth=BinWidth)
[OnLine,OffLine],[[lambda_on,lambda_off],[surf_temp],[surf_pres],[surf_humid]],HourLim = wv.Load_DLB_Data(basepath,FieldLabel_WV,[ON_FileBase,OFF_FileBase],MasterTimeWV,Years,Months,Days,Hours,MCSbins,lidar='WV-DIAL',dt=dt,Roffset=Roffset,BinWidth=BinWidth)

# create a humidity "lidar profile" to add to the derived water vapor before plotting
# an extra set of masked range bins is used to create range separation from the 
# lidar profile starting at ~ 350 m
surf_humid = np.hstack((surf_humid[:,np.newaxis],np.zeros(surf_humid[:,np.newaxis].shape)))
surf_humid_mask = np.zeros(surf_humid.shape)
surf_humid_mask[:,1] = True
surf_humid = np.ma.array(surf_humid,mask=surf_humid_mask)
SurfaceHumid = lp.LidarProfile(surf_humid,OffLine.time, \
    label='Absolute Humidity',descript = 'Absolute Humidity from Surface Station', \
    bin0=0,lidar='Surface Station',binwidth=2*100/lp.c, \
    StartDate=ProcStart)
SurfaceHumid.profile_type = '$g/m^{3}$'

#lp.pcolor_profiles([OnLine,OffLine],climits=[[-8.0,-4.0],[-8.0,-4.0]],scale=['log','log'],plotAsDays=plotAsDays) 
#lp.plotprofiles([OnLine,OffLine])

# WV-DIAL
OnLineRaw = OnLine.copy()
OnLine.mask_range('index',np.arange(1))
OnLine.conv(tsmooth_wv/tres_wv,zsmooth_wv/zres,keep_mask=True)
#OnLine.mask_range('<=',300)
OnLine.slice_time(HourLim*3600)
#OnLine.nonlinear_correct(30e-9);
OnLine.bg_subtract(BGIndex)

OffLineRaw = OffLine.copy()
OffLine.mask_range('index',np.arange(1))
OffLine.conv(tsmooth_wv/tres_wv,zsmooth_wv/zres,keep_mask=True)
#OffLine.mask_range('<=',300)
OffLine.slice_time(HourLim*3600)
#OffLine.nonlinear_correct(30e-9);
OffLine.bg_subtract(BGIndex)

#lp.plotprofiles([OnLine,OffLine,OnLineRaw,OffLineRaw])

# HSRL
Molecular.slice_time(HourLim*3600)
MolRaw = Molecular.copy()
#Molecular.nonlinear_correct(38e-9);
Molecular.bg_subtract(BGIndex)

CombHi.slice_time(HourLim*3600)
CombRaw = CombHi.copy()
CombHi.nonlinear_correct(29.4e-9);
CombHi.bg_subtract(BGIndex)



# HSRL data matched to WV-DIAL

Mol2 = MolRaw.copy()
Comb2 = CombRaw.copy()

Mol2.mask_range('index',np.arange(1))
Mol2.conv(tsmooth_wv/tres_wv,zsmooth_wv/zres,keep_mask=True)
Mol2.slice_time(HourLim*3600)
Mol2.bg_subtract(BGIndex)

Comb2.nonlinear_correct(29.4e-9);
Comb2.mask_range('index',np.arange(1))
Comb2.conv(5.0*60.0/tres_wv/2,4.0/2,keep_mask=True)
Comb2.slice_time(HourLim*3600)
Comb2.bg_subtract(BGIndex)


#####  NEED TO CORRECT TIME SLICES BASED ON ALL 4 PROFILES

# WV-DIAL time slices
if OffLine.time.size > OnLine.time.size:
    OffLine.slice_time_index(time_lim=np.array([0,OnLine.time.size]))
elif OffLine.time.size < OnLine.time.size:
    OnLine.slice_time_index(time_lim=np.array([0,OffLine.time.size]))

# HSRL time slices
if CombHi.time.size > Molecular.time.size:
    CombHi.slice_time_index(time_lim=np.array([0,Molecular.time.size]))
elif CombHi.time.size < Molecular.time.size:
    Molecular.slice_time_index(time_lim=np.array([0,CombHi.time.size-1]))

# HSRL time slices
if Comb2.time.size > Mol2.time.size:
    Comb2.slice_time_index(time_lim=np.array([0,Mol2.time.size]))
elif Comb2.time.size < Mol2.time.size:
    Mol2.slice_time_index(time_lim=np.array([0,Comb2.time.size-1]))

# mask based on raw counts - remove points where there are < 2 counts
if use_mask:
    NanMask_wv = np.logical_or(OnLine.profile < 2.0,OffLine.profile < 2.0)
    OnLine.profile = np.ma.array(OnLine.profile,mask=NanMask_wv)
    OffLine.profile = np.ma.array(OffLine.profile,mask=NanMask_wv)
    
    NanMask_hsrl = np.logical_or(Molecular.profile < 2.0,CombHi.profile < 2.0)
    Molecular.profile = np.ma.array(Molecular.profile,mask=NanMask_hsrl)
    CombHi.profile = np.ma.array(CombHi.profile,mask=NanMask_hsrl)




#OnLine.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_geo:
#    Molecular.geo_overlap_correct(geo_corr)
#OnLine.range_correct();
OnLine.slice_range(range_lim=[0,MaxAlt])
OnLine.range_resample(delta_R=zres,update=True)
#OnLine.conv(5.0,0.7)  # regrid by convolution
OnLine.slice_range_index(range_lim=[1,1e6])  # remove bottom bin

#OffLine.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_diff_geo:
#    OffLine.diff_geo_overlap_correct(diff_geo_corr,geo_reference='online')
#OffLine.range_correct()
OffLine.slice_range(range_lim=[0,MaxAlt])
OffLine.range_resample(delta_R=zres,update=True)
#OffLine.conv(5.0,0.7)  # regrid by convolution
OffLine.slice_range_index(range_lim=[1,1e6])  # remove bottom bin


Molecular.range_correct();
Molecular.slice_range(range_lim=[0,MaxAlt])
Molecular.range_resample(delta_R=zres,update=True)
Molecular.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
#Molecular.conv(1.5,2.0)  # regrid by convolution
MolRaw.slice_range_index(range_lim=[1,1e6])  # remove bottom bin

##CombHi.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_diff_geo:
#    CombHi.diff_geo_overlap_correct(diff_geo_corr,geo_reference='mol')

CombHi.range_correct()
CombHi.slice_range(range_lim=[0,MaxAlt])
CombHi.range_resample(delta_R=zres,update=True)
CombHi.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
#CombHi.conv(1.5,2.0)  # regrid by convolution
CombRaw.slice_range_index(range_lim=[1,1e6])  # remove bottom bin

# Rescale molecular channel to match combined channel gain
#MolGain = 1.33  # updated 4/11/2017
#MolGain = 3.17  # updated 5/12/2017
Molecular.gain_scale(MolGain)


Mol2.range_correct();
Mol2.slice_range(range_lim=[0,MaxAlt])
Mol2.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
Mol2.range_resample(delta_R=zres,update=True)

Comb2.range_correct()
Comb2.slice_range(range_lim=[0,MaxAlt])
Comb2.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
Comb2.range_resample(delta_R=zres,update=True)

Mol2.gain_scale(MolGain)

# Calculate backscatter ratio to build a cloud gradient mask in WV data
BSRwv = Comb2.profile/Mol2.profile
t1,t2,BSR_kernel = lp.get_conv_kernel(0.0,zsmooth2_wv/zres,norm=True)
BSRwv = lp.conv2d(BSRwv,BSR_kernel,keep_mask=True)
dBSRwv = np.diff(BSRwv,axis=1)/Mol2.mean_dR  # range derivative of BSR
BSRwv_interp = 0.5*(BSRwv[:,:-1]+BSRwv[:,1:])  # differential range interpolated BSR

BSR_mask = np.zeros(BSRwv_interp.shape)
BSR_mask[np.nonzero(np.abs(dBSRwv/BSRwv_interp) > 0.002) ] = 1 # 0.005
BSR_mask[:,np.nonzero(Comb2.range_array < 1e3)] = 0  # don't use the mask below 1 km


# Correct Molecular Cross Talk
if Cam > 0:
    lp.FilterCrossTalkCorrect(Molecular,CombHi,Cam,smart=True)

if model_atm: 
    beta_mol_sonde,temp,pres = lp.get_beta_m_model(OffLine,surf_temp,surf_pres,returnTP=True)
    pres.descript = 'Ideal Atmosphere Pressure in atm'
else:
    beta_mol_sonde,sonde_time,sonde_index_prof,temp,pres,sonde_index = lp.get_beta_m_sonde(OffLine,Years,Months,Days,sonde_path,interp=True,returnTP=True)
    pres.descript = 'Sonde Measured Pressure in atm'
# convert pressure from Pa to atm.
pres.gain_scale(9.86923e-6)  
pres.profile_type = '$atm.$'


# Plot Integrated Profiles
lp.plotprofiles([OffLine,OnLine,Molecular,CombHi])

OnInt = OnLine.copy();
OnInt.time_integrate();
OffInt = OffLine.copy();
OffInt.time_integrate();

# Calculate Aerosol Backscatter Coefficient
aer_beta_dlb = lp.AerosolBackscatter(Molecular,CombHi,beta_mol_sonde)

MolInt = Molecular.copy();
MolInt.time_integrate();
CombInt = CombHi.copy();
CombInt.time_integrate();
sonde_int = beta_mol_sonde.copy()
sonde_int.time_integrate();
aer_beta_dlb_int = lp.AerosolBackscatter(MolInt,CombInt,sonde_int)
lp.plotprofiles([aer_beta_dlb_int,sonde_int])


if use_geo:
    Molecular.geo_overlap_correct(geo_corr)
    CombHi.geo_overlap_correct(geo_corr)

# Obtain low pass filtered instances of the profiles for extinction and 
# MLE initialization
MolLP = Molecular.copy()
MolLP.conv(4,2)
CombLP = CombHi.copy()
CombLP.conv(4,2)
aer_beta_LP = lp.AerosolBackscatter(MolLP,CombLP,beta_mol_sonde)




#### Grab Sonde Data
#sondefilename = '/scr/eldora1/HSRL_data/'+YearStr+'/'+MonthStr+'/sondes.DNR.nc'
#sonde_index = 2*Days[-1]
##(Man or SigT)
#f = netcdf.netcdf_file(sondefilename, 'r')
#TempDat = f.variables['tpSigT'].data.copy()  # Kelvin
#PresDat = f.variables['prSigT'].data.copy()*100.0  # hPa - convert to Pa (or Man or SigT)
#SondeTime = f.variables['relTime'].data.copy() # synoptic time: Seconds since (1970-1-1 00:00:0.0) 
#SondeAlt = f.variables['htSigT'].data.copy()  # geopotential altitude in m
#StatElev = f.variables['staElev'].data.copy()  # launch elevation in m
#f.close()
#
#TempDat[np.nonzero(np.logical_or(TempDat < 173.0, TempDat > 373.0))] = np.nan;
#PresDat[np.nonzero(np.logical_or(PresDat < 1.0*100, PresDat > 1500.0*100))] = np.nan;
#
#sonde_index = np.min([np.shape(SondeAlt)[0]-1,sonde_index])
## Obtain sonde data for backscatter coefficient estimation
#Tsonde = np.interp(OffLine.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],TempDat[sonde_index,:])
#Psonde = np.interp(OffLine.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],PresDat[sonde_index,:])
#Psonde = Psonde*9.86923e-6  # convert Pressure to atm from Pa

isonde = np.argmin(pres.time-pres.time/2.0)
Psonde = pres.profile[isonde,:]
Tsonde = temp.profile[isonde,:]

#nWV = wv.WaterVapor_2D(OnLine,OffLine,lambda_on,lambda_off,pres,temp)
nWV = wv.WaterVapor_Simple(OnLine,OffLine,Psonde,Tsonde)
nWV.conv(0.0,zsmooth2_wv/zres)
nWV.mask_range('<=',WV_Min_Alt)

if use_mask:
    wv_snr_mask = np.zeros(nWV.profile.shape)
    wv_snr_mask[np.nonzero(nWV.SNR() < SNRmask)] = 1
    nWV.mask(wv_snr_mask)
    nWV.mask(BSR_mask)
    
    aer_mask = np.zeros(aer_beta_dlb.profile.shape)
    aer_mask[aer_beta_dlb.SNR() < SNRmask] = 1
    aer_beta_dlb.mask(aer_mask)

nWV.cat_range(SurfaceHumid)


#dnu = np.linspace(-7e9,7e9,400)
#inuL = np.argmin(np.abs(dnu))
#MolSpec = lp.RB_Spectrum(Tsonde,Psonde,OffLine.wavelength,nu=dnu)
#nuOff = lp.c/OffLine.wavelength
#nuOn = lp.c/OnLine.wavelength
#
#
#
#Filter = FO.FP_Etalon(1.0932e9,43.5e9,nuOff,efficiency=0.95,InWavelength=False)
#Toffline = Filter.spectrum(dnu+nuOff,InWavelength=False,aoi=0.0,transmit=True)
#Tonline = Filter.spectrum(dnu+nuOn,InWavelength=False,aoi=0.0,transmit=True)
#
#Toffline2 = Filter.spectrum(lp.c/lambda_on,InWavelength=False,aoi=0.0,transmit=True)
#Tonline2 = Filter.spectrum(lp.c/lambda_off,InWavelength=False,aoi=0.0,transmit=True)
#
#plt.figure(); 
#plt.plot(dnu+nuOn,Tonline); 
#plt.plot(dnu+nuOff,Toffline);
#plt.plot(lp.c/lambda_on,Tonline2,'bx',label='Online'); 
#plt.plot(lp.c/lambda_off,Toffline2,'gx',label='Offline');
#
##plt.plot(dnu[inuL]+nuOn,Tonline[inuL],'bx',label='Online')
##plt.plot(dnu[inuL]+nuOff,Toffline[inuL],'gx',label='Offline')
#plt.grid(b=True)
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Transmission')
#
#nuWV = np.linspace(lp.c/828.5e-9,lp.c/828e-9,500)
#sigWV = lp.WV_ExtinctionFromHITRAN(nuWV,Tsonde,Psonde) #,nuLim=np.array([lp.c/835e-9,lp.c/825e-9]))  
#ind_on = np.argmin(np.abs(lp.c/OnLine.wavelength-nuWV))
#ind_off = np.argmin(np.abs(lp.c/OffLine.wavelength-nuWV))
#
#sigOn = sigWV[:,ind_on]
#sigOff = sigWV[:,ind_off]
#
#sigF = scipy.interpolate.interp1d(nuWV,sigWV)
#sigWVOn = sigF(dnu+nuOn).T
#sigWVOff = sigF(dnu+nuOff).T
#
##sigWVOn = lp.WV_ExtinctionFromHITRAN(nuOn+dnu,Tsonde,Psonde) 
##sigWVOff = lp.WV_ExtinctionFromHITRAN(nuOff+dnu,Tsonde,Psonde)
#
#sigOn = sigWVOn[inuL,:]
#sigOff = sigWVOff[inuL,:]
#
#
#
#range_diff = OnLine.range_array[1:]-OnLine.mean_dR/2.0
#dsig = np.interp(range_diff,OnLine.range_array,sigOn-sigOff)
#nWVp = -1.0/(2*(dsig)[np.newaxis,:])*np.diff(np.log(OnLine.profile/OffLine.profile),axis=1)/OnLine.mean_dR
#
###nWV = lp.LidarProfile(nWVp,OnLine.time,label='Water Vapor Number Density',descript = 'Water Vapor Number Density',bin0=-Roffset/dR,lidar='WV-DIAL',binwidth=BinWidth,StartDate=ProcStart)
##nWV = OnLine.copy()
##nWV.profile = nWVp
##nWV.label = 'Water Vapor Number Density'
##nWV.descript = 'Water Vapor Number Density'
##nWV.profile_type = '$m^{-3}$'
##nWV.range_array = nWV.range_array[1:]-nWV.mean_dR/2.0
##
### convert to g/m^3
##nWV.gain_scale(lp.mH2O/lp.N_A)  
##nWV.profile_type = '$g/m^{3}$'
##
##nWV.conv(2.0,3.0)
#
##nWV = -1.0/(2*(sigOn-sigOff)[np.newaxis,:])*np.diff(OnLine.profile/OffLine.profile,axis=1)
##nWV2 = -1.0/(2*(dsig)[np.newaxis,:])*np.log(OnLine.profile[:,1:]*OffLine.profile[:,:-1]/(OffLine.profile[:,1:]*OnLine.profile[:,:-1]))/OnLine.mean_dR
#
#nWVp2 = -1.0/(2*(dsig)[np.newaxis,:])*(np.diff(OnLine.profile,axis=1)/OnLine.mean_dR/OnLine.profile[:,1:]-np.diff(OffLine.profile,axis=1)/OffLine.mean_dR/OffLine.profile[:,1:])


if save_as_nc:
    CombHi.write2nc(ncfilename)
    Molecular.write2nc(ncfilename)
    aer_beta_dlb.write2nc(ncfilename)
    beta_mol_sonde.write2nc(ncfilename)
    nWV.write2nc(ncfilename)
    OnLine.write2nc(ncfilename)
    OffLine.write2nc(ncfilename)
    if run_MLE and use_geo:
#        CamList,GmList,GcList,ProfileErrorMol,ProfileErrorComb
        beta_a_mle.write2nc(ncfilename)
        alpha_a_mle.write2nc(ncfilename)
        sLR_mle.write2nc(ncfilename)
        xvalid_mle.write2nc(ncfilename)
        fit_mol_mle.write2nc(ncfilename)
        fit_comb_mle.write2nc(ncfilename)


if plotAsDays:
    time_scale = 3600*24.0
else:
    time_scale = 3600.0

#plt.figure(figsize=(15,5)); 
#plt.pcolor(OnLine.time/3600,OnLine.range_array*1e-3, np.log10(1e9*OnLine.profile.T/OnLine.binwidth_ns/(dt*7e3)));
#plt.colorbar()
#plt.clim([3,8])
#plt.title('Online ' + OnLine.lidar + ' Count Rate [Hz]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)
#
#plt.figure(figsize=(15,5)); 
#plt.pcolor(OffLine.time/3600,OffLine.range_array*1e-3, np.log10(1e9*OffLine.profile.T/OffLine.binwidth_ns/(dt*7e3)));
#plt.colorbar()
#plt.clim([3,8])
#plt.title('Offline '+ OffLine.lidar + ' Count Rate [Hz]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)

lp.pcolor_profiles([nWV,aer_beta_dlb],climits=[[0,12],[-8.0,-4.0]],scale=['linear','log'],plotAsDays=plotAsDays)  # Standard
#lp.pcolor_profiles([nWV,aer_beta_dlb],climits=[[2,10],[-7.4,-5.5]],scale=['linear','log'],plotAsDays=plotAsDays)  # Aerosol Enhanced
#lp.pcolor_profiles([nWV,aer_beta_dlb],climits=[[0,12],[-8.0,-4.0]],scale=['linear','log'],plotAsDays=plotAsDays)  # Standard
#lp.pcolor_profiles([nWV,aer_beta_dlb],climits=[[0,12],[-7.4,-6.0]],scale=['linear','log'],plotAsDays=plotAsDays,ylimits=[0,4],tlimits=[10,17.75])
if save_figs:
    plt.savefig(figfilename+'_WaterVapor_AerosolBackscatter.png')
#plt.figure(figsize=(15,5)); 
#plt.pcolor(OffLine.time/3600,range_diff*1e-3, np.log10(nWV.T));
#plt.colorbar()
#plt.clim([22,25])
#plt.title('$n_{wv}$ '+ OffLine.lidar + ' [$m^{-3}$]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)




#sigWV = lp.WV_ExtinctionFromHITRAN(np.array([lp.c/OnLine.wavelength,lp.c/OffLine.wavelength]),Tsonde,Psonde,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]))

## note the operating wavelength of the lidar is 532 nm
#beta_m_sonde = sonde_scale*5.45*(550.0/780.24)**4*1e-32*Psonde/(Tsonde*lp.kB)






"""
Diff Geo Correction
"""

##timecal = np.array([1.5*3600,6*3600])
#MolCal = Molecular.copy();
#CombCal = CombHi.copy();
#
##MolCal.slice_time(timecal)
##CombCal.slice_time(timecal)
#
#MolCal.time_integrate();
#CombCal.time_integrate();
##
#plt.figure();
#plt.plot((MolCal.profile/CombCal.profile).T)
#plt.plot(np.max((MolCal.profile/CombCal.profile).T,axis=1),'k--',linewidth=2)
#plt.plot((MolCal.profile/CombCal.profile).flatten())
#
#FitProf = np.max((MolCal.profile/CombCal.profile).T,axis=1)
#FitProf2 = np.mean((MolCal.profile/CombCal.profile).T,axis=1)
#f1 = np.concatenate((np.arange(1,8),np.arange(42,61)));
##f1 = np.arange(1,100)
##f1 = np.array([6,7,43,44])
#
##pfit1 = np.polyfit(f1,np.log(FitProf[f1]),10)
##pfit1 = np.polyfit(f1,np.log(FitProf[f1]),2)
#
##np.interp(np.arange(8,43),f1,FitProf[f1]);
#finterp = scipy.interpolate.interp1d(f1,FitProf[f1],kind='cubic')
#finterpL = scipy.interpolate.interp1d(f1,FitProf[f1])
#Lweight = 0.7
#
#f2 = np.concatenate((np.arange(50,100),np.arange(130,200)));
#
#pfit1 = np.polyfit(f2,FitProf2[f2],4)
#
#xf = np.arange(f1[0],f1[-1])
#xf2 = np.arange(50,200)
#plt.figure()
#plt.plot((MolCal.profile/CombCal.profile).T)
#plt.plot(FitProf,'k--',linewidth=2)
#plt.plot(xf,(1-Lweight)*finterp(xf)+Lweight*finterpL(xf),'k-',linewidth=2)
#plt.plot(xf2,np.polyval(pfit1,xf2),'k-',linewidth=2)
#
#x0 = np.arange(np.size(FitProf))
#diff_geo_prof = np.zeros(np.size(FitProf))
#diff_geo_prof[0] = 1
#diff_geo_prof[f1[0]:f1[-1]] = (1-Lweight)*finterp(x0[f1[0]:f1[-1]])+Lweight*finterpL(x0[f1[0]:f1[-1]])
#diff_geo_prof[f2[0]:f1[-1]] = 0.5*((1-Lweight)*finterp(x0[f2[0]:f1[-1]])+Lweight*finterpL(x0[f2[0]:f1[-1]])) + 0.5*(np.polyval(pfit1,x0[f2[0]:f1[-1]]))
#diff_geo_prof[f1[-1]:] = np.polyval(pfit1,x0[f1[-1]:])
#
#plt.figure();
#plt.plot((MolCal.profile/CombCal.profile).T)
#plt.plot(diff_geo_prof,'k-',linewidth= 2)

#np.savez('diff_geo_DLB_20161212',diff_geo_prof=diff_geo_prof,Day=Day,Month=Month,Year=Year,HourLim=HourLim)

"""
Geo Overlap
"""
#
## 12/13/2016 - 10-12 UT
#
## 12/26/2016 - 20.4-1.2 UT
#
#plt.figure(); 
#plt.semilogx(Mol_Beta_Scale*MolInt.profile.flatten(),MolInt.range_array)
#plt.semilogx(beta_m_sonde,CombHi.range_array)
#plt.grid(b=True)
#
#plt.figure();
#plt.plot(beta_m_sonde/(Mol_Beta_Scale*MolInt.profile.flatten()))
#
### Set constant above 47th bin
##geo_prof = np.ones(np.size(MolInt.profile))
##geo_prof[0:47] = (beta_m_sonde/(Mol_Beta_Scale*MolInt.profile.flatten()))[np.newaxis,0:47]
##geo_prof = np.hstack((MolInt.range_array[:,np.newaxis],geo_prof[:,np.newaxis]))
#
### Run a linear fit above 65th bin
#geo_prof = beta_m_sonde/(Mol_Beta_Scale*MolInt.profile.flatten())
#xfit = np.arange(MolInt.profile[0,65:180].size)
#yfit = geo_prof[65:180]
#wfit = 1.0/np.sqrt(MolInt.profile_variance[0,65:180].flatten())
#pfit = np.polyfit(xfit,yfit,2,w=wfit)
#xprof = np.arange(MolInt.profile[0,65:].size)
#geo_prof[65:] = np.polyval(pfit,xprof)
#geo_prof = np.hstack((MolInt.range_array[:,np.newaxis],geo_prof[:,np.newaxis]))
#
#plt.figure(); 
#plt.plot(beta_m_sonde/(Mol_Beta_Scale*MolInt.profile.flatten()))
#plt.plot(geo_prof[:,1])
#
#np.savez('geo_DLB_20161227',geo_prof=geo_prof,Day=Days,Month=Months,Year=Years,HourLim=HourLim,Hours=Hours)

"""
RD Correction
"""

#import scipy as sp
#
#Mol2 = MolRaw.copy()
#Comb2 = CombRaw.copy()
#
#Mol2.mask_range('index',np.arange(1))
#Mol2.conv(tsmooth_wv/tres_wv,zsmooth_wv/zres,keep_mask=True)
#Mol2.slice_time(HourLim*3600)
#Mol2.bg_subtract(BGIndex)
#
#Comb2.nonlinear_correct(29.4e-9);
#Comb2.mask_range('index',np.arange(1))
#Comb2.conv(5.0*60.0/tres_wv/2,4.0/2,keep_mask=True)
#Comb2.slice_time(HourLim*3600)
#Comb2.bg_subtract(BGIndex)
#
#
#
## HSRL time slices
#if Comb2.time.size > Mol2.time.size:
#    Comb2.slice_time_index(time_lim=np.array([0,Mol2.time.size]))
#elif Comb2.time.size < Mol2.time.size:
#    Mol2.slice_time_index(time_lim=np.array([0,Comb2.time.size-1]))
#
#
#Mol2.range_correct();
#Mol2.slice_range(range_lim=[0,MaxAlt])
##Mol2.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
#Mol2.range_resample(delta_R=zres,update=True)
#
#Comb2.range_correct()
#Comb2.slice_range(range_lim=[0,MaxAlt])
##Comb2.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
#Comb2.range_resample(delta_R=zres,update=True)
#
#Mol2.gain_scale(MolGain)
#
#BSRwv = Comb2.profile/Mol2.profile
#
#dBSRwv = np.diff(BSRwv,axis=1)/Mol2.mean_dR
#BSRwv_int = 0.5*(BSRwv[:,:-1]+BSRwv[:,1:])  # differential range interpolated BSR
#
#varBSRwv = 1/Mol2.profile**2*(Comb2.profile_variance+BSRwv**2*Mol2.profile_variance)
#var_dBSRwv = (varBSRwv[:,:-1]+varBSRwv[:,1:])/Mol2.mean_dR**2
#var_BSRwv_int = (varBSRwv[:,:-1]+varBSRwv[:,1:])/4.0
#
#
#WVcorr = (dBSRwv/BSRwv_int**2)  # correction term for wv extinction
#var_WVcorr = 1/BSRwv_int**2*(var_dBSRwv+WVcorr**2*var_BSRwv_int)
#
#alpha_wv = np.diff(np.log(OnLine.profile/OffLine.profile),axis=1)/OnLine.mean_dR
#
#WVcorr_snr = np.abs(WVcorr/np.sqrt(var_WVcorr))
#WVcorr_snr = np.convolve(WVcorr_snr[30,:],np.ones(4),'same')
#WVcorr2 = WVcorr.copy()
#WVcorr2[np.nonzero(WVcorr_snr < 3)] = 0
#WVcorr3 = WVcorr*sp.special.erf(0.25*WVcorr_snr)
#plt.figure(); 
#plt.plot(WVcorr[200,:]); 
#plt.plot(np.sqrt(var_WVcorr[200,:])); 
#plt.plot(-np.sqrt(var_WVcorr[200,:]))
#plt.plot(WVcorr2[200,:])
#plt.plot(WVcorr3[200,:])
#plt.plot(alpha_wv[200,:],'k--')
#
#nWVrd = wv.WaterVapor_Simple_RD_Test(OnLine,OffLine,Psonde,Tsonde,BSRwv)
#nWVrd.conv(0.3,2.0)
#nWVrd.mask_range('<=',WV_Min_Alt)
#
#nWVrd.cat_range(SurfaceHumid)
#
#lp.pcolor_profiles([nWVrd,aer_beta_dlb],climits=[[0,12],[-8.0,-4.0]],scale=['linear','log'],plotAsDays=plotAsDays)
#
#
## compute frequencies from wavelength terms
#dnu = np.linspace(-70e9,70e9,4000)  # array of spectrum relative to transmitted frequency
#inuL = np.argmin(np.abs(dnu))  # index to the laser frequency in the spectrum
#nuOff = lp.c/OffLine.wavelength  # Offline laser frequency
#nuOn = lp.c/OnLine.wavelength   # Online laser frequency
#
##sigWV0 = lp.WV_ExtinctionFromHITRAN(np.array([nuOn,nuOff]),Tsonde,Psonde,nuLim=np.array([lp.c/831e-9,lp.c/825e-9]),freqnorm=True)
#sigWV0 = lp.WV_ExtinctionFromHITRAN(np.array([nuOn,nuOff]),Tsonde,Psonde,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True)
#
#dsig = sigWV0[:,0]-sigWV0[:,1]
#
#
##dnu0 = np.linspace(lp.c/827.0e-9,lp.c/829.5e-9,1000)
#dnu0 = np.arange(-3,3,0.05)*1e9
#nuOn = lp.c/lambda_on[200]
#nuOff = lp.c/lambda_off[200]
#dnu = np.abs(np.mean(np.diff(dnu0)))
#nu0on = nuOn+dnu0
#nu0off = nuOff+dnu0
#
#Etalon_angle = 0.00*np.pi/180
#
#Filter = FO.FP_Etalon(1.0932e9,43.5e9,nuOff,efficiency=0.95,InWavelength=False)
#Toffline = Filter.spectrum(dnu0+nuOff,InWavelength=False,aoi=Etalon_angle,transmit=True)
#Tonline = Filter.spectrum(dnu0+nuOn,InWavelength=False,aoi=Etalon_angle,transmit=True)
#
#inu0 = np.argmin(np.abs(dnu0))
#sigWV0on = lp.WV_ExtinctionFromHITRAN(nu0on,Tsonde,Psonde,nuLim=np.array([lp.c/835e-9,lp.c/825e-9]),freqnorm=True)
#sigWV0off = lp.WV_ExtinctionFromHITRAN(nu0off,Tsonde,Psonde,nuLim=np.array([lp.c/835e-9,lp.c/825e-9]),freqnorm=True)
#MolSpec = lp.RB_Spectrum(Tsonde,Psonde,lambda_on[200],nu=dnu0)
#
#
##nWVprof = nWV.profile[200,1:]*lp.N_A/lp.mH2O
#nWVprof = (-0.5*alpha_wv[150,:])/dsig[1:]
#nWVprof = np.convolve(nWVprof,np.ones(6),'same')/6.0
#nWVprof0 = nWVprof.copy()
#plt.figure();
#plt.plot(nWVprof)
#for ai in range(100):
#    Tmol1on = np.sum(np.exp(-2*np.cumsum(sigWV0on[1:,:]*nWVprof[:,np.newaxis],axis=0)*Mol2.mean_dR)*MolSpec[:,1:].T,axis=1)
#    Tmol2on = np.sum(Tonline[np.newaxis,:]*np.exp(-2*np.cumsum(sigWV0on[1:,:]*nWVprof[:,np.newaxis],axis=0)*Mol2.mean_dR)*MolSpec[:,1:].T,axis=1)
#    Tmol0on = Tonline[inu0]*np.exp(-2*np.cumsum(sigWV0on[1:,inu0]*nWVprof,axis=0)*Mol2.mean_dR)
#    
#    Tmol1off = np.sum(np.exp(-2*np.cumsum(sigWV0off[1:,:]*nWVprof[:,np.newaxis],axis=0)*Mol2.mean_dR)*MolSpec[:,1:].T,axis=1)
#    Tmol2off = np.sum(Toffline[np.newaxis,:]*np.exp(-2*np.cumsum(sigWV0off[1:,:]*nWVprof[:,np.newaxis],axis=0)*Mol2.mean_dR)*MolSpec[:,1:].T,axis=1)
#    Tmol0off = Toffline[inu0]*np.exp(-2*np.cumsum(sigWV0off[1:,inu0]*nWVprof,axis=0)*Mol2.mean_dR)
#    
#    eta_on = Tmol2on/Tmol0on
#    eta_off = Tmol2off/Tmol0off
#    
#    WVcorr_eta = np.concatenate((np.array([0]),0.5*np.diff(np.log((BSRwv_int[150,:]+(eta_on-1.0))/(BSRwv_int[150,:]+(eta_off-1.0))))/Mol2.mean_dR))
##    WVcorr_eta = 0.5*np.diff(np.log((BSRwv_int[150,:]+(eta_on-1.0))/(BSRwv_int[150,:]+(eta_off-1.0))))
#    
#    WVcorr_eta =  np.convolve(WVcorr_eta,np.ones(6),'same')/6.0   
#    
##    adj_area = np.nonzero(np.logical_and(np.abs(dBSRwv[150,:]/BSRwv_int[150,:])> 0.004,Mol2.range_array[1:]>1e3))[0]
#    adj_area = np.nonzero(np.logical_and(BSRwv_int[150,:]> 2.0,Mol2.range_array[1:]>1e3))[0]
##    adj_area = np.nonzero(np.abs(dBSRwv[150,:]/BSRwv_int[150,:])> 0.004)[0] 
##    adj_area = np.nonzero(Mol2.range_array[1:]>1e3)[0]
##    alpha_wv[150,adj_area] = alpha_wv[150,adj_area]-2*WVcorr_eta[adj_area-1]
#    nWVprof[adj_area] = nWVprof0[adj_area]-1.0*WVcorr_eta[adj_area-2]/dsig[adj_area]
##    nWVprof = nWVprof+WVcorr_eta/dsig[1:]
##    nWVprof = (-0.5*alpha_wv[150,:]+WVcorr_eta)/dsig[1:]
#
#plt.plot(nWVprof)
##WVcorr_eta = 0.5*np.diff(np.log((1+1.0/BSRwv*(eta_on-1.0))/(1+1.0/BSRwv*(eta_off-1.0))))/Mol2.mean_dR
#
##plt.figure(); 
##plt.plot(20*WVcorr_eta[200,:]); 
##plt.plot(alpha_wv[200,:])
##
#plt.figure(); 
#plt.semilogy(dnu0*1e-9,Tonline,'b',dnu0*1e-9,MolSpec[:,0]/MolSpec[inu0,0],'g',dnu0*1e-9,1e26*sigWV0on[0,:],'r')
#
#plt.semilogy((dnu0+nuOff-nuOn)*1e-9,Toffline,'b',(dnu0+nuOff-nuOn)*1e-9,MolSpec[:,0]/MolSpec[inu0,0],'g',(dnu0+nuOff-nuOn)*1e-9,1e26*sigWV0off[0,:],'r')
#
#plt.figure(); 
#plt.semilogy(dnu0*1e-9,Toffline,dnu0*1e-9,MolSpec[:,0]/MolSpec[inu0,0],dnu0*1e-9,1e26*sigWV0off[0,:])