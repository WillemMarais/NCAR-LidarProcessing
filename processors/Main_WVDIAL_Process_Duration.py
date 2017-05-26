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


Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,5,24,startHr=0,duration=5)


plotAsDays = True
getMLE_extinction = False
run_MLE = False
runKlett = False

save_as_nc = False
save_figs = False

run_geo_cal = False

model_atm = True

nctag = ''

use_diff_geo = False   # no diff geo correction after April ???
use_geo = False

use_mask = True
SNRmask = 2.0  #SNR level used to decide what data points we keep in the final data product
countLim = 2.0

MaxAlt = 6e3 #12e3
WV_Min_Alt = 350  # mask data below this altitude

KlettAlt = 14e3  # altitude where Klett inversion starts

# set bin sizes
tres_hsrl = 1.0*60.0  # HSRL bin time resolution in seconds (2 sec typical base)
tres_wv = 1.0*60.0    # HSRL bin time resolution in seconds (2 sec typical base)
zres = 37.5  # bin range resolution in m (37.5 m typical base)

# parameters for WV Channels smoothing
tsmooth_wv = 5*60 # convolution kernal time (HW sigma) in seconds
zsmooth_wv = 1 #150  # convolution kernal range (HW sigma) in meters
zsmooth2_wv = np.sqrt(150**2+75**2)  # 75 # second range smoothing conducted on the actual WV retrieval


"""
Paths
"""
# path to data
#basepath = '/scr/eldora1/MSU_h2o_data/'
basepath = '/scr/eldora1/h2o_data/'

# path for saving data
save_data_path = '/h/eol/mhayman/DIAL/Processed_Data/'
save_fig_path = '/h/eol/mhayman/DIAL/Processed_Data/Plots/'

# path to sonde data
sonde_path = '/scr/eldora1/HSRL_data/'

# path to calibration files
cal_path = '/h/eol/mhayman/PythonScripts/NCAR-LidarProcessing/calibrations/'
#cal_file = cal_path+'dlb_calvals_msu.json'
cal_file = cal_path+'dlb_calvals_ncar0.json'


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

#if use_diff_geo:
#    cal_value = lp.get_calval(ProcStart,cal_jdata,'Molecular Gain',cond=['diff_geo','!=','none'],returnlist=['value','diff_geo'])
#    diff_geo_file = cal_path+cal_value[1]
#else:
#    cal_value = lp.get_calval(ProcStart,cal_jdata,'Molecular Gain',cond=['diff_geo','=','none'])
#MolGain = cal_value[0]

if use_geo:
    cal_value = lp.get_calval(ProcStart,cal_jdata,'Geo File Record',returnlist=['filename'])
    geo_file = cal_path+cal_value[0]

dR = BinWidth*lp.c/2  # profile range resolution (500e-9*c/2)-typical became 100e-9*c/2 on 2/22/2017
Roffset = ((1.25+0.5)-0.5/2)*lp.c*LaserPulseWidth/2  # offset in range

zres = np.max([np.round(zres/dR),1.0])*dR  #only allow z resolution to be integer increments of the MCS range

BGIndex = -50; # negative number provides an index from the end of the array





if save_as_nc or save_figs:
    ncfilename0 = lp.create_ncfilename('NCAR0_WVDIAL_DLBHSRL',Years,Months,Days,Hours,tag=nctag)
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
MasterTimeWV = np.arange(Hours[0,0]*3600,Days.size*24*3600-(24-Hours[-1,-1])*3600,tres_wv)

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

Backscatter = OffLineRaw.copy()
Backscatter.mask_range('index',np.arange(1))
#Backscatter.mask_range('<=',300)
Backscatter.slice_time(HourLim*3600)
#OffLine.nonlinear_correct(30e-9);
Backscatter.bg_subtract(BGIndex)

#lp.plotprofiles([OnLine,OffLine,OnLineRaw,OffLineRaw])


#####  NEED TO CORRECT TIME SLICES BASED ON ALL 4 PROFILES

# WV-DIAL time slices
if OffLine.time.size > OnLine.time.size:
    OffLine.slice_time_index(time_lim=np.array([0,OnLine.time.size]))
    Backscatter.slice_time_index(time_lim=np.array([0,OnLine.time.size]))
elif OffLine.time.size < OnLine.time.size:
    OnLine.slice_time_index(time_lim=np.array([0,OffLine.time.size]))

# mask based on raw counts - remove points where there are < 2 counts
if use_mask:
    NanMask_wv = np.logical_or(OnLine.profile < 2.0,OffLine.profile < 2.0)
    OnLine.profile = np.ma.array(OnLine.profile,mask=NanMask_wv)
    OffLine.profile = np.ma.array(OffLine.profile,mask=NanMask_wv)
    Backscatter.mask(NanMask_wv)





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

Backscatter.slice_range(range_lim=[0,MaxAlt])
Backscatter.range_resample(delta_R=zres,update=True)
Backscatter.slice_range_index(range_lim=[1,1e6])
Backscatter.range_correct()


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
lp.plotprofiles([OffLine,OnLine])

OnInt = OnLine.copy();
OnInt.time_integrate();
OffInt = OffLine.copy();
OffInt.time_integrate();


isonde = np.argmin(pres.time-pres.time/2.0)
Psonde = pres.profile[isonde,:]
Tsonde = temp.profile[isonde,:]

#nWV = wv.WaterVapor_2D(OnLine,OffLine,lambda_on,lambda_off,pres,temp)
nWV = wv.WaterVapor_Simple(OnLine,OffLine,Psonde,Tsonde)  # solve with 1D Pres and Temp Profile
#nWV = wv.WaterVapor_2D(OnLine,OffLine,lambda_on,lambda_off,pres.profile,temp.profile)  # solve with 2D Pres and Temp profiles
nWV.conv(0.0,zsmooth2_wv/zres)
nWV.mask_range('<=',WV_Min_Alt)


OffLine2 = OffLine.copy()
OffLine2.conv(0,zsmooth2_wv/zres)
OffDiff = np.diff(OffLine2.profile,axis=1)/OffLine2.mean_dR
Off_int = 0.5*(OffLine2.profile[:,1:]+OffLine2.profile[:,:-1])
BSR_mask = np.zeros(nWV.profile.shape)
BSR_mask[np.nonzero(np.abs(OffDiff) > 2)] = 1

if use_mask:
    wv_snr_mask = np.zeros(nWV.profile.shape)
    wv_snr_mask[np.nonzero(nWV.SNR() < SNRmask)] = 1
    nWV.mask(wv_snr_mask)
    nWV.mask(BSR_mask)
    nWV.mask(NanMask_wv)

nWV.cat_range(SurfaceHumid)


if save_as_nc:
    beta_mol_sonde.write2nc(ncfilename)
    nWV.write2nc(ncfilename)
    OnLine.write2nc(ncfilename)
    OffLine.write2nc(ncfilename)



if plotAsDays:
    time_scale = 3600*24.0
else:
    time_scale = 3600.0


#lp.pcolor_profiles([nWV,aer_beta_dlb],climits=[[0,12],[-8.0,-4.0]],scale=['linear','log'],plotAsDays=plotAsDays)  # Standard
lp.pcolor_profiles([nWV,Backscatter],climits=[[0,12],[8,12]],scale=['linear','log'],plotAsDays=plotAsDays)  # Aerosol Enhanced
#lp.pcolor_profiles([nWV,aer_beta_dlb],climits=[[0,12],[-8.0,-4.0]],scale=['linear','log'],plotAsDays=plotAsDays)  # Standard
#lp.pcolor_profiles([nWV,aer_beta_dlb],climits=[[0,12],[-7.4,-6.0]],scale=['linear','log'],plotAsDays=plotAsDays,ylimits=[0,4],tlimits=[10,17.75])
if save_figs:
    plt.savefig(figfilename+'_WaterVapor.png')

