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

cal_path = '/h/eol/mhayman/PythonScripts/NCAR-LidarProcessing/calibrations/'
cal_file = cal_path+'dlb_calvals_msu.json'

Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,5,12,startHr=0,duration=24)


plotAsDays = False
getMLE_extinction = False
run_MLE = False
runKlett = False

save_as_nc = False
save_figs = False

run_geo_cal = False

nctag = ''

use_diff_geo = False   # no diff geo correction after April ???
use_geo = True

use_mask = False
SNRmask = 0.0  #SNR level used to decide what data points we keep in the final data product
countLim = 2.0

MaxAlt = 6e3 #12e3
WV_Min_Alt = 350  # mask data below this altitude

KlettAlt = 14e3  # altitude where Klett inversion starts

# set bin sizes
tres_hsrl = 0.5*60.0  # HSRL bin time resolution in seconds (2 sec typical base)
tres_wv = 0.5*60.0    # HSRL bin time resolution in seconds (2 sec typical base)
zres = 75  # bin range resolution in m (37.5 m typical base)

# parameters for WV Channels smoothing
tsmooth_wv = 5*60 # convolution kernal time (HW sigma) in seconds
zsmooth_wv = 150  # convolution kernal range (HW sigma) in meters


"""
Begin Processing
"""

ProcStart = datetime.datetime(Years[0],Months[0],Days[0],Hours[0][0])

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

basepath = '/scr/eldora1/MSU_h2o_data/'

FieldLabel_WV = 'FF'
ON_FileBase = 'Online_Raw_Data.dat'
OFF_FileBase = 'Offline_Raw_Data.dat'

FieldLabel_HSRL = 'NF'
MolFileBase = 'Online_Raw_Data.dat'
CombFileBase = 'Offline_Raw_Data.dat'

save_data_path = '/h/eol/mhayman/DIAL/Processed_Data/'
save_fig_path = '/h/eol/mhayman/DIAL/Processed_Data/Plots/'
sonde_path = '/scr/eldora1/HSRL_data/'


if save_as_nc or save_figs:
    ncfilename0 = lp.create_ncfilename('MSU_WVDIAL_DLBHSRL',Years,Months,Days,Hours,tag=nctag)
    ncfilename = save_data_path+ncfilename0
    figfilename = save_fig_path + ncfilename0[:-3] + nctag

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


[Molecular,CombHi],lambda_hsrl,HourLim = wv.Load_DLB_Data(basepath,FieldLabel_HSRL,[MolFileBase,CombFileBase],MasterTimeHSRL,Years,Months,Days,Hours,MCSbins,lidar='DLB-HSRL',dt=dt,Roffset=Roffset,BinWidth=BinWidth)
[OnLine,OffLine],[lambda_on,lambda_off],HourLim = wv.Load_DLB_Data(basepath,FieldLabel_WV,[ON_FileBase,OFF_FileBase],MasterTimeWV,Years,Months,Days,Hours,MCSbins,lidar='WV-DIAL',dt=dt,Roffset=Roffset,BinWidth=BinWidth)

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
OffLine.conv(5.0*60.0/tres_wv/2,4.0/2,keep_mask=True)
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

# Correct Molecular Cross Talk
if Cam > 0:
    lp.FilterCrossTalkCorrect(Molecular,CombHi,Cam,smart=True)

# add interpolated temperature and pressure
beta_mol_sonde,sonde_time,sonde_index_prof,temp,pres,sonde_index = lp.get_beta_m_sonde(Molecular,Years,Months,Days,sonde_path,interp=True,returnTP=True)

# convert pressure from Pa to atm.
pres.gain_scale(9.86923e-6)  
pres.descript = 'Sonde Measured Pressure in atm'
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
nWV.conv(0.3,2.0)
nWV.mask_range('<=',WV_Min_Alt)


dnu = np.linspace(-7e9,7e9,400)
inuL = np.argmin(np.abs(dnu))
MolSpec = lp.RB_Spectrum(Tsonde,Psonde,OffLine.wavelength,nu=dnu)
nuOff = lp.c/OffLine.wavelength
nuOn = lp.c/OnLine.wavelength



Filter = FO.FP_Etalon(1.0932e9,43.5e9,nuOff,efficiency=0.95,InWavelength=False)
Toffline = Filter.spectrum(dnu+nuOff,InWavelength=False,aoi=0.0,transmit=True)
Tonline = Filter.spectrum(dnu+nuOn,InWavelength=False,aoi=0.0,transmit=True)

Toffline2 = Filter.spectrum(lp.c/lambda_on,InWavelength=False,aoi=0.0,transmit=True)
Tonline2 = Filter.spectrum(lp.c/lambda_off,InWavelength=False,aoi=0.0,transmit=True)

plt.figure(); 
plt.plot(dnu+nuOn,Tonline); 
plt.plot(dnu+nuOff,Toffline);
plt.plot(lp.c/lambda_on,Tonline2,'bx',label='Online'); 
plt.plot(lp.c/lambda_off,Toffline2,'gx',label='Offline');

#plt.plot(dnu[inuL]+nuOn,Tonline[inuL],'bx',label='Online')
#plt.plot(dnu[inuL]+nuOff,Toffline[inuL],'gx',label='Offline')
plt.grid(b=True)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Transmission')

nuWV = np.linspace(lp.c/828.5e-9,lp.c/828e-9,500)
sigWV = lp.WV_ExtinctionFromHITRAN(nuWV,Tsonde,Psonde) #,nuLim=np.array([lp.c/835e-9,lp.c/825e-9]))  
ind_on = np.argmin(np.abs(lp.c/OnLine.wavelength-nuWV))
ind_off = np.argmin(np.abs(lp.c/OffLine.wavelength-nuWV))

sigOn = sigWV[:,ind_on]
sigOff = sigWV[:,ind_off]

sigF = scipy.interpolate.interp1d(nuWV,sigWV)
sigWVOn = sigF(dnu+nuOn).T
sigWVOff = sigF(dnu+nuOff).T

#sigWVOn = lp.WV_ExtinctionFromHITRAN(nuOn+dnu,Tsonde,Psonde) 
#sigWVOff = lp.WV_ExtinctionFromHITRAN(nuOff+dnu,Tsonde,Psonde)

sigOn = sigWVOn[inuL,:]
sigOff = sigWVOff[inuL,:]



range_diff = OnLine.range_array[1:]-OnLine.mean_dR/2.0
dsig = np.interp(range_diff,OnLine.range_array,sigOn-sigOff)
nWVp = -1.0/(2*(dsig)[np.newaxis,:])*np.diff(np.log(OnLine.profile/OffLine.profile),axis=1)/OnLine.mean_dR
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

plt.figure(figsize=(15,5)); 
plt.pcolor(OnLine.time/3600,OnLine.range_array*1e-3, np.log10(1e9*OnLine.profile.T/OnLine.binwidth_ns/(dt*7e3)));
plt.colorbar()
plt.clim([3,8])
plt.title('Online ' + OnLine.lidar + ' Count Rate [Hz]')
plt.ylabel('Altitude [km]')
plt.xlabel('Time [UTC]')
plt.xlim(HourLim)

plt.figure(figsize=(15,5)); 
plt.pcolor(OffLine.time/3600,OffLine.range_array*1e-3, np.log10(1e9*OffLine.profile.T/OffLine.binwidth_ns/(dt*7e3)));
plt.colorbar()
plt.clim([3,8])
plt.title('Offline '+ OffLine.lidar + ' Count Rate [Hz]')
plt.ylabel('Altitude [km]')
plt.xlabel('Time [UTC]')
plt.xlim(HourLim)

lp.pcolor_profiles([nWV,aer_beta_dlb],climits=[[0,12],[-8.0,-4.0]],scale=['linear','log'],plotAsDays=plotAsDays)  # 
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
SVD Denoising
"""
#
#
#
#NpcaM = 3
#NpcaC = 18
#
#FitMol0 = MolRaw.profile
#MeanFitMol0 = np.mean(FitMol0,axis=0)
#
#[u,s,v] = np.linalg.svd((FitMol0-MeanFitMol0).T);
#
#weights = np.dot(u[:,:NpcaM].T,(FitMol0-MeanFitMol0).T).T
#
#Molsvd = np.dot(u[:,:NpcaM],weights.T).T
#
##FitComb0 = CombRaw.profile
##MeanFitComb0 = np.mean(FitComb0,axis=0)
##
##[uC,sC,vC] = np.linalg.svd((FitComb0-MeanFitComb0).T);
##
##weightsC = np.dot(u[:,:NpcaC].T,(FitComb0-MeanFitComb0).T).T
##
##Combsvd = np.dot(uC[:,:NpcaC],weightsC.T).T
#
#### Optimization ###
#
#FitMol = MolRaw.profile
#
#FitComb = CombRaw.profile
#
#stopIndex = MolRaw.range_array.size
#
#MolEst = Molsvd+MeanFitMol0[np.newaxis,:]
##CombEst = Combsvd+MeanFitComb0[np.newaxis,:]
#
#lamMol = np.array([0.1,1e-3])
##lamCom = np.array([0.01,10e-3])
#
#FitProfMol = lambda x: ProfEstimateSVD2D(x,FitMol,MolEst,lamMol)
#FitProfMolDeriv = lambda x: ProfEstimateSVD2D_prime(x,FitMol,MolEst,lamMol)
#
##bndsP = np.zeros((FitMol.size,2))
##bndsP[:,1] = np.max(FitMol)*20
#
#x0 = np.random.rand(FitMol.size)-0.5
#wMol = scipy.optimize.fmin_slsqp(FitProfMol,x0,fprime=FitProfMolDeriv,iter=200)
#
##FitProfComb = lambda x: ProfEstimateSVD2D(x,FitComb,CombEst,lamCom)
##FitProfCombDeriv = lambda x: ProfEstimateSVD2D_prime(x,FitComb,CombEst,lamCom)
##
###bndsP = np.zeros((FitComb.size,2))
###bndsP[:,1] = np.max(FitComb)*20
##
##x0 = np.random.rand(FitComb.size)-0.5
##wCom = scipy.optimize.fmin_slsqp(FitProfComb,x0,fprime=FitProfCombDeriv,iter=500)
##
##(wCom.reshape(FitComb.shape)+CombEst)
#
#
#####  Process SVD Profiles ####
#MolSVD = lp.LidarProfile(wMol.reshape(FitMol.shape)+MolEst,MolRaw.time,label='SVD Denoised Molecular Backscatter Channel',descript = 'Unpolarization\nMolecular Backscatter Returns',bin0=-Roffset/dR,lidar='DLB-HSRL')
##CombSVD = lp.LidarProfile(wCom.reshape(FitComb.shape)+CombEst,CombRaw.time,label='SVD Denoised Combined Backscatter Channel',descript = 'Unpolarization\nCombined Backscatter Returns',bin0=-Roffset/dR,lidar='DLB-HSRL')
#CombSVD = CombRaw.copy()
#
#MolSVD.nonlinear_correct(38e-9);
#MolSVD.bg_subtract(BGIndex)
#
#CombSVD.nonlinear_correct(29.4e-9);
#CombSVD.bg_subtract(BGIndex)
#
#
#if CombSVD.time.size > Molecular.time.size:
#    CombSVD.slice_time_index(time_lim=np.array([0,MolSVD.time.size]))
#elif CombSVD.time.size < MolSVD.time.size:
#    MolSVD.slice_time_index(time_lim=np.array([0,CombSVD.time.size]))
#
##MolSVD.range_correct();
#MolSVD.slice_range(range_lim=[0,MaxAlt])
#MolSVD.range_resample(delta_R=zres*dR,update=True)
##Molecular.conv(2.0,1.0)  # regrid by convolution
#
##CombHi.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_diff_geo:
#    CombSVD.diff_geo_overlap_correct(diff_geo_corr,geo_reference='mol')
##if use_geo:
##    CombHi.geo_overlap_correct(geo_corr)
##CombSVD.range_correct()
#CombSVD.slice_range(range_lim=[0,MaxAlt])
#CombSVD.range_resample(delta_R=zres*dR,update=True)
##CombHi.conv(2.0,1.0)  # regrid by convolution
#
#
#MolSVD.gain_scale(MolGain)
#
## Correct Molecular Cross Talk
#if Cam > 0:
#    lp.FilterCrossTalkCorrect(MolSVD,CombSVD,Cam,smart=True)
##    Molecular.profile = 1.0/(1-Cam)*(Molecular.profile-CombHi.profile*Cam);
#
##lp.plotprofiles([CombSVD,MolSVD])
#
#
#
### note the operating wavelength of the lidar is 532 nm
##beta_m_sonde = sonde_scale*5.45*(550.0/780.24)**4*1e-32*Psonde/(Tsonde*kB)
#
##plt.figure(); plt.semilogx(beta_m_sonde/np.nanmean(Molecular.profile,axis=0),Molecular.range_array)
#if sonde_scale == 1.0:
#    Mol_Beta_Scale = 1.36*0.925e-6*2.49e-11*MolSVD.mean_dt/(MolSVD.time[-1]-MolSVD.time[0])  # conversion from profile counts to backscatter cross section
#else:
#    Mol_Beta_Scale = 1.0/sonde_scale    
#
#BSR = (CombSVD.profile)/MolSVD.profile
#
#beta_bs = BSR*beta_m_sonde[np.newaxis,:]  # total backscatter including molecules
##aer_beta_bs = (BSR-1)*beta_m_sonde[np.newaxis,:]    # only aerosol backscatter
##aer_beta_bs[np.nonzero(aer_beta_bs <= 0)] = 1e-10;
#
#aer_beta_dlb_svd = lp.Calc_AerosolBackscatter(MolSVD,CombSVD,Temp=Tsonde,Pres=Psonde,beta_sonde_scale=sonde_scale)
#
#plt.figure(figsize=(15,5)); 
#plt.pcolor(aer_beta_dlb_svd.time/time_scale,aer_beta_dlb_svd.range_array*1e-3, np.log10(aer_beta_dlb_svd.profile.T));
#plt.colorbar()
#plt.clim([-9,-3])
#plt.title(DateLabel + ', ' + aer_beta_dlb_svd.lidar + ' Aerosol Backscatter Coefficient (SVD) [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#    
#plt.show()

