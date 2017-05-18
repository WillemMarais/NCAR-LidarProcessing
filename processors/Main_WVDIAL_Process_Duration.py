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
from scipy.io import netcdf
import LidarProfileFunctions as lp
import scipy.interpolate

import FourierOpticsLib as FO

import datetime

import glob


#filePath = '/scr/eldora1/MSU_h2o_data/2016/161209NF/18/'
#fileName1 = 'Online_Raw_Data.dat'  # molecular
#fileName2 = 'Offline_Raw_Data.dat'  # combined

#Day = 4
#Month = 1
#Year = 2017
#HourLim = np.array([0,24])  # Limits on the processing time
#
#
# Set the process chunk in year, month, day and duration
# function format:
# generate_WVDIAL_day_list(startYr,startMo,startDay,startHr=0,duration=0.0,stopYr=0,stopMo=0,stopDay=0,stopHr=24)
# duration is set in hours
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2016,12,28,startHr=6.0,duration=1.0) #,stopYr=0,stopMo=0,stopDay=0,stopHr=24):
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2016,12,28,startHr=13.5,stopYr=2017,stopMo=1,stopDay=2)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2016,12,27,startHr=19.85,stopHr=19.95)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,19,startHr=22.0,duration=1.0)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,21,startHr=19.3,duration=1.5)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,17,startHr=10.0,duration=1.0)

#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,27,startHr=13.75,duration=0.1)
Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,20,startHr=5,duration=4.0)
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,2)

#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,12,stopDay=19)

plotAsDays = False
getMLE_extinction = False
runKlett = False

ProcStart = datetime.date(Years[0],Months[0],Days[0])

DateLabel = ProcStart.strftime("%A %B %d, %Y")

MaxAlt = 12e3

tres = 5*60.0  # resolution in time points (2 sec)
zres = 1.0  # resolution in altitude points (75 m)

use_diff_geo = False
use_geo = False

use_mask = False
SNRmask = 0.0  #SNR level used to decide what data points we keep in the final data product
countLim = 2.0

#kB = 1.3806504e-23;
#c = 3e8

MCSbins = 280  # number of bins in a range resolved profile
dR = 500e-9*lp.c/2  # profile range resolution
dt = 2  # profile accumulation time
Roffset = ((1.25+0.5)-0.5/2)*150  # offset in range

BGIndex = -50; # negative number provides an index from the end of the array
#Cam = 0.000 # Cross talk of aerosols into the molecular channel - 0.005 on Dec 21 2016 after 18.5UTC

basepath = '/scr/eldora1/h2o_data/'
FieldLabel = 'FF'
ON_FileBase = 'Online_Raw_Data.dat'
OFF_FileBase = 'Offline_Raw_Data.dat'

# define time grid for lidar signal processing to occur
MasterTime = np.arange(Hours[0,0]*3600,Days.size*24*3600-(24-Hours[-1,-1])*3600,tres)


firstFile = True

for dayindex in range(Years.size):

    if Days[dayindex] < 10:
        DayStr = '0' + str(Days[dayindex])
    else:
        DayStr = str(Days[dayindex])
        
    if Months[dayindex] < 10:
        MonthStr = '0' + str(Months[dayindex])
    else:
        MonthStr = str(Months[dayindex])
    
    YearStr = str(Years[dayindex])
    HourLim = Hours[:,dayindex]
    
    # calculate the time offset due to being a different day than we started with
    if firstFile:
        deltat_0 = 0;
    else:
        deltat_0_date = datetime.date(Years[dayindex],Months[dayindex],Days[dayindex])-datetime.date(Years[0],Months[0],Days[0])
        deltat_0 = deltat_0_date.days
        
    FilePath0 = basepath + YearStr + '/' + YearStr[-2:] + MonthStr + DayStr + FieldLabel
    
    SubDirs = glob.glob(FilePath0+'/*/')
    
    for idir in range(len(SubDirs)):
        Hour = np.double(SubDirs[idir][-3:-1])
        if Hour >= np.floor(HourLim[0]) and Hour <= HourLim[1]:
            loadfile_on = SubDirs[idir]+ON_FileBase
            loadfile_off = SubDirs[idir]+OFF_FileBase
            Hour = np.double(SubDirs[idir][-3:-1])
            
            #### LOAD NETCDF DATA ####
            on_data,on_vars = lp.read_WVDIAL_binary(loadfile_on,MCSbins)
            off_data,off_vars = lp.read_WVDIAL_binary(loadfile_off,MCSbins)
                   
            
            timeDataOn =3600*24*(np.remainder(on_vars[0,:],1)+deltat_0)
            timeDataOff =3600*24*(np.remainder(off_vars[0,:],1)+deltat_0)
            
            wavelen_on0 = on_vars[1,:]*1e-9      
            wavelen_off0 = off_vars[1,:]*1e-9 
            
            itimeBad = np.nonzero(np.diff(timeDataOn)<0)[0]        
            if itimeBad.size > 0:
                timeDataOn[itimeBad+1] = timeDataOn[itimeBad]+dt
                
            itimeBad = np.nonzero(np.diff(timeDataOff)<0)[0]        
            if itimeBad.size > 0:
                timeDataOff[itimeBad+1] = timeDataOff[itimeBad]+dt
            
    #        print timeDataOn.size
    #        print timeDataOff.size        
            
            # load profile data
            if firstFile:
    #            timeMaster = np.arange(np.floor(np.min((timeDataOn[0],timeDataOff[0]))),np.floor(np.min((timeDataOn[0],timeDataOff[0]))),tres)
                
                OnLine = lp.LidarProfile(on_data.T,timeDataOn,label='Online Backscatter Channel',descript = 'Unpolarization\nOnline WV-DIAL Backscatter Returns',bin0=-Roffset/dR,lidar='WV-DIAL')
                OnLine.wavelength = wavelen_on0[0]
#                RemMol = Molecular.time_resample(delta_t=tres,t0=HourLim[0]*3600,update=True,remainder=True)
                RemOn = OnLine.time_resample(tedges=MasterTime,update=True,remainder=True)
                
                OffLine = lp.LidarProfile(off_data.T,timeDataOff,label='Offline Backscatter Channel',descript = 'Unpolarization\nOffline WV-DIAL Backscatter Returns',bin0=-Roffset/dR,lidar='WV-DIAL')
                OffLine.wavelength = wavelen_off0[0]
#                RemCom = CombHi.time_resample(delta_t=tres,t0=HourLim[0]*3600,update=True,remainder=True)
                RemOff = OffLine.time_resample(tedges=MasterTime,update=True,remainder=True)
                
                wavelen_on = wavelen_on0
                wavelen_off = wavelen_off0                
                
                firstFile = False
                
                
            else:
                if np.size(RemOn.time) > 0:
                    OnTmp = lp.LidarProfile(on_data.T,timeDataOn,label='Online Backscatter Channel',descript = 'Unpolarization\nOnline WV-DIAL Backscatter Returns',bin0=0,lidar='WV-DIAL')
                    OnTmp.cat_time(RemOn)
                    RemOn = OnTmp.time_resample(tedges=MasterTime,update=True,remainder=True)
                    OnLine.cat_time(OnTmp,front=False)
                    
                    OffTmp = lp.LidarProfile(off_data.T,timeDataOff,label='Offline Backscatter Channel',descript = 'Unpolarization\nOffline WV-DIAL Backscatter Returns',bin0=0,lidar='WV-DIAL')
                    OffTmp.cat_time(RemOff)
                    RemOff = OffTmp.time_resample(tedges=MasterTime,update=True,remainder=True)
                    OffLine.cat_time(OffTmp,front=False)
                else:
                    OnTmp = lp.LidarProfile(off_data.T,timeDataOn,label='Online Backscatter Channel',descript = 'Unpolarization\nOnline WV-DIAL Backscatter Returns',bin0=0,lidar='WV-DIAL')
                    RemOn = OnTmp.time_resample(tedges=MasterTime,update=True,remainder=True)
                    OnLine.cat_time(OnTmp,front=False)
                    
                    OffTmp = lp.LidarProfile(off_data.T,timeDataOff,label='Offline Backscatter Channel',descript = 'Unpolarization\nOffline WV-DIAL Backscatter Returns',bin0=0,lidar='WV-DIAL')                
                    RemOff = OffTmp.time_resample(tedges=MasterTime,update=True,remainder=True)
                    OffLine.cat_time(OffTmp,front=False)
                wavelen_on = np.concatenate((wavelen_on,wavelen_on0))
                wavelen_off = np.concatenate((wavelen_off,wavelen_off0))
#            print(Molecular.profile.shape)
#            print(CombHi.profile.shape)





# Update the HourLim definition to account for multiple days.  Plots use this
# to display only the desired plot portion.
HourLim = np.array([Hours[0,0],Hours[1,-1]+deltat_0*24])

OnLine.slice_time(HourLim*3600)
OnLineRaw = OnLine.copy()
OnLine.nonlinear_correct(30e-9);
OnLine.bg_subtract(BGIndex)


OffLine.slice_time(HourLim*3600)
OffLineRaw = OffLine.copy()
OffLine.nonlinear_correct(30e-9);
OffLine.bg_subtract(BGIndex)



#plt.figure(figsize=(15,5)); 
#plt.pcolor(Molecular.time/3600,Molecular.range_array*1e-3, np.log10(1e9*Molecular.profile.T/Molecular.binwidth_ns/(dt*7e3)));
#plt.colorbar()
#plt.clim([3,8])
#plt.title(Molecular.lidar + ' Molecular Count Rate [Hz] (BG Subtracted)')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)
#
#plt.figure(figsize=(15,5)); 
#plt.pcolor(CombHi.time/3600,CombHi.range_array*1e-3, np.log10(1e9*CombHi.profile.T/CombHi.binwidth_ns/(dt*7e3)));
#plt.colorbar()
#plt.clim([3,8])
#plt.title(Molecular.lidar + ' Combined Count Rate [Hz] (BG Subtracted)')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)

if OffLine.time.size > OnLine.time.size:
    OffLine.slice_time_index(time_lim=np.array([0,OnLine.time.size]))
elif OffLine.time.size < OnLine.time.size:
    OnLine.slice_time_index(time_lim=np.array([0,OffLine.time.size]))

# mask based on raw counts - remove points where there are < 2 counts
if use_mask:
    NanMask = np.logical_or(OnLine.profile < 2.0,OffLine.profile < 2.0)
    OnLine.profile = np.ma.array(OnLine.profile,mask=NanMask)
    OffLine.profile = np.ma.array(OffLine.profile,mask=NanMask)
#    Molecular.profile[NanMask] = np.nan
#    CombHi.profile[NanMask] = np.nan



#OnLine.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_geo:
#    Molecular.geo_overlap_correct(geo_corr)
#OnLine.range_correct();
OnLine.slice_range(range_lim=[0,MaxAlt])
OnLine.range_resample(delta_R=zres*dR,update=True)
#OnLine.conv(5.0,0.7)  # regrid by convolution

#OffLine.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_diff_geo:
#    OffLine.diff_geo_overlap_correct(diff_geo_corr,geo_reference='online')
#OffLine.range_correct()
OffLine.slice_range(range_lim=[0,MaxAlt])
OffLine.range_resample(delta_R=zres*dR,update=True)
#OffLine.conv(5.0,0.7)  # regrid by convolution



#plt.figure(); 
#plt.plot(np.sum(CombHi.profile[3600:,:],axis=0)/np.sum(Molecular.profile[3600:,:],axis=0),'.');

#if use_diff_geo:
#    #MolGain = 2.00;  # GeoCorrect prior to 12/12/2016
##    MolGain = 2.25;  # Correction after 12/12/2016
##    MolGain = 1.20;  # Correction after 12/19/2016
##    MolGain = 1.0/0.397# /64.8#2.68  # Correction after 12/21/2016
##    MolGain = 1.58/1.13  # Gain used for Scott's profile from 12/27 and additional clear data in surrounding days
#    MolGain = 1.33  # Gain updated 1/18/2016 based in very clear integrated retrievals between 0-16UTC
#else:
##    MolGain = 1.0/9*2.25*1.1*1.25  # No diff_geo
##    MolGain = 1.0/9*2.25*1.1*1.25/1.39 #*1.76  # No diff_geo After Dec. 19, 2016
##    MolGain = 1.3/9*2.25*1.1*1.25/1.39 # # No diff_geo starting Dec. 21, 2016
#    MolGain = 1.2821          # after Dec. 21, 2016 18.5 UTC - switched to 70/30 splitter
##    MolGain = 1.0
#    
#
#
#
#Molecular.gain_scale(MolGain)
#
## Correct Molecular Cross Talk
#if Cam > 0:
#    lp.FilterCrossTalkCorrect(Molecular,CombHi,Cam,smart=True)
##    Molecular.profile = 1.0/(1-Cam)*(Molecular.profile-CombHi.profile*Cam);

lp.plotprofiles([OffLine,OnLine])

OnInt = OnLine.copy();
OnInt.time_integrate();
OffInt = OffLine.copy();
OffInt.time_integrate();

#plt.figure(); plt.semilogy(CombInt.profile.flatten()/CombInt.range_array**2); plt.semilogy(5.0/2.0*OffInt.profile.flatten());

### Grab Sonde Data
sondefilename = '/scr/eldora1/HSRL_data/'+YearStr+'/'+MonthStr+'/sondes.DNR.nc'
sonde_index = 2*Days[-1]
#(Man or SigT)
f = netcdf.netcdf_file(sondefilename, 'r')
TempDat = f.variables['tpSigT'].data.copy()  # Kelvin
PresDat = f.variables['prSigT'].data.copy()*100.0  # hPa - convert to Pa (or Man or SigT)
SondeTime = f.variables['relTime'].data.copy() # synoptic time: Seconds since (1970-1-1 00:00:0.0) 
SondeAlt = f.variables['htSigT'].data.copy()  # geopotential altitude in m
StatElev = f.variables['staElev'].data.copy()  # launch elevation in m
f.close()

TempDat[np.nonzero(np.logical_or(TempDat < 173.0, TempDat > 373.0))] = np.nan;
PresDat[np.nonzero(np.logical_or(PresDat < 1.0*100, PresDat > 1500.0*100))] = np.nan;

sonde_index = np.min([np.shape(SondeAlt)[0]-1,sonde_index])
# Obtain sonde data for backscatter coefficient estimation
Tsonde = np.interp(OffLine.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],TempDat[sonde_index,:])
Psonde = np.interp(OffLine.range_array,SondeAlt[sonde_index,:]-StatElev[sonde_index],PresDat[sonde_index,:])
Psonde = Psonde*9.86923e-6  # convert Pressure to atm from Pa

dnu = np.linspace(-7e9,7e9,400)
inuL = np.argmin(np.abs(dnu))
MolSpec = lp.RB_Spectrum(Tsonde,Psonde,OffLine.wavelength,nu=dnu)
nuOff = lp.c/OffLine.wavelength
nuOn = lp.c/OnLine.wavelength



Filter = FO.FP_Etalon(1.0932e9,43.5e9,nuOff,efficiency=0.95,InWavelength=False)
Toffline = Filter.spectrum(dnu+nuOff,InWavelength=False,aoi=0.0,transmit=True)
Tonline = Filter.spectrum(dnu+nuOn,InWavelength=False,aoi=0.0,transmit=True)

plt.figure(); 
plt.plot(dnu+nuOn,Tonline); 
plt.plot(dnu+nuOff,Toffline);
plt.plot(dnu[inuL]+nuOn,Tonline[inuL],'bx',label='Online')
plt.plot(dnu[inuL]+nuOff,Toffline[inuL],'gx',label='Offline')
plt.grid(b=True)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Transmission')

nuWV = np.linspace(lp.c/828.5e-9,lp.c/828e-9,500)
sigWV = lp.WV_ExtinctionFromHITRAN(nuWV,Tsonde,Psonde)  
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
nWV = -1.0/(2*(dsig)[np.newaxis,:])*np.diff(np.log(OnLine.profile/OffLine.profile),axis=1)/OnLine.mean_dR
#nWV = -1.0/(2*(sigOn-sigOff)[np.newaxis,:])*np.diff(OnLine.profile/OffLine.profile,axis=1)
#nWV2 = -1.0/(2*(dsig)[np.newaxis,:])*np.log(OnLine.profile[:,1:]*OffLine.profile[:,:-1]/(OffLine.profile[:,1:]*OnLine.profile[:,:-1]))/OnLine.mean_dR

nWV2 = -1.0/(2*(dsig)[np.newaxis,:])*(np.diff(OnLine.profile,axis=1)/OnLine.mean_dR/OnLine.profile[:,1:]-np.diff(OffLine.profile,axis=1)/OffLine.mean_dR/OffLine.profile[:,1:])

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

plt.figure(figsize=(15,5)); 
plt.pcolor(OffLine.time/3600,range_diff*1e-3, np.log10(nWV.T));
plt.colorbar()
plt.clim([22,25])
plt.title('$n_{wv}$ '+ OffLine.lidar + ' [$m^{-3}$]')
plt.ylabel('Altitude [km]')
plt.xlabel('Time [UTC]')
plt.xlim(HourLim)




#sigWV = lp.WV_ExtinctionFromHITRAN(np.array([lp.c/OnLine.wavelength,lp.c/OffLine.wavelength]),Tsonde,Psonde,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]))

## note the operating wavelength of the lidar is 532 nm
#beta_m_sonde = sonde_scale*5.45*(550.0/780.24)**4*1e-32*Psonde/(Tsonde*lp.kB)

##plt.figure(); plt.semilogx(beta_m_sonde/np.nanmean(Molecular.profile,axis=0),Molecular.range_array)
#if sonde_scale == 1.0:
#    Mol_Beta_Scale = 1.36*0.925e-6*2.49e-11*Molecular.mean_dt/(Molecular.time[-1]-Molecular.time[0])  # conversion from profile counts to backscatter cross section
#else:
#    Mol_Beta_Scale = 1.0/sonde_scale    
#
#BSR = (CombHi.profile)/Molecular.profile
#
#beta_bs = BSR*beta_m_sonde[np.newaxis,:]  # total backscatter including molecules
##aer_beta_bs = (BSR-1)*beta_m_sonde[np.newaxis,:]    # only aerosol backscatter
##aer_beta_bs[np.nonzero(aer_beta_bs <= 0)] = 1e-10;
#
#aer_beta_dlb = lp.Calc_AerosolBackscatter(Molecular,CombHi,Temp=Tsonde,Pres=Psonde,beta_sonde_scale=sonde_scale)


#### Dynamic Integration ####
## Dynamically integrate in layers to obtain lower resolution only in areas that need it for better
## SNR.  Returned values are
## dynamically integrated molecular profile, dynamically integrated combined profile, dynamically integrated aerosol profile, resolution in time, resolution in altitude
#MolLayer,CombLayer,aer_layer_dlb,layer_t,layer_z = lp.AerBackscatter_DynamicIntegration(Molecular,CombHi,Temp=Tsonde,Pres=Psonde,num=3,snr_th=1.2,sigma = np.array([1.5,1.0]))

#MolInt = Molecular.copy();
#MolInt.time_integrate();
#CombInt = CombHi.copy();
#CombInt.time_integrate();
#aer_beta_dlb_int = lp.Calc_AerosolBackscatter(MolInt,CombInt,Tsonde,Psonde)
#lp.plotprofiles([aer_beta_dlb_int])
#
#
#if use_geo:
#    Molecular.geo_overlap_correct(geo_corr)
#    CombHi.geo_overlap_correct(geo_corr)
#
#Extinction,OptDepth,ODmol = lp.Calc_Extinction(Molecular, MolConvFactor=Mol_Beta_Scale, Temp=Tsonde, Pres=Psonde, AerProf=aer_beta_dlb)

############## Eventually fold into LidarProfileFunctions
#aerfit = aer_beta_dlb.profile
##aerfit[np.nonzero(aerfit<0)] = 0
##aerfit[np.nonzero(np.isnan(aerfit))] = np.nan
#aerfit_std = np.sqrt(aer_beta_dlb.profile_variance)
#ODfit = OptDepth.profile-ODmol[np.newaxis,:]
#ODvar = OptDepth.profile_variance
#
#
#aerfit[np.nonzero(np.isnan(aerfit))] = 0
#x_invalid = np.nonzero(aerfit < 3.3*aerfit_std)
#
#sLROpt,extinctionOpt,ODOpt,ODbiasOpt = lp.Retrieve_Ext_Block_MLE(ODfit,ODvar,aerfit,x_invalid,maxblock=30,maxLR=1e5,minLR=aer_beta_dlb.mean_dR,lam=np.array([10.0,3.0]),grad_gain=1e-1,max_iterations=200,optout=-1)
#if getMLE_extinction:
#    ExtinctionMLE,OptDepthMLE,LidarRatioMLE,ODbiasProf,xMLE_fit = lp.Retrieve_Ext_MLE(OptDepth,aer_beta_dlb,ODmol,overlap=3,lam=np.array([10.0,3.0]))


#plt.figure(); 
#plt.semilogx(aer_beta_dlb_int.range_array,aer_beta_dlb)

#OptDepth = np.log(CombHi.profile/beta_bs)*0.5
#
#extinction = -np.diff(np.log(Molecular.profile/beta_m_sonde[np.newaxis,:]),axis=1)*0.5
#
#extinction2 = -np.diff(np.log(CombHi.profile/beta_bs),axis=1)*0.5

### Run Klett Inversion for comparision
#,geo_corr=np.array([])
if runKlett:
    aer_beta_klett = lp.Klett_Inv(OffLineRaw,10e3,Temp=Tsonde,Pres=Psonde,avgRef=False,BGIndex=BGIndex,Nmean=40,kLR=1.05)
#    diff_aer_beta = np.ma.array(aer_beta_dlb.profile-aer_beta_klett.profile,mask=np.logical_and(aer_beta_dlb.profile < 1e-7,aer_beta_dlb.SNR() < 3.0))
#    diff_aer_beta = np.ma.array(aer_beta_klett.profile/aer_beta_dlb.profile,mask=np.logical_and(aer_beta_dlb.profile < 1e-7,aer_beta_dlb.SNR() < 3.0))



if plotAsDays:
    time_scale = 3600*24.0
else:
    time_scale = 3600.0

#plt.figure(figsize=(15,5)); 
#plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
#plt.colorbar()
#plt.clim([-9,-3])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#
#if runKlett:
#    plt.figure(figsize=(15,5)); 
#    plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(aer_beta_klett.profile.T));
#    plt.colorbar()
#    plt.clim([-9,-3])
#    plt.title(DateLabel + ', ' + 'Klett Estimated' + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#    plt.ylabel('Altitude [km]')
#    if plotAsDays:
#        plt.xlabel('Days [UTC]')
#        plt.xlim(HourLim/24.0)
#    else:
#        plt.xlabel('Time [UTC]')
#        plt.xlim(HourLim)
#        
#    plt.figure(figsize=(15,5)); 
#    plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(diff_aer_beta.T));
#    plt.colorbar()
#    plt.clim([-1,1])
#    plt.title(DateLabel + ', ' + 'Logarithmic Difference in' + ' Aerosol Backscatter Coefficient')
#    plt.ylabel('Altitude [km]')
#    if plotAsDays:
#        plt.xlabel('Days [UTC]')
#        plt.xlim(HourLim/24.0)
#    else:
#        plt.xlabel('Time [UTC]')
#        plt.xlim(HourLim)
#
#
#plt.figure(figsize=(15,5)); 
#plt.pcolor(CombHi.time/time_scale,CombHi.range_array*1e-3, np.log10(CombHi.profile.T));
#plt.colorbar()
#plt.clim([9,12])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Attenuated Backscatter [Counts]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#
#if getMLE_extinction:
#    plt.figure(figsize=(15,5)); 
#    plt.pcolor(ExtinctionMLE.time/time_scale,ExtinctionMLE.range_array*1e-3, np.log10(ExtinctionMLE.profile.T));
#    plt.colorbar()
#    plt.clim([-7,-3])
#    plt.title(DateLabel + ', ' + ExtinctionMLE.lidar + ' Aerosol Extinction Coefficient [$m^{-1}$]')
#    plt.ylabel('Altitude [km]')
#    if plotAsDays:
#        plt.xlabel('Days [UTC]')
#        plt.xlim(HourLim/24.0)
#    else:
#        plt.xlabel('Time [UTC]')
#        plt.xlim(HourLim)
#        
#    plt.figure(figsize=(15,5)); 
#    plt.pcolor(LidarRatioMLE.time/time_scale,LidarRatioMLE.range_array*1e-3, (xMLE_fit*LidarRatioMLE.profile).T);
#    plt.colorbar()
#    plt.clim([0,100])
#    plt.title(DateLabel + ', ' + ExtinctionMLE.lidar + ' Lidar Ratio [$sr$]')
#    plt.ylabel('Altitude [km]')
#    if plotAsDays:
#        plt.xlabel('Days [UTC]')
#        plt.xlim(HourLim/24.0)
#    else:
#        plt.xlabel('Time [UTC]')
#        plt.xlim(HourLim)

#plt.figure(figsize=(15,5)); 
#plt.pcolor(Extinction.time/3600,Extinction.range_array*1e-3, np.log10(Extinction.profile.T));
#plt.colorbar()
#plt.clim([-4,-2])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Extinction Coefficient [$m^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)

#plt.figure(); 
#plt.pcolor(OptDepth.time/3600,OptDepth.range_array*1e-3, np.log10(OptDepth.profile.T));
#plt.colorbar()
#plt.clim([0,3])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Optical Depth')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)


#plt.figure(figsize=(15,5)); 
#plt.pcolor(Extinction.time/3600,Extinction.range_array*1e-3, np.log10(Extinction.profile/aer_beta_dlb.profile).T);
#plt.colorbar()
##plt.clim([-4,-2])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Lidar Ratio [$sr$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)


#plt.figure(figsize=(15,10)); 
#plt.subplot(2,1,2)
#plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
#plt.colorbar()
#plt.clim([-8.5,-4])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#    
#plt.subplot(2,1,1)
#plt.pcolor(CombHi.time/time_scale,CombHi.range_array*1e-3, np.log10(CombHi.profile).T);  # /(CombHi.binwidth_ns*1e-9*tres*7e3)
#plt.colorbar()
#plt.clim([9,12])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Attenuated Backscatter [Counts]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#
#plt.figure(figsize=(15,10)); 
#plt.subplot(2,1,1)
#plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
#plt.colorbar()
#plt.clim([-9,-3])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)
#
#plt.subplot(2,1,2)
#plt.pcolor(aer_layer_dlb.time/time_scale,aer_layer_dlb.range_array*1e-3, np.log10(aer_layer_dlb.profile.T));
#plt.colorbar()
#plt.clim([-9,-3])
#plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#if plotAsDays:
#    plt.xlabel('Days [UTC]')
#    plt.xlim(HourLim/24.0)
#else:
#    plt.xlabel('Time [UTC]')
#    plt.xlim(HourLim)


if runKlett:
    plt.figure(figsize=(15,10)); 
    plt.subplot(3,1,1)
    plt.pcolor(aer_beta_dlb.time/time_scale,aer_beta_dlb.range_array*1e-3, np.log10(aer_beta_dlb.profile.T));
    plt.colorbar()
    plt.clim([-9,-3])
    plt.title(DateLabel + ', ' + CombHi.lidar + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
#        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
#        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    plt.ylim([0,10])
    
    plt.subplot(3,1,2)
    plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(aer_beta_klett.profile.T));
    plt.colorbar()
    plt.clim([-9,-3])
    plt.title(DateLabel + ', ' + 'Klett Estimated' + ' Aerosol Backscatter Coefficient [$m^{-1}sr^{-1}$]')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
#        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
#        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    plt.ylim([0,10])

    plt.subplot(3,1,3)
    plt.pcolor(aer_beta_klett.time/time_scale,aer_beta_klett.range_array*1e-3, np.log10(diff_aer_beta.T));
    plt.colorbar()
    plt.clim([-1,1])
    plt.title(DateLabel + ', ' + 'Logarithmic Difference in' + ' Aerosol Backscatter Coefficient')
    plt.ylabel('Altitude [km]')
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)
    plt.ylim([0,10])

#plt.show()

#np.savez('20170121_Ext1_5h_2',range_array=LidarRatioMLE.range_array,time_array=LidarRatioMLE.time,LidarRatio=LidarRatioMLE.profile,Extinction=ExtinctionMLE.profile,aer_beta_dlb=aer_beta_dlb.profile,xMLE_fit=xMLE_fit)





#plt.figure(figsize=(15,10)); 
#plt.subplot(2,1,2)
#plt.pcolor(aer_layer_dlb.time/3600,aer_layer_dlb.range_array*1e-3, (layer_z.T+1.0)*aer_layer_dlb.mean_dR);
#plt.colorbar()
##plt.clim([-8.5,-4])
#plt.title(DateLabel + ', ' + aer_layer_dlb.lidar + ' Range Resolution [m]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)
#
#plt.subplot(2,1,1)
#plt.pcolor(aer_layer_dlb.time/3600,aer_layer_dlb.range_array*1e-3, (layer_t.T+1.0)*aer_layer_dlb.mean_dt);  # /(CombHi.binwidth_ns*1e-9*tres*7e3)
#plt.colorbar()
##plt.clim([9,12])
#plt.title(DateLabel + ', ' + aer_layer_dlb.lidar + ' Time Resolution [seconds]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [UTC]')
#plt.xlim(HourLim)

## Plot SNR
#plt.figure(); 
#plt.pcolor(aer_beta_dlb.time/3600,aer_beta_dlb.range_array*1e-3, np.log10((aer_beta_dlb.profile/np.sqrt(aer_beta_dlb.profile_variance)).T));
#plt.colorbar()
##plt.clim([-8,-3])
#plt.title(CombHi.lidar + ' Aerosol Backscatter Coeffient [$m^{-1}sr^{-1}$]')
#plt.ylabel('Altitude [km]')
#plt.xlabel('Time [h-UT]')



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

