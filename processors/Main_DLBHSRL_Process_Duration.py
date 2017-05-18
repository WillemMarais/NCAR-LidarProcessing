# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 08:45:42 2017

@author: mhayman
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.io import netcdf
import LidarProfileFunctions as lp
import WVProfileFunctions as wv
#import MLELidarProfileFunctions as mle
#import scipy.interpolate

#from mpl_toolkits.axes_grid1 import make_axes_locatable

import datetime
import json

import glob

cal_path = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/'
cal_file = cal_path+'dlb_calvals_msu.json'

Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,5,17,startHr=20,duration=4)  # WV-DIAL RD Correction ,startHr=5,duration=4.0
#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,4,18,startHr=4.5,stopHr=5.4)  # WV-DIAL RD Correction ,startHr=5,duration=4.0

#Years,Months,Days,Hours = lp.generate_WVDIAL_day_list(2017,1,12,stopDay=19)

plotAsDays = False
getMLE_extinction = False
run_MLE = False
runKlett = False

save_as_nc = False
save_figs = False

nctag = ''  # additional tag for netcdf and figure filename

run_geo_cal = False

#ProcStart = datetime.date(Years[0],Months[0],Days[0])
ProcStart = datetime.datetime(Years[0],Months[0],Days[0],Hours[0][0])

DateLabel = ProcStart.strftime("%A %B %d, %Y")

MaxAlt = 12e3 #12e3

KlettAlt = 14e3  # altitude where Klett inversion starts

tres = 1*60.0  # resolution in time points (2 sec)
zres = 1.0  # resolution in altitude points (75 m)

use_diff_geo = False   # no diff geo correction after April ???
use_geo = True

use_mask = False
SNRmask = 0.0  #SNR level used to decide what data points we keep in the final data product
countLim = 2.0

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
    

#MCSbins = 2*280  # number of bins in a range resolved profile,  280-typical became 1400 on 2/22/2017.  After April ? use 2*280 when binwidths were changed to 250e-9.
#BinWidth = 250e-9 # MCS timing bin width in seconds.  typically 500e-9 before April ?.  250e-9 after April ?
#dt = 2  # profile accumulation time in seconds

dR = BinWidth*lp.c/2  # profile range resolution (500e-9*c/2)-typical became 100e-9*c/2 on 2/22/2017
Roffset = ((1.25+0.5)-0.5/2)*lp.c*LaserPulseWidth  # offset in range

BGIndex = -50; # negative number provides an index from the end of the array
Cam = 0.00 # Cross talk of aerosols into the molecular channel - 0.005 on Dec 21 2016 after 18.5UTC
            # 0.033 found for 4/18/2017 11UTC extinction test case

save_data_path = '/h/eol/mhayman/HSRL/DLBHSRL/Processed_Data/'
save_fig_path = '/h/eol/mhayman/HSRL/DLBHSRL/Processed_Data/Plots/'
sonde_path = '/scr/eldora1/HSRL_data/'


#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20161212.npz'
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20161219_2.npz'
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20161221_2.npz'
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20161227.npz'  #Provided by Scott
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20170301.npz'  #Provided by Scott
#diff_geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/diff_geo_DLB_20170222.npz'  #Update with Zeeman calibration

diff_geo_data = np.load(diff_geo_file)
diff_geo_corr = diff_geo_data['diff_geo_prof']
#diff_lim_index = 14  # correction is not applied below this index - used on 20161219_2
#diff_geo_corr[:diff_lim_index] = 1.2/(1.0/9*2.25*1.1*1.25/1.39)

#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20161219.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20161213.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20161227.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20161227.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170119.npz'
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170120_0.npz'  # obtained using OD, includes variance in profile
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170413.npz'  # obtained using OD, includes variance in profile
#geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170418.npz'  # obtained using OD, includes variance in profile
geo_file = '/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170512.npz'  # obtained using OD, includes variance in profile

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

basepath = '/scr/eldora1/MSU_h2o_data/'
FieldLabel = 'NF'
MolFileBase = 'Online_Raw_Data.dat'
CombFileBase = 'Offline_Raw_Data.dat'

# define time grid for lidar signal processing to occur
MasterTime = np.arange(Hours[0,0]*3600,Days.size*24*3600-(24-Hours[-1,-1])*3600+tres,tres)

if save_as_nc or save_figs:
    ncfilename0 = lp.create_ncfilename('DLBHSRL',Years,Months,Days,Hours,tag=nctag)
    ncfilename = save_data_path+ncfilename0
    figfilename = save_fig_path + ncfilename0[:-3] + nctag

[Molecular,CombHi],lambda_hsrl,HourLim = wv.Load_DLB_Data(basepath,FieldLabel,[MolFileBase,CombFileBase],MasterTime,Years,Months,Days,Hours,MCSbins,lidar='DLB-HSRL',dt=dt,Roffset=Roffset,BinWidth=BinWidth)

Molecular.slice_time(HourLim*3600)
MolRaw = Molecular.copy()
#Molecular.nonlinear_correct(38e-9);
Molecular.bg_subtract(BGIndex)




CombHi.slice_time(HourLim*3600)
CombRaw = CombHi.copy()
#CombHi.nonlinear_correct(29.4e-9);
CombHi.bg_subtract(BGIndex)


if CombHi.time.size > Molecular.time.size:
    CombHi.slice_time_index(time_lim=np.array([0,Molecular.time.size]))
elif CombHi.time.size < Molecular.time.size:
    Molecular.slice_time_index(time_lim=np.array([0,CombHi.time.size-1]))

# mask based on raw counts - remove points where there are < 2 counts
if use_mask:
    NanMask = np.logical_or(Molecular.profile < 2.0,CombHi.profile < 2.0)
    Molecular.profile = np.ma.array(Molecular.profile,mask=NanMask)
    CombHi.profile = np.ma.array(CombHi.profile,mask=NanMask)
#    Molecular.profile[NanMask] = np.nan
#    CombHi.profile[NanMask] = np.nan



#Molecular.energy_normalize(TransEnergy*EnergyNormFactor)
#if use_geo:
#    Molecular.geo_overlap_correct(geo_corr)
Molecular.range_correct();
Molecular.slice_range(range_lim=[0,MaxAlt])
Molecular.range_resample(delta_R=zres*dR,update=True)
Molecular.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
#Molecular.conv(1.5,2.0)  # regrid by convolution
MolRaw.slice_range_index(range_lim=[1,1e6])  # remove bottom bin

#CombHi.energy_normalize(TransEnergy*EnergyNormFactor)
if use_diff_geo:
    CombHi.diff_geo_overlap_correct(diff_geo_corr,geo_reference='mol')
#if use_geo:
#    CombHi.geo_overlap_correct(geo_corr)
CombHi.range_correct()
CombHi.slice_range(range_lim=[0,MaxAlt])
CombHi.range_resample(delta_R=zres*dR,update=True)
CombHi.slice_range_index(range_lim=[1,1e6])  # remove bottom bin
#CombHi.conv(1.5,2.0)  # regrid by convolution

# if running the Klett inversion, the raw data is used and we need it to be on the same range grid as the sondes
if runKlett:
    CombRaw.slice_range(range_lim=[0,MaxAlt])
    CombRaw.range_resample(delta_R=zres*dR,update=True)

CombRaw.slice_range_index(range_lim=[1,1e6])  # remove bottom bin


#plt.figure(); 
#plt.plot(np.sum(CombHi.profile[3600:,:],axis=0)/np.sum(Molecular.profile[3600:,:],axis=0),'.');
#
#if use_diff_geo:
#    #MolGain = 2.00;  # GeoCorrect prior to 12/12/2016
##    MolGain = 2.25;  # Correction after 12/12/2016
##    MolGain = 1.20;  # Correction after 12/19/2016
##    MolGain = 1.0/0.397# /64.8#2.68  # Correction after 12/21/2016
##    MolGain = 1.58/1.13  # Gain used for Scott's profile from 12/27 and additional clear data in surrounding days
#    MolGain = 1.33  # Gain updated 1/18/2016 based in very clear integrated retrievals between 0-16UTC
##    MolGain = 2.8789  # Gain used for Scott's profile from 3/1/2017 and additional clear data in surrounding days
#else:
##    MolGain = 1.0/9*2.25*1.1*1.25  # No diff_geo
##    MolGain = 1.0/9*2.25*1.1*1.25/1.39 #*1.76  # No diff_geo After Dec. 19, 2016
##    MolGain = 1.3/9*2.25*1.1*1.25/1.39 # # No diff_geo starting Dec. 21, 2016
#    MolGain = 1.2821          # after Dec. 21, 2016 18.5 UTC - switched to 70/30 splitter
##    MolGain = 1.33          # after April. 14, 2017 mode scrambler
##    MolGain = 3.17          # after May 12, 2017 - combined with WV DIAL
##    MolGain = 1.0
    



Molecular.gain_scale(MolGain)

# Correct Molecular Cross Talk
if Cam > 0:
    lp.FilterCrossTalkCorrect(Molecular,CombHi,Cam,smart=True)
#    Molecular.profile = 1.0/(1-Cam)*(Molecular.profile-CombHi.profile*Cam);

lp.plotprofiles([CombHi,Molecular])


beta_mol_sonde,sonde_time,sonde_index_prof,temp,pres,sonde_index = lp.get_beta_m_sonde(Molecular,Years,Months,Days,sonde_path,interp=True,returnTP=True)
#beta_mol_sonde.gain_scale(sonde_scale)

isonde = np.argmin(pres.time-pres.time/2.0)
Psonde = pres.profile[isonde,:]
Tsonde = temp.profile[isonde,:]

#plt.figure(); plt.semilogx(beta_m_sonde/np.nanmean(Molecular.profile,axis=0),Molecular.range_array)
if sonde_scale == 1.0:
    Mol_Beta_Scale = 1.36*0.925e-6*2.49e-11*Molecular.mean_dt/(Molecular.time[-1]-Molecular.time[0])  # conversion from profile counts to backscatter cross section
else:
    Mol_Beta_Scale = 1.0/sonde_scale    

BSR = (CombHi.profile)/Molecular.profile

#beta_bs = BSR*beta_m_sonde[np.newaxis,:]  # total backscatter including molecules
beta_bs = BSR*beta_mol_sonde.profile

## Depricated aerosol backscatter retrieval.  Better if there are no sondes to use as reference.
#aer_beta_dlb = lp.Calc_AerosolBackscatter(Molecular,CombHi,Temp=Tsonde,Pres=Psonde)

# Latest aerosol backscatter retrival using 2D sonde profiles
aer_beta_dlb = lp.AerosolBackscatter(Molecular,CombHi,beta_mol_sonde)


#### Dynamic Integration ####
## Dynamically integrate in layers to obtain lower resolution only in areas that need it for better
## SNR.  Returned values are
## dynamically integrated molecular profile, dynamically integrated combined profile, dynamically integrated aerosol profile, resolution in time, resolution in altitude
#MolLayer,CombLayer,aer_layer_dlb,layer_t,layer_z = lp.AerBackscatter_DynamicIntegration(Molecular,CombHi,Temp=Tsonde,Pres=Psonde,num=3,snr_th=1.2,sigma = np.array([1.5,1.0]))

MolInt = Molecular.copy();
MolInt.time_integrate();
CombInt = CombHi.copy();
CombInt.time_integrate();
sonde_int = beta_mol_sonde.copy()
sonde_int.time_integrate();
#aer_beta_dlb_int = lp.Calc_AerosolBackscatter(MolInt,CombInt,Tsonde,Psonde)
aer_beta_dlb_int = lp.AerosolBackscatter(MolInt,CombInt,sonde_int)
lp.plotprofiles([aer_beta_dlb_int])


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

Extinction,OptDepth,ODmol = lp.Calc_Extinction(MolLP, MolConvFactor=Mol_Beta_Scale, Temp=Tsonde, Pres=Psonde, AerProf=aer_beta_dlb)


if run_MLE and use_geo:
#    beta_a_mle,alpha_a_mle,sLR_mle,xvalid_mle,CamList,GmList,GcList,ProfileErrorMol,ProfileErrorComb,fit_mol_mle,fit_comb_mle = \
#            mle.MLE_Estimate_OptProp(MolRaw,CombRaw,aer_beta_LP,geo_data,PresDat[sonde_index,:],TempDat[sonde_index,:],SondeAlt[sonde_index,:]-StatElev[sonde_index], \
#            minSNR=1.0,dG=0.04,fitfilt=True)
    beta_a_mle,alpha_a_mle,sLR_mle,xvalid_mle,CamList,GmList,GcList,ProfileErrorMol,ProfileErrorComb,fit_mol_mle,fit_comb_mle = \
            mle.MLE_Estimate_OptProp(MolRaw,CombRaw,aer_beta_LP,geo_data,sonde_path, \
            minSNR=1.0,dG=0.04,fitfilt=True)
    # merge mle backscatter data with direct retrieval
    beta_a_merge = aer_beta_dlb.copy()
    iMLE = np.nonzero(xvalid_mle.profile)
    beta_a_merge.profile[iMLE] = beta_a_mle.profile[iMLE]
    # trim fit profile ranges so they mesh with current processed ranges
    fit_mol_mle.slice_range_index(range_lim=[0,aer_beta_dlb.profile.shape[1]])
    fit_comb_mle.slice_range_index(range_lim=[0,aer_beta_dlb.profile.shape[1]])
    

### Run Klett Inversion for comparision
#,geo_corr=np.array([])
if runKlett:
    aer_beta_klett = lp.Klett_Inv(CombRaw,KlettAlt,Temp=Tsonde,Pres=Psonde,avgRef=False,BGIndex=BGIndex,geo_corr=geo_corr,Nmean=40,kLR=1.05)
#    diff_aer_beta = np.ma.array(aer_beta_dlb.profile-aer_beta_klett.profile,mask=np.logical_and(aer_beta_dlb.profile < 1e-7,aer_beta_dlb.SNR() < 3.0))
    diff_aer_beta = np.ma.array(aer_beta_klett.profile/aer_beta_dlb.profile,mask=np.logical_and(aer_beta_dlb.profile < 1e-7,aer_beta_dlb.SNR() < 3.0))


#ncfilename = '/h/eol/mhayman/write_py_netcdf3.nc'
if save_as_nc:
    CombHi.write2nc(ncfilename)
    Molecular.write2nc(ncfilename)
    aer_beta_dlb.write2nc(ncfilename)
    beta_mol_sonde.write2nc(ncfilename)
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



#lp.pcolor_profiles([Molecular,CombHi],climits=[[8,12],[8,12]],plotAsDays=plotAsDays)

# plot aerosol backscatter
# climits=[[-8,-4]] for clouds and aerosols
# climits=[[-7.4,-6.0]] for aerosols
lp.pcolor_profiles([aer_beta_dlb],climits=[[-7.4,-6.0]],plotAsDays=plotAsDays)    
if save_figs:
    plt.savefig(figfilename+'_AerosolBackscatter_Aerosol_Colorbar.png')   

lp.pcolor_profiles([aer_beta_dlb],climits=[[-8.0,-4.0]],plotAsDays=plotAsDays)    
if save_figs:
    plt.savefig(figfilename+'_AerosolBackscatter.png')
    

if runKlett:       
    lp.pcolor_profiles([aer_beta_klett],climits=[[-8,-4]],plotAsDays=plotAsDays)  
        
    plt.figure(figsize=(15,5)); 
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

    
lp.pcolor_profiles([CombHi],climits=[[8,12]],plotAsDays=plotAsDays)  
if save_figs:
    plt.savefig(figfilename+'_CombHi.png')


lp.pcolor_profiles([aer_beta_dlb,CombHi],climits=[[-8,-4],[8,12]],plotAsDays=plotAsDays)  
if save_figs:
    plt.savefig(figfilename+'_AerosolBackscatter_and_CombinedBackscatter.png')

if runKlett:
    lp.pcolor_profiles([aer_beta_dlb,aer_beta_klett],climits=[[-8,-4],[-8,-4]],plotAsDays=plotAsDays)  
    if save_figs:
        plt.savefig(figfilename+'_AerosolBackscatter_Direct_and_Klett.png')
    

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

if run_MLE and use_geo:    
    ### Plot Both MLE and Direct Retrievals ###
    lp.pcolor_profiles([beta_merge],climits=[[-8.0,-4.0]],plotAsDays=plotAsDays) 
    if save_figs:
        plt.savefig(figfilename+'_MLE_AerosolBackscatter.png')
    
    lp.pcolor_profiles([aer_beta_dlb,beta_merge],climits=[[-8.0,-4.0],[-8.0,-4.0]],plotAsDays=plotAsDays)
    if save_figs:
        plt.savefig(figfilename+'MLE_and_Direct_AerosolBackscatter.png')
    
    lp.pcolor_profiles([beta_a_mle,alpha_a_mle,sLR_mle],climits=[[-8.0,-4.0],[-7.0,-3.0],[15,40]],scale=['log','log','linear'],plotAsDays=plotAsDays)
    if save_figs:
        plt.savefig(figfilename+'MLE_Backscatter_Extinction_LidarRatio.png')
    
    nmask = np.ones(xvalid_mle.profile.shape)
    nmask[np.nonzero(xvalid_mle.profile==0)] = np.nan;
    plt.figure(); 
    plt.scatter((sLR_mle.profile*nmask).flatten(),np.log10(beta_merge.profile.flatten()),c=np.log10(CombHi.flatten()),alpha=0.5)
    plt.plot(-8.5*np.array([-3,-9])-24.5,np.array([-3,-9]),'k--')
    plt.grid(b=True)
    plt.xlabel('Lidar Ratio [$sr$]')
    plt.ylabel('$log_{10}$ Aerosol Backscatter [$m^{-1}sr^{-1}$]')
    plt.colorbar()
    plt.xlim([0,100])
    plt.title(DateLabel + ', ' +aer_beta_dlb.lidar)
    if save_figs:
        plt.savefig(figfilename+'MLE_LidarRatio_vs_Backscatter.png')    
    
    plt.figure(); 
    plt.semilogy(beta_a_mle.time/time_scale,ProfileErrorMol)
    plt.semilogy(beta_a_mle.time/time_scale,ProfileErrorComb)
    plt.grid(b=True)
    plt.ylabel('Profile RMS Error')
    plt.legend(('Molecular','Combined'))
    if plotAsDays:
        plt.xlabel('Days [UTC]')
        plt.xlim(HourLim/24.0)
    else:
        plt.xlabel('Time [UTC]')
        plt.xlim(HourLim)



plt.show()


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
##f1 = np.concatenate((np.arange(1,8),np.arange(42,61)));
#f1 = np.arange(1,61)
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
##f2 = np.concatenate((np.arange(50,100),np.arange(130,200)));
##FitOrder = 4
#f2 = np.arange(61,108)
#FitOrder = 0
#
#pfit1 = np.polyfit(f2,FitProf2[f2],FitOrder)
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
#
#np.savez('diff_geo_DLB_20170223',diff_geo_prof=diff_geo_prof,Day=Day,Month=Month,Year=Year,HourLim=HourLim)





"""
Geo Overlap
can only be run if the profile is run at minimum z-resolution, does not have
a geo correction already imparted and is run over the full range of the 
lidar.
"""

## 12/13/2016 - 10-12 UT
#
## 12/26/2016 - 20.4-1.2 UT
#

if run_geo_cal and not use_geo and zres==1.0 and MaxAlt > 25:
    Mol_Beta_Scale = 1.0
    beta_m_sonde = beta_mol_sonde.profile[isonde,:]
    lin_fit_lower = 150
    lin_fit_upper = 330
    
    plt.figure(); 
    plt.semilogx(Mol_Beta_Scale*MolInt.profile.flatten(),MolInt.range_array)
    plt.semilogx(beta_m_sonde,CombHi.range_array)
    plt.grid(b=True)
    
    # add a correction for aerosol loading
    # extinction term actually includes molecular extinction as well
    calLR = 0  # assumed aerosol lidar ratio
    z_upper = 4e3  # upper limit on aerosol extinction effects
    calLR_upper = 0;  # assumed lidar ratio for profile above i_upper
    calLRvar = 20**2  # assumed variance in the assumed lidar ratio
    calLRvec = np.zeros(aer_beta_dlb_int.profile.size)
    calLRvec[np.nonzero(aer_beta_dlb_int.range_array <= z_upper)] = calLR
    calLRvec[np.nonzero(aer_beta_dlb_int.range_array > z_upper)] = calLR_upper
    aerExt = np.exp(-2*aer_beta_dlb_int.mean_dR*np.cumsum(calLR*aer_beta_dlb_int.profile.flatten()+8*np.pi/3*beta_m_sonde))
    
    # propagate error in assumed extinction
    varExt = calLRvar*(-2*aer_beta_dlb_int.mean_dR*np.cumsum(aer_beta_dlb_int.profile.flatten())*aerExt)**2 \
        + aer_beta_dlb_int.profile_variance.flatten()*(-2*aer_beta_dlb_int.range_array[-1]*calLR*aerExt)**2
    
    
    plt.figure();
    plt.plot(beta_m_sonde/(MolInt.profile.flatten()))
    plt.plot(beta_m_sonde*aerExt/(MolInt.profile.flatten()))
    
    ## Set constant above 47th bin
    #geo_prof = np.ones(np.size(MolInt.profile))
    #geo_prof[0:47] = (beta_m_sonde/(Mol_Beta_Scale*MolInt.profile.flatten()))[np.newaxis,0:47]
    #geo_prof = np.hstack((MolInt.range_array[:,np.newaxis],geo_prof[:,np.newaxis]))
    
    ## Run a linear fit above 65th bin
    geo_prof = beta_m_sonde*aerExt/(Mol_Beta_Scale*MolInt.profile.flatten())
    xfit = np.arange(MolInt.profile[0,lin_fit_lower:lin_fit_upper].size)
    yfit = geo_prof[lin_fit_lower:lin_fit_upper]
    wfit = 1.0/np.sqrt(MolInt.profile_variance[0,lin_fit_lower:lin_fit_upper].flatten())
    wfit[0] = 10*np.max(wfit)
    #pfit = lp.polyfit_with_fixed_points(1,xfit,yfit, np.array([0]) ,np.array([geo_prof[200]]))
    pfit = np.polyfit(xfit,yfit,1,w=wfit)
    xprof = np.arange(MolInt.profile[0,lin_fit_lower:].size)
    geo_prof[lin_fit_lower:] = np.polyval(pfit,xprof)
    
    
    
    var_geo_prof =  varExt*(beta_m_sonde/(Mol_Beta_Scale*MolInt.profile.flatten()))**2 \
        +  MolInt.profile_variance.flatten()*(beta_m_sonde*aerExt/(Mol_Beta_Scale*MolInt.profile.flatten()**2))**2
    
    
    #geo_prof = np.hstack((MolGeo.range_array[:,np.newaxis],GeoFromOD,geo_corr_var))
    geo_prof = np.hstack((MolInt.range_array[:,np.newaxis],geo_prof[:,np.newaxis],var_geo_prof[:,np.newaxis]))
    
    plt.figure(); 
    plt.plot(beta_m_sonde*aerExt/(Mol_Beta_Scale*MolInt.profile.flatten()))
    plt.plot(geo_prof[:,1])
    plt.plot(np.sqrt(var_geo_prof))
    
#    np.savez('/h/eol/mhayman/PythonScripts/HSRL_Processing/NewHSRLPython/calibrations/geo_DLB_20170512',geo_prof=geo_prof,Day=Days,Month=Months,Year=Years,HourLim=HourLim,Hours=Hours,Mol_Beta_Scale=Mol_Beta_Scale,tres=tres,zres=zres,Nprof=Molecular.time.size)
