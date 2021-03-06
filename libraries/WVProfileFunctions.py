# -*- coding: utf-8 -*-
"""
Created on Mon May  8 13:13:03 2017

@author: mhayman
"""

import numpy as np
import LidarProfileFunctions as lp
#import FourierOpticsLib as FO
import scipy as sp

import datetime

import glob


def WaterVapor_Simple(OnLine,OffLine,Psonde,Tsonde):
    """
    Performs a simple compuation of the water vapor profile using
    a single radiosonde and assumes the wavelength is constant over the
    profile.
    
    WaterVapor_Simple(OnLine,OffLine,Psonde,Tsonde)
    OnLine - the online lidar profile
    OffLine - the offline lidar profile
    
    Psonde - pressure in (in atm) obtained from a sonde or model
            altitude profile is matched to the lidar profiles
    Tsonde - temperature (in K) obtainted from a sonde or model
            altitude profile is matched to the lidar profiles
    
    returns
    nWV - a lidar profile containing the directly computed water vapor in g/m^3
    
    """
    
    # compute frequencies from wavelength terms
    nuOff = lp.c/OffLine.wavelength  # Offline laser frequency
    nuOn = lp.c/OnLine.wavelength   # Online laser frequency


    sigWV0 = lp.WV_ExtinctionFromHITRAN(np.array([nuOn,nuOff]),Tsonde,Psonde,nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True)
    sigOn = sigWV0[:,0]
    sigOff= sigWV0[:,1]

    range_diff = OnLine.range_array[1:]-OnLine.mean_dR/2.0  # range grid for diffentiated signals
    dsig = np.interp(range_diff,OnLine.range_array,sigOn-sigOff)  # interpolate difference in absorption to range_diff grid points
    
    # create the water vapor profile
    nWV = OnLine.copy()
    nWV.profile = -1.0/(2*(dsig)[np.newaxis,:])*np.diff(np.log(OnLine.profile/OffLine.profile),axis=1)/OnLine.mean_dR
    nWV.label = 'Water Vapor Number Density'
    nWV.descript = 'Water Vapor Number Density'
    nWV.profile_type = '$m^{-3}$'
    nWV.range_array = range_diff
    nWV.profile_variance = (0.5/(dsig)/OnLine.mean_dR)**2*( \
        OnLine.profile_variance[:,1:]/OnLine.profile[:,1:]**2 + \
        OnLine.profile_variance[:,:-1]/OnLine.profile[:,:-1]**2 + \
        OffLine.profile_variance[:,1:]/OffLine.profile[:,1:]**2 + \
        OffLine.profile_variance[:,:-1]/OffLine.profile[:,:-1]**2)
    
    # convert to g/m^3
    nWV.gain_scale(lp.mH2O/lp.N_A)  
    nWV.profile_type = '$g/m^{3}$'
    nWV.label = 'Absolute Humidity'
    nWV.descript = 'Absolute Humidity'
    
#    nWV.conv(2.0,3.0)  # low pass filter if desired
    return nWV
    
def WaterVapor_Simple_RD_Test(OnLine,OffLine,Psonde,Tsonde,BSR):
    """
    Performs a simple compuation of the water vapor profile using
    a single radiosonde and assumes the wavelength is constant over the
    profile.
    
    WaterVapor_Simple(OnLine,OffLine,Psonde,Tsonde)
    OnLine - the online lidar profile
    OffLine - the offline lidar profile
    
    Psonde - pressure in (in atm) obtained from a sonde or model
            altitude profile is matched to the lidar profiles
    Tsonde - temperature (in K) obtainted from a sonde or model
            altitude profile is matched to the lidar profiles
    
    returns
    nWV - a lidar profile containing the directly computed water vapor in g/m^3
    
    """
    
    # compute frequencies from wavelength terms
    dnu = np.linspace(-70e9,70e9,4000)  # array of spectrum relative to transmitted frequency
    inuL = np.argmin(np.abs(dnu))  # index to the laser frequency in the spectrum
    nuOff = lp.c/OffLine.wavelength  # Offline laser frequency
    nuOn = lp.c/OnLine.wavelength   # Online laser frequency

    dBSR = np.diff(BSR,axis=1)/OffLine.mean_dR  # Calculate backscatter ratio derivative
    BSR_interp = 0.5*(BSR[:,:-1]+BSR[:,1:])  # differential range interpolated BSR
    
    WVcorr = dBSR/BSR_interp  # correction term for wv extinction
    
#    Filter = FO.FP_Etalon(1.0932e9,43.5e9,nuOff,efficiency=0.95,InWavelength=False)
#    Toffline = Filter.spectrum(dnu+nuOff,InWavelength=False,aoi=0.0,transmit=True)
#    Tonline = Filter.spectrum(dnu+nuOn,InWavelength=False,aoi=0.0,transmit=True)
    
    nuWV = np.linspace(lp.c/828.5e-9,lp.c/828e-9,500)   # set region in hitran profile to use
    # sigWV is a 2D array with laser frequency on the 0 axis and range on the 1 axis. 
    sigWV = lp.WV_ExtinctionFromHITRAN(nuWV,Tsonde,Psonde,freqnorm=True) #,nuLim=np.array([lp.c/835e-9,lp.c/825e-9]))    # get the WV spectrum from hitran data
    
    sigF = sp.interpolate.interp1d(nuWV,sigWV)  # set up frequency interpolation for the two DIAL wavelengths
    sigWVOn = sigF(dnu+nuOn).T  # get the absorption spectrum around the online wavelength
    sigWVOff = sigF(dnu+nuOff).T  # get the absorption spectrum around the offline wavelength
    
    #sigWVOn = lp.WV_ExtinctionFromHITRAN(nuOn+dnu,Tsonde,Psonde) 
    #sigWVOff = lp.WV_ExtinctionFromHITRAN(nuOff+dnu,Tsonde,Psonde)
    
    sigOn = sigWVOn[inuL,:]
    sigOff = sigWVOff[inuL,:]

    
#    sigWV = lp.WV_ExtinctionFromHITRAN(np.array([nuOn,nuOff]),Tsonde,Psonde,nuLim=np.array([lp.c/835e-9,lp.c/825e-9])) 
#    sigOn = sigWV[:,0]
#    sigOff = sigWV[:,1]
    
    
    
    range_diff = OnLine.range_array[1:]-OnLine.mean_dR/2.0  # range grid for diffentiated signals
    dsig = np.interp(range_diff,OnLine.range_array,sigOn-sigOff)  # interpolate difference in absorption to range_diff grid points
    
    # create the water vapor profile
    nWV = OnLine.copy()
    nWV.profile = -1.0/(2*(dsig)[np.newaxis,:])*(np.diff(np.log(OnLine.profile/OffLine.profile),axis=1)/OnLine.mean_dR+WVcorr)
    nWV.label = 'Water Vapor Number Density'
    nWV.descript = 'Water Vapor Number Density'
    nWV.profile_type = '$m^{-3}$'
    nWV.range_array = range_diff
    nWV.profile_variance = (0.5/(dsig)/OnLine.mean_dR)**2*( \
        OnLine.profile_variance[:,1:]/OnLine.profile[:,1:] - \
        OnLine.profile_variance[:,:-1]/OnLine.profile[:,:-1] - \
        OffLine.profile_variance[:,1:]/OffLine.profile[:,1:] + \
        OffLine.profile_variance[:,:-1]/OffLine.profile[:,:-1])
    
    # convert to g/m^3
    nWV.gain_scale(lp.mH2O/lp.N_A)  
    nWV.profile_type = '$g/m^{3}$'
    nWV.label = 'Absolute Humidity'
    nWV.descript = 'Absolute Humidity'
    
#    nWV.conv(2.0,3.0)  # low pass filter if desired
    return nWV


def WaterVapor_2D(OnLine,OffLine,lam_On,lam_Off,pres,temp):
    """
    Performs a simple compuation of the water vapor profile using
    a single radiosonde and assumes the wavelength is constant over the
    profile.
    
    WaterVapor_Simple(OnLine,OffLine,Psonde,Tsonde)
    OnLine - the online lidar profile
    OffLine - the offline lidar profile
    
    Psonde - pressure in (in atm) obtained from a sonde or model
            altitude profile is matched to the lidar profiles
    Tsonde - temperature (in K) obtainted from a sonde or model
            altitude profile is matched to the lidar profiles
    
    returns
    nWV - a lidar profile containing the directly computed water vapor in g/m^3
    
    """
    
    range_diff = OnLine.range_array[1:]-OnLine.mean_dR/2.0  # range grid for diffentiated signals
    # create the water vapor profile
    nWV = OnLine.copy()
    nWV.label = 'Water Vapor Number Density'
    nWV.descript = 'Water Vapor Number Density'
    nWV.profile_type = '$m^{-3}$'
    nWV.range_array = range_diff

    dsig = np.zeros((nWV.time.size,nWV.range_array.size))
    for ai in range(OnLine.time.size):
        # compute frequencies from wavelength terms
        nuOff = lp.c/lam_Off[ai]  # Offline laser frequency
        nuOn = lp.c/lam_On[ai]   # Online laser frequency
        sigWV0 = lp.WV_ExtinctionFromHITRAN(np.array([nuOn,nuOff]),temp[ai,:],pres[ai,:],nuLim=np.array([lp.c/828.5e-9,lp.c/828e-9]),freqnorm=True)
        sigOn = sigWV0[:,0]
        sigOff= sigWV0[:,1]
    
        
        dsig[ai,:] = np.interp(range_diff,OnLine.range_array,sigOn-sigOff)  # interpolate difference in absorption to range_diff grid points
        
    nWV.profile = -1.0/(2*(dsig))*np.diff(np.log(OnLine.profile/OffLine.profile),axis=1)/OnLine.mean_dR
    nWV.profile_variance = (0.5/(dsig)/OnLine.mean_dR)**2*( \
        OnLine.profile_variance[:,1:]/OnLine.profile[:,1:]**2 + \
        OnLine.profile_variance[:,:-1]/OnLine.profile[:,:-1]**2 + \
        OffLine.profile_variance[:,1:]/OffLine.profile[:,1:]**2 + \
        OffLine.profile_variance[:,:-1]/OffLine.profile[:,:-1]**2)
    
    # convert to g/m^3
    nWV.gain_scale(lp.mH2O/lp.N_A)  
    nWV.profile_type = '$g/m^{3}$'
    nWV.label = 'Absolute Humidity'
    nWV.descript = 'Absolute Humidity'
    
    return nWV
    
#def WaterVapor_2D(OnLine,OffLine,lam_On,lam_Off,pres,temp,sonde_index):
#    """
#    Performs a direct inversion of the water vapor profile using
#    2D radiosonde profiles and time resolved wavelength information.
#    
#    WaterVapor_Simple(OnLine,OffLine,Psonde,Tsonde)
#    OnLine - the online lidar profile
#    OffLine - the offline lidar profile
#    
#    lam_On - online wavelength resolved in time
#    lam_Off - offline wavelength resolved in time    
#    
#    pres - pressure in (in atm) obtained from a sonde or model
#            altitude profile is matched to the lidar profiles
#    temp - temperature (in K) obtainted from a sonde or model
#            altitude profile is matched to the lidar profiles
#    
#    sonde_index - time array indicating the sonde used as reference for the profile
#    
#    returns
#    nWV - a lidar profile containing the directly computed water vapor in g/m^3
#    
#    """
#    
#    # compute frequencies from wavelength terms
##    dnu = np.linspace(-7e9,7e9,400)  # array of spectrum relative to transmitted frequency
##    inuL = np.argmin(np.abs(dnu))  # index to the laser frequency in the spectrum
##    nuOff = lp.c/lam_Off  # Offline laser frequency
##    nuOn = lp.c/lam_On   # Online laser frequency
#    
#    
#    
##    Filter = FO.FP_Etalon(1.0932e9,43.5e9,nuOff,efficiency=0.95,InWavelength=False)
##    Toffline = Filter.spectrum(dnu+nuOff,InWavelength=False,aoi=0.0,transmit=True)
##    Tonline = Filter.spectrum(dnu+nuOn,InWavelength=False,aoi=0.0,transmit=True)
##    
##    nuWV = np.linspace(lp.c/828.5e-9,lp.c/828e-9,500)   # set region in hitran profile to use
##    
##    for ai in range(OnLine.time.size):
##        # sigWV is a 2D array with laser frequency on the 0 axis and range on the 1 axis. 
##        sigWV = lp.WV_ExtinctionFromHITRAN(nuWV,temp.profile[ai,:],pres.profile[ai,:])    # get the WV spectrum from hitran data
##        
##        sigF = sp.interpolate.interp1d(nuWV,sigWV)  # set up frequency interpolation for the two DIAL wavelengths
##        sigWVOn = sigF(dnu+nuOn).T  # get the absorption spectrum around the online wavelength
##        sigWVOff = sigF(dnu+nuOff).T  # get the absorption spectrum around the offline wavelength
##        
##        #sigWVOn = lp.WV_ExtinctionFromHITRAN(nuOn+dnu,Tsonde,Psonde) 
##        #sigWVOff = lp.WV_ExtinctionFromHITRAN(nuOff+dnu,Tsonde,Psonde)
##        
##        sigOn = sigWVOn[inuL,:]
##        sigOff = sigWVOff[inuL,:]
##        
##        
##        
##    range_diff = OnLine.range_array[1:]-OnLine.mean_dR/2.0  # range grid for diffentiated signals
##    dsig = np.interp(range_diff,OnLine.range_array,sigOn-sigOff)  # interpolate difference in absorption to range_diff grid points
#    
#    
#    sigOn,sigOff,sigOn_dr,sigOff_dr,range_diff = Get_Absorption_2D(lam_On,lam_Off,pres,temp,sonde_index)    
#    dsig = sigOn_dr - sigOff_dr
#    
#    # create the water vapor profile
#    nWV = OnLine.copy()
#    nWV.profile = -1.0/(2*(dsig)[np.newaxis,:])*np.diff(np.log(OnLine.profile/OffLine.profile),axis=1)/OnLine.mean_dR
#    nWV.label = 'Water Vapor Number Density'
#    nWV.descript = 'Water Vapor Number Density'
#    nWV.profile_type = '$m^{-3}$'
#    nWV.range_array = range_diff
#    
#    # convert to g/m^3
#    nWV.gain_scale(lp.mH2O/lp.N_A)  
#    nWV.profile_type = '$g/m^{3}$'
#    
##    nWV.conv(2.0,3.0)  # low pass filter if desired
#    return nWV
#    
#def Get_Absorption_2D(lam_On,lam_Off,pres,temp,sonde_index):
#    """
#    Obtain the absorption cross section of water vapor at the
#    Online and Offline wavelengths provided for a 2D pressure and temeprature
#    profile
#    """    
#    
#    # get time indices where sonde reference changes
#    ichange = np.nonzero(np.diff(sonde_index)>=1)[0]+1
#    
#    # find all the instances where the wavelength changes significanty
#    dlam = 0.0001e-9  # allowed change in wavelength before reestimating the WV absorption
#    no_on_change = False  # flag indicating no more changes are found (terminate loop)
#    no_off_change = False
#    ref_on = lam_On[0]
#    ref_off = lam_Off[0]
#    i_ref = 0
#    i_on = np.array([0])
#    i_off = np.array([0])
#    i_lam_change = np.array([])
#    while not no_on_change and not no_off_change:
#        i_ref = np.nanmin([i_on[0]+i_ref+1,i_off[0]+i_ref+1])
#        ref_on = lam_On[i_ref]
#        ref_off = lam_Off[i_ref]
#        i_lam_change = np.concatenate((i_lam_change,np.array([i_ref])))
#        i_on = np.nonzero(np.abs(np.cumsum(lam_On[i_ref+1:]-ref_on))>dlam)[0]
#        if len(i_on) == 0:
#            i_on = np.nan
#            no_on_change = True
#            
#        i_off = np.nonzero(np.abs(np.cumsum(lam_Off[i_ref+1:]-ref_off))>dlam)[0] 
#        if len(i_off) == 0:
#            i_off = np.nan
#            no_off_change = True
#            
#            
#        
##    dnu = np.linspace(-7e9,7e9,400)  # array of spectrum relative to transmitted frequency
##    inuL = np.argmin(np.abs(dnu))  # index to the laser frequency in the spectrum
#    nuWV = np.linspace(lp.c/828.5e-9,lp.c/828e-9,500)
#     
#    nuOff = lp.c/lam_Off  # Offline laser frequency
#    nuOn = lp.c/lam_On   # Online laser frequency     
#    
#    # cross sections on standard range grid
#    sigOn = np.zeros(pres.profile.shape)
#    sigOff = np.zeros(pres.profile.shape)
#    
#    # cross sections on grid for range differentiated profiles
#    sigOn_dr = np.zeros((pres.profile.shape[0],pres.profile.shape[1]-1))
#    sigOff_dr = np.zeros((pres.profile.shape[0],pres.profile.shape[1]-1))
#    
#    range_diff = pres.range_array[1:]-pres.mean_dR/2.0  # range grid for diffentiated signals
#     
#    for ai in range(pres.time.size):
#        # sigWV is a 2D array with laser frequency on the 0 axis and range on the 1 axis. 
#        sigWV = lp.WV_ExtinctionFromHITRAN(nuWV,temp.profile[ai,:],pres.profile[ai,:])    # get the WV spectrum from hitran data
#        
#        sigF = sp.interpolate.interp1d(nuWV,sigWV)  # set up frequency interpolation for the two DIAL wavelengths
#        sigOn[ai,:] = sigF(nuOn[ai])
#        sigOff[ai,:] = sigF(nuOff[ai])
#        
#        sigOn_dr[ai,:] = np.interp(range_diff,pres.range_array,sigOn[ai,:])
#        sigOff_dr[ai,:] = np.interp(range_diff,pres.range_array,sigOff[ai,:])
##        sigWVOn = sigF(dnu+nuOn[ai]).T  # get the absorption spectrum around the online wavelength
##        sigWVOff = sigF(dnu+nuOff[ai]).T  # get the absorption spectrum around the offline wavelength
#        
##        sigOn[ai,:] = sigWVOn[inuL,:]
##        sigOff[ai,:] = sigWVOff[inuL,:]
#        
#    return sigOn,sigOff,sigOn_dr,sigOff_dr,range_diff
    

def Load_DLB_Data(basepath,FieldLabel,FileBase,MasterTime,Years,Months,Days,Hours,MCSbins,lidar='WV-DIAL',dt=2,
    Roffset=225.0,BinWidth=250e-9):
    """
    reads in data from WV-DIAL custom format binaries.
    basepath - path to data files e.g. basepath = '/scr/eldora1/MSU_h2o_data/'
    FieldLabel - 'NF' or 'FF'
    FileBase - list of channel base names e.g. FileBase = ['Online_Raw_Data.dat','Offline_Raw_Data.dat']
    
    MasterTime - Time grid to reshape the data to

    Years - list of years from lp.generate_WVDIAL_day_list
    Months - list of months from lp.generate_WVDIAL_day_list
    Days - list of days from lp.generate_WVDIAL_day_list
    Hours - list of hours from lp.generate_WVDIAL_day_list
    
    lidar - string identifying the DLB lidar.  This determines the channel
        labels used for the profiles.
        Defaults to 'WV-DIAL'.  Also accepts
        'DLB-HSRL'
    returns list of lidar profiles requested and 
        list of corresponding wavelength and surface station data and 
        the time limits in Hours
        
    
    FieldLabel_WV = 'FF'
    ON_FileBase = 'Online_Raw_Data.dat'
    OFF_FileBase = 'Offline_Raw_Data.dat'
    
    FieldLabel_HSRL = 'NF'
    MolFileBase = 'Online_Raw_Data.dat'
    CombFileBase = 'Offline_Raw_Data.dat'
    
    example function call:
    [OnLine,OffLine],[[lambda_on,lambda_off],[surf_temp],[surf_pres],[surf_humid]],HourLim = \
        wv.Load_DLB_Data(basepath,FieldLabel_WV,[ON_FileBase,OFF_FileBase], \
        MasterTimeWV,Years,Months,Days,Hours,MCSbins,lidar='WV-DIAL',dt=dt, \
        Roffset=Roffset,BinWidth=BinWidth)

    """    
    
    # set channel labels and descriptions based on lidar provided    
    
    label = ['']*len(FileBase)
    descript = ['']*len(FileBase)
    if lidar == 'wv-dial' or lidar == 'WV-DIAL':
        lidar = 'WV-DIAL'
        for ifb in range(len(FileBase)):
            if FileBase[ifb] == 'Online_Raw_Data.dat':
                label[ifb] = 'Online Backscatter Channel'
                descript[ifb] = 'Unpolarization\nOnline WV-DIAL Backscatter Returns'
                i_shot_data = ifb  # index where shot counts can be found
            elif FileBase[ifb] == 'Offline_Raw_Data.dat':
                label[ifb] = 'Offline Backscatter Channel'
                descript[ifb] = 'Unpolarization\nOffline WV-DIAL Backscatter Returns'
            else:
                label[ifb] = 'Unknown Backscatter Channel'
                descript[ifb] = 'Unknown Channel WV-DIAL Backscatter Returns'
    elif lidar == 'dlb-hsrl' or lidar == 'DLB-HSRL' or 'db-hsrl' or lidar == 'DB-HSRL':
        lidar = 'DLB-HSRL'
        for ifb in range(len(FileBase)):
            if FileBase[ifb] == 'Online_Raw_Data.dat':
                label[ifb] = 'Molecular Backscatter Channel'
                descript[ifb] = 'Unpolarization\nMolecular Backscatter Returns'
                i_shot_data = ifb  # index where shot counts can be found
            elif FileBase[ifb] == 'Offline_Raw_Data.dat':
                label[ifb] = 'Total Backscatter Channel'
                descript[ifb] = 'Unpolarization\nHigh Gain\nCombined Aerosol and Molecular Returns'
            else:
                label[ifb] = 'Unknown Backscatter Channel'
                descript[ifb] = 'Unknown Channel DLB-HSRL Backscatter Returns'  
                
                
    dR = BinWidth*lp.c/2 
    
    ProcStart = datetime.date(Years[0],Months[0],Days[0])
    
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
                loadfile = []
                for ifb in range(len(FileBase)):
                    loadfile.extend([SubDirs[idir]+FileBase[ifb]])
                
                Hour = np.double(SubDirs[idir][-3:-1])
                
                #### LOAD NETCDF DATA ####
                prf_data = []
                prf_vars = []
                wavelen0 = []
                timeData = []
                shots = []
                for ifb in range(len(FileBase)):
                    tmp_data,tmp_vars = lp.read_WVDIAL_binary(loadfile[ifb],MCSbins)
                    prf_data.extend([tmp_data])
                    prf_vars.extend([tmp_vars])
                    
                    wavelen0.extend([tmp_vars[1,:]])
                    timeData.extend([3600*24*(np.remainder(tmp_vars[0,:],1)+deltat_0)])
                    shots.extend([np.ones(np.shape(timeData[ifb]))*np.mean(prf_vars[i_shot_data][6,:])])                                     
                    itimeBad = np.nonzero(np.diff(timeData[ifb])<0)[0]        
                    if itimeBad.size > 0:
                        timeData[ifb][itimeBad+1] = timeData[ifb][itimeBad]+dt

#                # load profile data
                if firstFile:
                    profiles = []
                    prof_rem = []
                    wavelen = []
                    t_wavelen = []
                    h_vars = []
                    for ifb in range(len(FileBase)):
                        ptemp = lp.LidarProfile(prf_data[ifb].T,timeData[ifb],label=label[ifb],descript = descript[ifb],bin0=-Roffset/dR,lidar=lidar,shot_count=shots[ifb],binwidth=BinWidth,StartDate=ProcStart)
                        prem = ptemp.time_resample(tedges=MasterTime,update=True,remainder=True)
                        profiles.extend([ptemp.copy()])
                        prof_rem.extend([prem.copy()])
                        
                        wavelen.extend([wavelen0[ifb]])
                        t_wavelen.extend([timeData[ifb]])
                        h_vars.extend([prf_vars[ifb]])
                            
                    firstFile = False
                    
                    
                    
                else:
                    for ifb in range(len(FileBase)):
                        if np.size(prof_rem[ifb].time) > 0:
                            ptemp = lp.LidarProfile(prf_data[ifb].T,timeData[ifb],label=label[ifb],descript = descript[ifb],bin0=-Roffset/dR,lidar=lidar,shot_count=shots[ifb],binwidth=BinWidth,StartDate=ProcStart)
                            ptemp.cat_time(prof_rem[ifb])

                            prem = ptemp.time_resample(tedges=MasterTime,update=True,remainder=True)
                            profiles[ifb].cat_time(ptemp,front=False)
                            prof_rem[ifb] = prem.copy()
                            

                        else:
                            ptemp = lp.LidarProfile(prf_data[ifb].T,timeData[ifb],label=label[ifb],descript = descript[ifb],bin0=-Roffset/dR,lidar=lidar,shot_count=shots[ifb],binwidth=BinWidth,StartDate=ProcStart)

                            prem = ptemp.time_resample(tedges=MasterTime,update=True,remainder=True)
                            profiles[ifb].cat_time(ptemp,front=False)
                            prof_rem[ifb] = prem.copy()

                        wavelen[ifb] = np.concatenate((wavelen[ifb],wavelen0[ifb]))
                        t_wavelen[ifb] = np.concatenate((t_wavelen[ifb],timeData[ifb]))
                        h_vars[ifb] = np.concatenate((h_vars[ifb],prf_vars[ifb]),axis=1)
    lambda_lidar = []
    surf_temp = []
    surf_pres = []
    surf_humid = []
    for ifb in range(len(FileBase)):               
        dt_WL = np.mean(np.diff(t_wavelen[ifb]))  # determine convolution kernel size
        conv_kern = np.ones(np.ceil(profiles[0].mean_dt/dt_WL))  # build convolution kernel
        conv_kern = conv_kern/np.sum(conv_kern)  # normalize convolution kernel
        wavelen_filt = np.convolve(conv_kern,wavelen[ifb],'valid')    
        t_filt = np.convolve(conv_kern,t_wavelen[ifb],'valid')
        lambda_lidar.extend([1e-9*np.interp(profiles[0].time,t_filt,wavelen_filt)])
        profiles[ifb].wavelength = 1e-9*np.median(np.round(lambda_lidar[ifb]*1e9,decimals=6))  # set profile wavelength based on the median value
        
        # get temperature, pressure and humidity        
        if 'Offline' in FileBase[ifb]:
             # % relative humidity is stored at index 6.
             # convert it to absolute humidity in g/m^3
             abs_humid = 6.112*np.exp(17.67*h_vars[ifb][4,:]/(h_vars[ifb][4,:]+243.5))*h_vars[ifb][6,:]*2.1674/(273.15+h_vars[ifb][4,:])
             humid_filt = np.convolve(conv_kern,abs_humid,'valid')    
             surf_humid.extend([np.interp(profiles[0].time,t_filt,humid_filt)])
             
             pres_filt = np.convolve(conv_kern,h_vars[ifb][5,:]/1013.25,'valid')    # convert to atm
             surf_pres.extend([np.interp(profiles[0].time,t_filt,pres_filt)])
             
             temp_filt = np.convolve(conv_kern,h_vars[ifb][4,:]+273.15,'valid')     # convert to K
             surf_temp.extend([np.interp(profiles[0].time,t_filt,temp_filt)])
    
    HourLim = np.array([Hours[0,0],Hours[1,-1]+deltat_0*24])
    
    return profiles,[lambda_lidar,surf_temp,surf_pres,surf_humid],HourLim
    