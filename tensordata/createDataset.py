from .kaplonek import *
from .zohar import *
import xarray as xr

def dataset():
    M, S = serology_rename()
    Z = data3D()

    M_sub, S_sub, Z_sub = M['Subject'].values, S['Subject'].values, Z['Sample'].values

    M_time, S_time = M['Day'].values, S['Time'].values

    subjects = np.array([M_sub, S_sub, Z_sub], dtype=object)
    times = np.array([M_time, S_time], dtype=object)

    receptors = ['FcR2A','FcR2B','FcR3A','FcR3B','FcRalpha','IgA','IgA1','IgA2','IgG1','IgG2','IgG3','IgG4','IgM']
    antigens = ['SARS.CoV2_N', 'CoV.OC43', 'Flu_HA', 'SARS.CoV2_S1', 'Ebola_gp','CMV','SARS.CoV2_S', 'SARS.CoV2_S2', 
                'SARS.CoV2_RBD', 'CoV.HKU1_RBD', 'CoV.HKU1_S', 'CoV.OC43_RBD','CoV.OC43_S', 'MERS_RBD', 
                'MERS_S','SARS.CoV2_S1trimer', 'SARS_RBD', 'SARS_S']

    ds = xr.Dataset(
        data_vars=dict( 
            MGH=(["M_subject","M_time", "M_receptor","M_antigen"], M.values), 
            SpaceX=(["S_subject","S_antigen","S_receptor","S_time"], S.values), 
            Zohar=(["Z_subject","Z_antigen","Z_receptor"], Z.values)
            ), 
        coords=dict(
            Subject=subjects, 
            Time=times, 
            Antigen=antigens, 
            Receptor=receptors
            ), 
        )

    
    DS = xr.Dataset(
        data_vars=dict( 
            MGH=(["M_subject","M_time", "M_receptor","M_antigen"], M.values), 
            SpaceX=(["S_subject","S_antigen","S_receptor","S_time"], S.values), 
            Zohar=(["Z_subject","Z_antigen","Z_receptor"], Z.values)
            ), 
        coords=dict(
            Subject=(["M_subject","S_subject","Z_subject"],subjects), 
            Time=times, 
            Antigen=antigens, 
            Receptor=receptors, 
            ), 
        )
    

    return DS
