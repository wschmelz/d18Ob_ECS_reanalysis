import os
import sys
import numpy
import scipy
import glob
import loess_py

backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

# directories
data_wkspc = wkspc + "../00_Data" + "/"
kohler_clark_2026_co2_dir = data_wkspc + "00_Kohler_Clark_2026" + "/"
kohler_2023_co2_dir = data_wkspc + "01_Kohler_2023_CO2" + "/"
clark_etal_2024_gmst_dir = data_wkspc + "02_Clark_2024_GMST" + "/"
clark_etal_2025_gmsl_dir = data_wkspc + "03_Clark_2025_GMSL" + "/"
clark_etal_2025_mot_dir = data_wkspc + "04_Clark_2025_MOT_d18Osw" + "/"
miller_etal_2024_co2_dir = data_wkspc + "05_Miller_2024_CO2" + "/"
schmelz_etal_2026_dir = data_wkspc + "06_Schmelz_2026_decomposition" + "/"
deboer_etal_2014_anice_dir = data_wkspc + "07_deBoer_2014_ANICE" + "/"
kohler_etal_2015_land_ice_forcing_dir = data_wkspc + "08_Kohler_2015_land_ice_forcing" + "/"

output_dir = wkspc + "00_Output" + "/"

# input files

kohler_clark_2026_co2_filename = kohler_clark_2026_co2_dir + "BICYCLE-SE__3scenarios__atmCO2__4Ma__withheader.dat"
kohler_2023_co2_filename = kohler_2023_co2_dir + "BICYCLE-SE__18scenarios__atmCO2__5Ma__withheader.dat"
clark_etal_2024_gmst_filename = clark_etal_2024_gmst_dir + "Clark_etal_age_T_dO.csv"
clark_etal_2025_gmsl_filename = clark_etal_2025_gmsl_dir + "Clark_2025_AgeMa_SL.csv"
clark_etal_2025_mot_filename = clark_etal_2025_mot_dir + "Clark_2024_AgeMa_temp.csv"
miller_etal_2024_co2_filename = miller_etal_2024_co2_dir + "CO2_output_original.csv"
schmelz_etal_2026_filename = schmelz_etal_2026_dir + "output_BSL_T_d18Osw_d18Ob.csv"
deboer_etal_2014_gmsl_filename = deboer_etal_2014_anice_dir + "deBoer_2014_age_gmsl.csv"
kohler_etal_2015_rli_filename = kohler_etal_2015_land_ice_forcing_dir + "Kohler_2015_age_rli.dat"

# analysis constants
analysis_min_ma = 0.0
analysis_max_ma = 4.0
analysis_dt = 0.001
miller_co2_max_ma = 0.8
co2_rmse_min_ma = 1.0
co2_rmse_dt_ma = 0.25
loess_min_points = 3
loess_radius_ma = 0.4
loess_factor = 1
years_per_ma = 1000000.0
kyr_per_ma = 1000.0
co2_reference_ppm = 278.0
co2_forcing_coefficient = 5.35
land_ice_efficacy = 1.0
doubled_co2_forcing_wm2 = 3.71

# analysis arrays
age_ma = numpy.arange(analysis_min_ma,analysis_max_ma + analysis_dt / 2.0,analysis_dt)
co2_rmse_age_ma = numpy.arange(co2_rmse_min_ma,analysis_max_ma + co2_rmse_dt_ma / 2.0,co2_rmse_dt_ma)

# CO2 scenario data
kohler_clark_2026_co2 = numpy.genfromtxt(kohler_clark_2026_co2_filename,comments="!")
kohler_clark_2026_co2_age_ma = numpy.abs(kohler_clark_2026_co2[:,0]) / years_per_ma

kohler_2023_co2 = numpy.genfromtxt(kohler_2023_co2_filename,comments="!")
kohler_2023_co2_age_ma = numpy.abs(kohler_2023_co2[:,0]) / years_per_ma

# Schmelz et al. (2026) DST and BSL
schmelz_etal_2026 = numpy.genfromtxt(schmelz_etal_2026_filename,delimiter=',')
schmelz_etal_2026_age_ma = schmelz_etal_2026[:,0] / kyr_per_ma
schmelz_etal_2026_bsl_m = schmelz_etal_2026[:,1]
schmelz_etal_2026_dst_c = schmelz_etal_2026[:,2]

f_schmelz_dst_interp = scipy.interpolate.interp1d(schmelz_etal_2026_age_ma,schmelz_etal_2026_dst_c)
schmelz_dst_anomaly_k = f_schmelz_dst_interp(age_ma) - f_schmelz_dst_interp(0.0)

f_schmelz_bsl_interp = scipy.interpolate.interp1d(schmelz_etal_2026_age_ma,schmelz_etal_2026_bsl_m)
schmelz_bsl_m = f_schmelz_bsl_interp(age_ma)

# Miller et al. (2024) CO2
miller_etal_2024 = numpy.genfromtxt(miller_etal_2024_co2_filename,delimiter=',')
miller_etal_2024_age_ma = miller_etal_2024[:,0]
miller_etal_2024_co2_ppm = miller_etal_2024[:,1]

f_miller_co2_interp = scipy.interpolate.interp1d(miller_etal_2024_age_ma,miller_etal_2024_co2_ppm)
miller_co2_ppm = f_miller_co2_interp(age_ma)
miller_co2_comp_ppm = f_miller_co2_interp(co2_rmse_age_ma)

# Clark et al. (2024) GMST
clark_etal_2024_gmst = numpy.genfromtxt(clark_etal_2024_gmst_filename,delimiter=',')
clark_etal_2024_gmst_age_ma = clark_etal_2024_gmst[:,0]
clark_etal_2024_gmst_k = clark_etal_2024_gmst[:,1]

f_clark_gmst_interp = scipy.interpolate.interp1d(clark_etal_2024_gmst_age_ma,clark_etal_2024_gmst_k)
clark_gmst_anomaly_k = f_clark_gmst_interp(age_ma) - f_clark_gmst_interp(0.0)

# Clark et al. (2025) GMSL
clark_etal_2025_gmsl = numpy.genfromtxt(clark_etal_2025_gmsl_filename,delimiter=',',skip_header=1,usecols=(0,1))
clark_etal_2025_gmsl = clark_etal_2025_gmsl[numpy.all(numpy.isfinite(clark_etal_2025_gmsl),axis=1)]
clark_etal_2025_gmsl_age_ma = clark_etal_2025_gmsl[:,0]
clark_etal_2025_gmsl_m = clark_etal_2025_gmsl[:,1]

f_clark_gmsl_interp = scipy.interpolate.interp1d(clark_etal_2025_gmsl_age_ma,clark_etal_2025_gmsl_m)
clark_gmsl_m = f_clark_gmsl_interp(age_ma)

# Clark et al. (2025) mean ocean temperature
clark_etal_2025_mot = numpy.genfromtxt(clark_etal_2025_mot_filename,delimiter=',')
clark_etal_2025_mot_age_ma = clark_etal_2025_mot[:,0]
clark_etal_2025_mot_k = clark_etal_2025_mot[:,1]

f_clark_mot_interp = scipy.interpolate.interp1d(clark_etal_2025_mot_age_ma,clark_etal_2025_mot_k)
clark_mot_anomaly_k = f_clark_mot_interp(age_ma) - f_clark_mot_interp(0.0)

# de Boer et al. (2014) GMSL
deboer_etal_2014_gmsl = numpy.genfromtxt(deboer_etal_2014_gmsl_filename,delimiter=',',skip_header=1)
deboer_etal_2014_gmsl_age_ma = (numpy.abs(deboer_etal_2014_gmsl[:,0]) / kyr_per_ma)[::-1]
deboer_etal_2014_gmsl_m = deboer_etal_2014_gmsl[:,1][::-1]
deboer_etal_2014_gmsl_age_ma = numpy.append(0.0,deboer_etal_2014_gmsl_age_ma)
deboer_etal_2014_gmsl_m = numpy.append(deboer_etal_2014_gmsl_m[0],deboer_etal_2014_gmsl_m)

f_deboer_gmsl_interp = scipy.interpolate.interp1d(deboer_etal_2014_gmsl_age_ma,deboer_etal_2014_gmsl_m)
deboer_gmsl_m = f_deboer_gmsl_interp(age_ma)

# Kohler et al. (2015) land-ice radiative forcing
kohler_etal_2015_rli = numpy.genfromtxt(kohler_etal_2015_rli_filename,comments="#")
kohler_etal_2015_rli_age_ma = (numpy.abs(kohler_etal_2015_rli[:,0]) / kyr_per_ma)[::-1]
kohler_etal_2015_rli_wm2 = kohler_etal_2015_rli[:,1][::-1]
kohler_etal_2015_rli_age_ma = numpy.append(0.0,kohler_etal_2015_rli_age_ma)
kohler_etal_2015_rli_wm2 = numpy.append(kohler_etal_2015_rli_wm2[0],kohler_etal_2015_rli_wm2)

f_kohler_rli_interp = scipy.interpolate.interp1d(kohler_etal_2015_rli_age_ma,kohler_etal_2015_rli_wm2)
kohler_rli_wm2 = f_kohler_rli_interp(age_ma)

# MOT to GMSST coefficients
# Clark et al. (2025), Climate of the Past, doi:10.5194/cp-21-973-2025.

hse_young_age_ma = 0.9
hse_old_age_ma = 1.5

mot_gmsst_0 = 1.0
mot_gmsst_1 = 0.5

hse_age_ma = numpy.array([analysis_min_ma,hse_young_age_ma,hse_old_age_ma,analysis_max_ma])
hse_value = numpy.array([mot_gmsst_0,mot_gmsst_0,mot_gmsst_1,mot_gmsst_1])

# GMSST to GMST coefficients
# Clark et al. (2024), Science, doi:10.1126/science.adi1908
gmsst_gmst_b_0 = -0.037629
gmsst_gmst_b_1 = 1.603508
gmsst_gmst_b_2 = -0.058842

# BSL to land-ice radiative-forcing coefficients
# GMSL is from de Boer et al. (2014), doi:10.1038/ncomms3999
# Land-ice forcing is from Kohler et al. (2015), doi:10.1594/PANGAEA.855449

bsl_rli_design = numpy.column_stack([deboer_gmsl_m,deboer_gmsl_m**2,deboer_gmsl_m**3])
bsl_rli_b_0 = 0.0
bsl_rli_b_1,bsl_rli_b_2,bsl_rli_b_3 = numpy.linalg.lstsq(bsl_rli_design,kohler_rli_wm2,rcond=None)[0]
bsl_rli_fit_wm2 = bsl_rli_b_0 + bsl_rli_b_1 * deboer_gmsl_m + bsl_rli_b_2 * deboer_gmsl_m**2 + bsl_rli_b_3 * deboer_gmsl_m**3
bsl_rli_r_squared = 1.0 - numpy.sum((kohler_rli_wm2 - bsl_rli_fit_wm2)**2) / numpy.sum((kohler_rli_wm2 - numpy.mean(kohler_rli_wm2))**2)

# Actuo coefficient and uncertainty, Kohler et al. (2010)
actuo_alpha = 1.66
kohler_2010_r_co2_wm2 = -2.10
kohler_2010_r_co2_sd_wm2 = 0.22
kohler_2010_r_ch4_wm2 = -0.40
kohler_2010_r_ch4_sd_wm2 = 0.05
kohler_2010_r_n2o_wm2 = -0.30
kohler_2010_r_n2o_sd_wm2 = 0.10
kohler_2010_r_vegetation_wm2 = -1.09
kohler_2010_r_vegetation_sd_wm2 = 0.57
kohler_2010_r_dust_wm2 = -1.88
kohler_2010_r_dust_sd_wm2 = 0.94
kohler_2010_r_other_wm2 = kohler_2010_r_ch4_wm2 + kohler_2010_r_n2o_wm2 + kohler_2010_r_vegetation_wm2 + kohler_2010_r_dust_wm2
kohler_2010_actuo_ratio = numpy.abs(kohler_2010_r_other_wm2 / kohler_2010_r_co2_wm2)

actuo_mc_iterations = 10000
r_co2_samples_wm2 = numpy.random.normal(kohler_2010_r_co2_wm2,kohler_2010_r_co2_sd_wm2,actuo_mc_iterations)
r_ch4_samples_wm2 = numpy.random.normal(kohler_2010_r_ch4_wm2,kohler_2010_r_ch4_sd_wm2,actuo_mc_iterations)
r_n2o_samples_wm2 = numpy.random.normal(kohler_2010_r_n2o_wm2,kohler_2010_r_n2o_sd_wm2,actuo_mc_iterations)
r_vegetation_samples_wm2 = numpy.random.normal(kohler_2010_r_vegetation_wm2,kohler_2010_r_vegetation_sd_wm2,actuo_mc_iterations)
r_dust_samples_wm2 = numpy.random.normal(kohler_2010_r_dust_wm2,kohler_2010_r_dust_sd_wm2,actuo_mc_iterations)

r_other_samples_wm2 = r_ch4_samples_wm2 + r_n2o_samples_wm2 + r_vegetation_samples_wm2 + r_dust_samples_wm2
actuo_ratio_samples = numpy.abs(r_other_samples_wm2 / r_co2_samples_wm2)
actuo_alpha_samples = actuo_alpha + actuo_ratio_samples - kohler_2010_actuo_ratio

actuo_alpha_sd = numpy.std(actuo_alpha_samples)
actuo_alpha_min,actuo_alpha_max = numpy.percentile(actuo_alpha_samples,[2.5,97.5])

#functions

def mot_to_gmst(mot_anomaly_k, age_ma):

	f_hse_interp = scipy.interpolate.interp1d(hse_age_ma,hse_value)
	hse = f_hse_interp(age_ma)

	gmsst_anomaly_k = mot_anomaly_k / hse

	gmst_anomaly_k = (gmsst_gmst_b_0 + gmsst_gmst_b_1 * gmsst_anomaly_k + gmsst_gmst_b_2 * gmsst_anomaly_k**2) - gmsst_gmst_b_0

	return gmsst_anomaly_k, gmst_anomaly_k

def f_r_co2(co2_ppm):
	r_co2_wm2 = co2_forcing_coefficient * numpy.log(co2_ppm / co2_reference_ppm)
	return r_co2_wm2

def f_r_li(bsl_m):
	r_li_wm2 = bsl_rli_b_0 + bsl_rli_b_1 * bsl_m + bsl_rli_b_2 * bsl_m**2 + bsl_rli_b_3 * bsl_m**3
	return r_li_wm2

def lin_regress(x,y):
	slope = numpy.sum(x * y) / numpy.sum(x**2)
	return slope

def rmse(x,y):
	rmse_value = numpy.sqrt(numpy.mean((x - y)**2))
	return rmse_value

def f_total_forcing(r_co2_wm2,r_li_wm2):
	total_forcing_wm2 = r_co2_wm2 + land_ice_efficacy * r_li_wm2
	return total_forcing_wm2

def f_actuo_forcing(r_co2_wm2,r_li_wm2,actuo_alpha):
	actuo_forcing_wm2 = (1.0 + actuo_alpha) * r_co2_wm2 + land_ice_efficacy * r_li_wm2
	return actuo_forcing_wm2

#analysis

for idx_1 in range(0,2):

	if idx_1 == 0:
		analysis_id = "K_C_2026_reconstruction"
		co2_data_dir = kohler_clark_2026_co2_dir
		dst_anomaly_k = clark_mot_anomaly_k.copy()
		bsl_m = clark_gmsl_m.copy()
		gmsst_anomaly_k, gmst_from_mot_anomaly_k = mot_to_gmst(dst_anomaly_k,age_ma)
		gmst_anomaly_k = clark_gmst_anomaly_k.copy()

	if idx_1 == 1:
		analysis_id = "Sch_etal_2026_reanalysis"
		co2_data_dir = kohler_2023_co2_dir
		dst_anomaly_k = schmelz_dst_anomaly_k.copy()
		bsl_m = schmelz_bsl_m.copy()
		gmsst_anomaly_k, gmst_anomaly_k = mot_to_gmst(dst_anomaly_k, age_ma)

	print ("Directory:")
	print (co2_data_dir)
	print (" ")

	co2_data_list = glob.glob(co2_data_dir + "*.csv")
	scenario_names = []
	sensitivity_results = numpy.zeros((len(co2_data_list),6)) * numpy.nan

	for idx2 in range(0,len(co2_data_list)):
		co2_data_filename = co2_data_list[idx2]
		name = os.path.splitext(os.path.basename(co2_data_filename))[0]
		scenario_names.append(name)

		scenario_data = numpy.genfromtxt(co2_data_filename, delimiter=',', skip_header=1)

		f_bicycle_co2_interp = scipy.interpolate.interp1d(scenario_data[:, 0], scenario_data[:, 1])
		bicycle_co2_ppm = f_bicycle_co2_interp(age_ma)

		bicycle_co2_loess_ppm = loess_py.loess(co2_rmse_age_ma,age_ma,bicycle_co2_ppm,loess_min_points,loess_radius_ma,loess_factor)
		co2_rmse_ppm = rmse(bicycle_co2_loess_ppm,miller_co2_comp_ppm)

		if idx_1 == 1:
			miller_idx = numpy.where(age_ma <= miller_co2_max_ma)
			bicycle_co2_ppm[miller_idx] = miller_co2_ppm[miller_idx]

		r_co2_wm2 = f_r_co2(bicycle_co2_ppm)
		r_li_wm2 = f_r_li(bsl_m)

		total_forcing_wm2 = f_total_forcing(r_co2_wm2,r_li_wm2)
		s_co2_li_pointwise = gmst_anomaly_k / total_forcing_wm2
		negative_s_co2_li_percent = 100.0 * numpy.sum(s_co2_li_pointwise < 0.0) / len(s_co2_li_pointwise)
		
		s_co2_li = lin_regress(total_forcing_wm2,gmst_anomaly_k)

		actuo_forcing_wm2 = f_actuo_forcing(r_co2_wm2,r_li_wm2,actuo_alpha)
		s_actuo = lin_regress(actuo_forcing_wm2,gmst_anomaly_k)
		ecs_k = doubled_co2_forcing_wm2 * s_actuo

		ecs_uncertainty_results = numpy.zeros(actuo_mc_iterations) * numpy.nan

		for idx3 in range(0,actuo_mc_iterations):

			actuo_alpha_sample = actuo_alpha_samples[idx3]

			if (actuo_alpha_sample >= actuo_alpha_min) & (actuo_alpha_sample <= actuo_alpha_max):

				actuo_forcing_sample_wm2 = f_actuo_forcing(r_co2_wm2,r_li_wm2,actuo_alpha_sample)
				s_actuo_sample = lin_regress(actuo_forcing_sample_wm2,gmst_anomaly_k)
				ecs_sample_k = doubled_co2_forcing_wm2 * s_actuo_sample
				ecs_uncertainty_results[idx3] = ecs_sample_k

		ecs_5_perc,ecs_95_perc = numpy.nanpercentile(ecs_uncertainty_results,[5.,95.])

		sensitivity_results[idx2,:] = [s_co2_li,ecs_k,co2_rmse_ppm,negative_s_co2_li_percent,ecs_5_perc,ecs_95_perc]
		scenario_output = numpy.column_stack([age_ma,dst_anomaly_k,gmsst_anomaly_k,gmst_anomaly_k,bsl_m,r_li_wm2,bicycle_co2_ppm,r_co2_wm2,s_co2_li_pointwise,])
		header_1 = "age_ma,DST_anomaly_K,GMSST_anomaly_K,GMST_anomaly_K,BSL_m,R_LI_Wm2,CO2_ppm,R_CO2_Wm2,S_CO2_LI_K_per_Wm2"
		numpy.savetxt(output_dir+f"{analysis_id}_"+f"{name}.csv",scenario_output,delimiter=",",header=header_1,fmt="%.10g")

	summary_output = numpy.column_stack([scenario_names,sensitivity_results])
	header_2 = "scenario,S_CO2_LI_K_per_Wm2,ECS_K,CO2_RMSE_ppm,negative_S_CO2_LI_percent,ecs_5th_perc,ecs_95th_perc"
	numpy.savetxt(output_dir+f"{analysis_id}_summary.csv",summary_output,delimiter=",",header=header_2,fmt="%s")
