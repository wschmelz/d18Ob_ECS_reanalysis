import os
import numpy
import scipy
import loess_py

backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

# directories
data_wkspc = wkspc + "../00_Data" + "/"
kohler_2023_co2_dir = data_wkspc + "01_Kohler_2023_CO2" + "/"
miller_etal_2024_co2_dir = data_wkspc + "05_Miller_2024_CO2" + "/"
schmelz_etal_2026_dir = data_wkspc + "06_Schmelz_2026_decomposition" + "/"
deboer_etal_2014_anice_dir = data_wkspc + "07_deBoer_2014_ANICE" + "/"
kohler_etal_2015_land_ice_forcing_dir = data_wkspc + "08_Kohler_2015_land_ice_forcing" + "/"

output_dir = wkspc + "00_Output" + "/"

# input files
kohler_2023_seplusv6_filename = kohler_2023_co2_dir + "CO2_Kohler_2023__SEplusV6.csv"
miller_etal_2024_co2_filename = miller_etal_2024_co2_dir + "CO2_output_original.csv"
schmelz_etal_2026_filename = schmelz_etal_2026_dir + "output_BSL_T_d18Osw_d18Ob.csv"
deboer_etal_2014_gmsl_filename = deboer_etal_2014_anice_dir + "deBoer_2014_age_gmsl.csv"
kohler_etal_2015_rli_filename = kohler_etal_2015_land_ice_forcing_dir + "Kohler_2015_age_rli.dat"

# analysis constants
analysis_min_ma = 0.0
analysis_max_ma = 4.0
analysis_dt = 0.001
miller_co2_max_ma = 0.8
loess_min_points = 3
loess_radius_ma = 0.4
loess_factor = 1
kyr_per_ma = 1000.0
co2_reference_ppm = 278.0
co2_forcing_coefficient = 5.35
land_ice_efficacy = 1.0
doubled_co2_forcing_wm2 = 3.71
residual_multipliers = numpy.linspace(0.25,2.0,15)
d18osw_per_10_m = 0.10
meters_per_permil = 10.0 / d18osw_per_10_m
d18osw_modern_reference = 0.0

# analysis arrays
age_ma = numpy.arange(analysis_min_ma,analysis_max_ma + analysis_dt / 2.0,analysis_dt)

# Schmelz et al. (2026) decomposition
schmelz_etal_2026 = numpy.genfromtxt(schmelz_etal_2026_filename,delimiter=',')
schmelz_etal_2026_age_ma = schmelz_etal_2026[:,0] / kyr_per_ma
schmelz_etal_2026_bsl_m = schmelz_etal_2026[:,1]
schmelz_etal_2026_dst_k = schmelz_etal_2026[:,2]
schmelz_etal_2026_d18osw_permil = schmelz_etal_2026[:,3]
schmelz_etal_2026_d18ob_permil = schmelz_etal_2026[:,4]

f_schmelz_bsl_interp = scipy.interpolate.interp1d(schmelz_etal_2026_age_ma,schmelz_etal_2026_bsl_m)
bsl_base_m = f_schmelz_bsl_interp(age_ma)

f_schmelz_dst_interp = scipy.interpolate.interp1d(schmelz_etal_2026_age_ma,schmelz_etal_2026_dst_k)
dst_base_k = f_schmelz_dst_interp(age_ma)
dst_base_anomaly_k = dst_base_k - f_schmelz_dst_interp(0.0)

f_schmelz_d18osw_interp = scipy.interpolate.interp1d(schmelz_etal_2026_age_ma,schmelz_etal_2026_d18osw_permil)
d18osw_input_permil = f_schmelz_d18osw_interp(age_ma)

f_schmelz_d18ob_interp = scipy.interpolate.interp1d(schmelz_etal_2026_age_ma,schmelz_etal_2026_d18ob_permil)
d18ob_permil = f_schmelz_d18ob_interp(age_ma)

# Köhler et al. (2023) SE+V6 CO2 scenario
kohler_2023_seplusv6 = numpy.genfromtxt(kohler_2023_seplusv6_filename,delimiter=',',skip_header=1)
kohler_2023_seplusv6_age_ma = kohler_2023_seplusv6[:,0]
kohler_2023_seplusv6_co2_ppm = kohler_2023_seplusv6[:,1]

f_kohler_2023_seplusv6_interp = scipy.interpolate.interp1d(kohler_2023_seplusv6_age_ma,kohler_2023_seplusv6_co2_ppm)
co2_ppm = f_kohler_2023_seplusv6_interp(age_ma)

# Miller et al. (2024) CO2
miller_etal_2024 = numpy.genfromtxt(miller_etal_2024_co2_filename,delimiter=',')
miller_etal_2024_age_ma = miller_etal_2024[:,0]
miller_etal_2024_co2_ppm = miller_etal_2024[:,1]

f_miller_co2_interp = scipy.interpolate.interp1d(miller_etal_2024_age_ma,miller_etal_2024_co2_ppm)
miller_co2_ppm = f_miller_co2_interp(age_ma)
miller_idx = numpy.where(age_ma <= miller_co2_max_ma)
co2_ppm[miller_idx] = miller_co2_ppm[miller_idx]

# de Boer et al. (2014) ANICE GMSL
deboer_etal_2014_gmsl = numpy.genfromtxt(deboer_etal_2014_gmsl_filename,delimiter=',',skip_header=1)
deboer_etal_2014_gmsl_age_ma = (numpy.abs(deboer_etal_2014_gmsl[:,0]) / kyr_per_ma)[::-1]
deboer_etal_2014_gmsl_m = deboer_etal_2014_gmsl[:,1][::-1]
deboer_etal_2014_gmsl_age_ma = numpy.append(0.0,deboer_etal_2014_gmsl_age_ma)
deboer_etal_2014_gmsl_m = numpy.append(deboer_etal_2014_gmsl_m[0],deboer_etal_2014_gmsl_m)

f_deboer_gmsl_interp = scipy.interpolate.interp1d(deboer_etal_2014_gmsl_age_ma,deboer_etal_2014_gmsl_m)
deboer_gmsl_m = f_deboer_gmsl_interp(age_ma)

# Köhler et al. (2015) land-ice radiative forcing
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

# Actuo coefficient
actuo_alpha = 1.66

# functions

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

def f_total_forcing(r_co2_wm2,r_li_wm2):
	total_forcing_wm2 = r_co2_wm2 + land_ice_efficacy * r_li_wm2
	return total_forcing_wm2

def f_actuo_forcing(r_co2_wm2,r_li_wm2,actuo_alpha):
	actuo_forcing_wm2 = (1.0 + actuo_alpha) * r_co2_wm2 + land_ice_efficacy * r_li_wm2
	return actuo_forcing_wm2

# baseline arrays
f_hse_interp = scipy.interpolate.interp1d(hse_age_ma,hse_value)
hse = f_hse_interp(age_ma)
gmsst_base_anomaly_k,gmst_base_anomaly_k = mot_to_gmst(dst_base_anomaly_k,age_ma)
dst_loess_k = loess_py.loess(age_ma,age_ma,dst_base_k,loess_min_points,loess_radius_ma,loess_factor)
dst_residual_k = dst_base_k - dst_loess_k

# analysis
summary_results = numpy.zeros((len(residual_multipliers),12)) * numpy.nan
timeseries_results = []

for idx in range(0,len(residual_multipliers)):
	residual_multiplier = residual_multipliers[idx]
	dst_adjusted_k = dst_loess_k + residual_multiplier * dst_residual_k
	dst_adjusted_anomaly_k = dst_adjusted_k - f_schmelz_dst_interp(0.0)

	d18osw_adjusted_permil = 0.27 + d18ob_permil + (dst_adjusted_k - 16.1) / 4.76
	bsl_adjusted_m = meters_per_permil * (d18osw_modern_reference - d18osw_adjusted_permil)

	gmsst_adjusted_anomaly_k,gmst_adjusted_anomaly_k = mot_to_gmst(dst_adjusted_anomaly_k,age_ma)

	r_co2_wm2 = f_r_co2(co2_ppm)
	r_li_wm2 = f_r_li(bsl_adjusted_m)
	total_forcing_wm2 = f_total_forcing(r_co2_wm2,r_li_wm2)
	s_co2_li_pointwise = gmst_adjusted_anomaly_k / total_forcing_wm2
	negative_s_co2_li_percent = 100.0 * numpy.sum(s_co2_li_pointwise < 0.0) / len(s_co2_li_pointwise)
	s_co2_li = lin_regress(total_forcing_wm2,gmst_adjusted_anomaly_k)

	actuo_forcing_wm2 = f_actuo_forcing(r_co2_wm2,r_li_wm2,actuo_alpha)
	s_actuo = lin_regress(actuo_forcing_wm2,gmst_adjusted_anomaly_k)
	ecs_k = doubled_co2_forcing_wm2 * s_actuo

	summary_results[idx,:] = [
		residual_multiplier,
		len(age_ma),
		numpy.std(dst_residual_k),
		numpy.std(residual_multiplier * dst_residual_k),
		numpy.min(d18osw_adjusted_permil),
		numpy.max(d18osw_adjusted_permil),
		numpy.min(bsl_adjusted_m),
		numpy.max(bsl_adjusted_m),
		s_co2_li,
		s_actuo,
		ecs_k,
		negative_s_co2_li_percent,
	]

	timeseries_results.append(numpy.column_stack([
		numpy.zeros(len(age_ma)) + residual_multiplier,
		age_ma,
		dst_base_k,
		dst_loess_k,
		dst_residual_k,
		dst_adjusted_k,
		dst_adjusted_anomaly_k,
		hse,
		gmsst_adjusted_anomaly_k,
		gmst_base_anomaly_k,
		gmst_adjusted_anomaly_k,
		d18ob_permil,
		d18osw_input_permil,
		d18osw_adjusted_permil,
		bsl_base_m,
		bsl_adjusted_m,
		r_co2_wm2,
		r_li_wm2,
		total_forcing_wm2,
		actuo_forcing_wm2,
		s_co2_li_pointwise,
	]))

# output
summary_header = "residual_multiplier,n_regression_points,base_DST_residual_std_K,adjusted_DST_residual_std_K,d18Osw_min_permil,d18Osw_max_permil,BSL_min_m,BSL_max_m,S_CO2_LI_K_per_Wm2,S_actuo_K_per_Wm2,ECS_K,negative_S_CO2_LI_percent"
numpy.savetxt(output_dir+"ECS_bias_sensitivity_summary.csv",summary_results,delimiter=",",header=summary_header,comments="",fmt="%.8g")

timeseries_header = "residual_multiplier,age_ma,DST_base_K,DST_800kyr_LOESS_K,DST_residual_K,DST_adjusted_K,MOT_adjusted_anomaly_K,HSE,GMSST_adjusted_anomaly_K,GMST_base_anomaly_K,GMST_adjusted_anomaly_K,d18Ob_permil,d18Osw_input_permil,d18Osw_recalculated_permil,BSL_base_m,BSL_adjusted_m,R_CO2_Wm2,R_LI_adjusted_Wm2,R_total_Wm2,R_actuo_Wm2,S_CO2_LI_pointwise_K_per_Wm2"
numpy.savetxt(output_dir+"ECS_bias_sensitivity_timeseries.csv",numpy.vstack(timeseries_results),delimiter=",",header=timeseries_header,comments="",fmt="%.8g")
