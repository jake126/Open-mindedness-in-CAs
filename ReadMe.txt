#############
# GitHub File Structure
#############

CA of Scotland data sets: 
	4 files pertain to participant survey responses, with prefix "member_survey_10_shared_dataset_weekend_".
	1 file contains participant demographics: "member_survey_10_shared_dataset_demographics.csv".
	2 files pertain to wider population panel survey responses, with prefix "population_survey".
	1 file contains the (imputed) mapping of expert speaker opinions from the CA of Scotland: "CAS_expert_mapping.csv".

100_20_200_protocol.csv: the algorithmically derived optimal table allocations for 100 speakers over 10 tables for 200 time steps, using the methodology in Barrett et al's "Now We're Talking: Better Deliberation Groups through Submodular Optimization" (Proceedings of the AAAI Conference on Artificial Intelligence, 37(5), 2023, https://doi.org/10.1609/aaai.v37i5.25682.

Two Python files are included: a helper file containing all functions, and a main file that runs the relevant simulations.
