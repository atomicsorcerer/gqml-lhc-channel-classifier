import math

import ROOT
import numpy as np
import polars as pl

file = ROOT.TFile.Open("Data/Raw Data/mc_341081.ttH125_gamgam.GamGam.root", "READ")
raw_data = file.mini

"""
TO DO: The magnitude of the missing transverse momentum
"""

events = []
for i, entry in enumerate(raw_data):
	# Variables chosen based on: https://iopscience.iop.org/article/10.1088/1361-6471/ac1391#gac1391s1
	
	print(f"\rProcessing... Iteration: {i:07d}", end="")
	if entry.photon_n != 2 or entry.jet_n < 3 or entry.jet_n > 6:
		continue
	
	photon_pt = {f"photon_pt_{i}": value for i, value in enumerate(entry.photon_pt)}
	photon_eta = {f"photon_eta_{i}": value for i, value in enumerate(entry.photon_eta)}
	photon_1_phi, photon_2_phi = tuple(entry.photon_phi)
	photon_1_E, photon_2_E = tuple(entry.photon_E)
	
	# Photon Coordinates
	photon_1_x, photon_2_x = [entry.photon_pt[i] * math.cos(entry.photon_phi[i]) for i in range(2)]
	photon_1_y, photon_2_y = [entry.photon_pt[i] * math.sin(entry.photon_phi[i]) for i in range(2)]
	photon_1_z, photon_2_z = [entry.photon_pt[i] * math.sinh(entry.photon_eta[i]) for i in range(2)]
	
	photon_1_vector, photon_2_vector = (np.array([photon_1_x, photon_1_y, photon_1_z]),
	                                    np.array([photon_2_x, photon_2_y, photon_2_z]))
	
	# Invariant mass of the photon pair equation credited to: https://arxiv.org/pdf/2101.11004.pdf
	# Energy asymmetry of photons
	energy_asym = (photon_1_E - photon_2_E) / (photon_1_E + photon_2_E)
	# Opening angle between photon pair
	psi = np.dot(photon_1_vector, photon_2_vector) / (np.linalg.norm(photon_1_vector) * np.linalg.norm(photon_2_vector))
	photon_inv_mass = (photon_1_E + photon_2_E) * math.sqrt(((1 - energy_asym ** 2) / 2) * (1 - math.cos(psi)))
	
	jet_pt = {f"jet_pt_{i}": value for i, value in enumerate(list(entry.jet_pt)[:6])}
	jet_eta = {f"jet_eta_{i}": value for i, value in enumerate(list(entry.jet_eta)[:6])}
	jet_true_flav = {f"jet_true_flav_{i}": value for i, value in enumerate(list(entry.jet_trueflav)[:6])}
	
	event = {
		k: v for k, v in
		zip([*photon_pt.keys(), *photon_eta.keys(), *jet_pt.keys(), *jet_eta.keys(), *jet_true_flav.keys()],
		    [*photon_pt.values(), *photon_eta.values(), *jet_pt.values(), *jet_eta.values(), *jet_true_flav.values()])
	}
	
	event["id"] = i
	event["photon_inv_mass"] = photon_inv_mass
	
	events.append(event)

if __name__ == "__main__":
	df = pl.DataFrame(events)
	df.write_csv("Data/Processed Data/341081.csv")
