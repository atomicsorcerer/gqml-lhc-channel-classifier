import ROOT
import numpy as np
import polars as pl


def process_root_data(path_to_root_file: str) -> list[dict]:
	file = ROOT.TFile.Open(path_to_root_file, "READ")
	raw_data = file.mini
	
	events = []
	for i, entry in enumerate(raw_data):
		print(f"\rProcessing... Iteration: {i:09d}", end="")
		
		if entry.lep_n != 2:
			continue
		
		# Lepton metrics
		lepton_type = {f"lepton_type_{i}": list(set(entry.lep_type)).index(value) for i, value in
		               enumerate(entry.lep_type)}
		lepton_charge = {f"lepton_charge_{i}": value for i, value in enumerate(entry.lep_charge)}
		
		# Differences in angles between the two leptons
		lepton_thetas = [2 * np.arctan(np.exp(-value)) for value in entry.lep_eta]
		lepton_theta_diff = np.abs(lepton_thetas[1] - lepton_thetas[0])
		lepton_phi_diff = np.abs(entry.lep_phi[1] - entry.lep_phi[0])
		lepton_angular_diff = np.sqrt(lepton_theta_diff ** 2 + lepton_phi_diff ** 2)
		
		# Invariant mass of lepton pair
		lepton_pt = [value for value in entry.lep_pt]
		lepton_x_momentum = [pT * np.cos(phi) for pT, phi in zip(entry.lep_pt, entry.lep_phi)]
		lepton_y_momentum = [pT * np.sin(phi) for pT, phi in zip(entry.lep_pt, entry.lep_phi)]
		lepton_z_momentum = [pT * np.sinh(eta) for pT, eta in zip(entry.lep_pt, entry.lep_eta)]
		
		lepton_inv_mass = np.sqrt(sum(entry.lep_E) ** 2 - (
				sum(lepton_x_momentum) ** 2 + sum(lepton_y_momentum) ** 2 + sum(lepton_z_momentum) ** 2))
		
		# Jet metrics
		jet_pt = {f"jet_pt_{i}": value for i, value in enumerate(entry.jet_pt)}
		jet_theta = {f"jet_eta_{i}": 2 * np.arctan(np.exp(-value)) for i, value in enumerate(entry.jet_eta)}
		jet_phi = {f"jet_phi_{i}": value for i, value in enumerate(entry.jet_phi)}
		jet_type = {f"jet_type_{i}": list(set(entry.jet_trueflav)).index(value) for i, value in
		            enumerate(entry.jet_trueflav)}
		jet_energy = {f"lepton_energy_{i}": value for i, value in enumerate(entry.jet_E)}
		
		# Photon Coordinates
		# photon_1_x, photon_2_x = [entry.photon_pt[i] * math.cos(entry.photon_phi[i]) for i in range(2)]
		# photon_1_y, photon_2_y = [entry.photon_pt[i] * math.sin(entry.photon_phi[i]) for i in range(2)]
		# photon_1_z, photon_2_z = [entry.photon_pt[i] * math.sinh(entry.photon_eta[i]) for i in range(2)]
		#
		# photon_1_vector, photon_2_vector = (np.array([photon_1_x, photon_1_y, photon_1_z]),
		#                                     np.array([photon_2_x, photon_2_y, photon_2_z]))
		
		# Invariant mass of the photon pair equation credited to: https://arxiv.org/pdf/2101.11004.pdf
		# Energy asymmetry of photons
		# energy_asym = (photon_1_E - photon_2_E) / (photon_1_E + photon_2_E)
		# Opening angle between photon pair
		# psi = np.dot(photon_1_vector, photon_2_vector) / (
		# 		np.linalg.norm(photon_1_vector) * np.linalg.norm(photon_2_vector))
		# photon_inv_mass = (photon_1_E + photon_2_E) * math.sqrt(((1 - energy_asym ** 2) / 2) * (1 - math.cos(psi)))
		
		# jet_pt = {f"jet_pt_{i}": value for i, value in enumerate(list(entry.jet_pt)[:6])}
		# jet_eta = {f"jet_eta_{i}": value for i, value in enumerate(list(entry.jet_eta)[:6])}
		# jet_true_flav = {f"jet_true_flav_{i}": value for i, value in enumerate(list(entry.jet_trueflav)[:6])}
		
		event = {
			k: v for k, v in
			zip([*lepton_type.keys(),
			     *lepton_charge.keys(), *jet_pt.keys(), *jet_theta.keys(), *jet_phi.keys(), *jet_type.keys(),
			     *jet_energy.keys()],
			    
			    [*lepton_type.values(), *lepton_charge.values(), *jet_pt.values(), *jet_theta.values(),
			     *jet_phi.values(), *jet_type.values(), *jet_energy.values()])
		}
		
		event["id"] = i
		event["lepton_theta_diff"] = lepton_theta_diff
		event["lepton_phi_diff"] = lepton_phi_diff
		event["lepton_angular_diff"] = lepton_angular_diff
		event["lepton_inv_mass"] = lepton_inv_mass
		
		events.append(event)
	
	return events


if __name__ == "__main__":
	file_name: str = input("File name: ")
	
	processed_data: list[dict] = process_root_data("Data/Raw Data/" + file_name)
	
	print("\nWriting to CSV...")
	df: pl.DataFrame = pl.DataFrame(processed_data)
	df.write_csv(output_path := "Data/Processed Data/" + file_name.replace(".root", ".csv"))
	print(f"Saved to {output_path}")
