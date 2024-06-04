import ROOT
import numpy as np
import polars as pl


def process_root_data(path_to_root_file: str, limit: int = None, show_iteration_count: bool = False, label=None) -> \
		list[dict]:
	"""
	Convert a ROOT file to a CSV with certain high-level variables pre-computed.
	
	Args:
		path_to_root_file: Path to the ROOT file to convert.
		limit: Maximum number of events to be processed.
		show_iteration_count: Print a running iteration count to track the function's progress.
		label: Identifier data that can be optionally added to each entry in a dataset.
	
	Returns:
		list[dict]: List of each event processed.
	"""
	file = ROOT.TFile.Open(path_to_root_file, "READ")
	raw_data = file.mini
	
	events = []
	for i, entry in enumerate(raw_data):
		if limit is not None and i >= limit:
			break
		
		if show_iteration_count:
			print(f"\rProcessing... Iteration: {(i + 1):09d}", end="")
		
		if entry.lep_n != 2:  # Reject all events with more or less than two leptons
			continue
		
		# Lepton metrics (one hot encoded)
		lepton_type = {f"lepton_type_{i}": list(set(entry.lep_type)).index(value) for i, value in
		               enumerate(entry.lep_type)}
		lepton_charge = {f"lepton_charge_{i}": value for i, value in enumerate(entry.lep_charge)}
		lepton_phi = {f"lepton_phi_{i}": value for i, value in enumerate(entry.lep_phi)}
		lepton_pt = {f"lepton_pt_{i}": value for i, value in enumerate(entry.lep_pt)}
		lepton_eta = {f"lepton_eta_{i}": value for i, value in enumerate(entry.lep_eta)}
		
		# Differences in angles between the two leptons
		lepton_thetas = [2 * np.arctan(np.exp(-value)) for value in entry.lep_eta]
		lepton_theta_diff = np.abs(lepton_thetas[1] - lepton_thetas[0])
		lepton_phi_diff = np.abs(entry.lep_phi[1] - entry.lep_phi[0])
		lepton_angular_dist = np.sqrt(lepton_theta_diff ** 2 + lepton_phi_diff ** 2)
		
		# Invariant mass of lepton pair
		lepton_x_momentum = [pT * np.cos(phi) for pT, phi in zip(entry.lep_pt, entry.lep_phi)]
		lepton_y_momentum = [pT * np.sin(phi) for pT, phi in zip(entry.lep_pt, entry.lep_phi)]
		lepton_z_momentum = [pT * np.sinh(eta) for pT, eta in zip(entry.lep_pt, entry.lep_eta)]
		
		lepton_inv_mass = np.sqrt(sum(entry.lep_E) ** 2 - (
				sum(lepton_x_momentum) ** 2 + sum(lepton_y_momentum) ** 2 + sum(lepton_z_momentum) ** 2))
		
		# Jet metrics
		jet_pt = {f"jet_pt_{i}": value for i, value in enumerate(entry.jet_pt)}
		jet_theta = {f"jet_eta_{i}": 2 * np.arctan(np.exp(-value)) for i, value in enumerate(entry.jet_eta)}
		jet_phi = {f"jet_phi_{i}": value for i, value in enumerate(entry.jet_phi)}
		jet_energy = {f"lepton_energy_{i}": value for i, value in enumerate(entry.jet_E)}
		# One hot encode jet types
		jet_type = {f"jet_type_{i}": list(set(entry.jet_trueflav)).index(value) for i, value in
		            enumerate(entry.jet_trueflav)}
		
		# Invariant mass of the system
		jet_x_momentum = [pT * np.cos(phi) for pT, phi in zip(entry.jet_pt, entry.jet_phi)]
		jet_y_momentum = [pT * np.sin(phi) for pT, phi in zip(entry.jet_pt, entry.jet_phi)]
		jet_z_momentum = [pT * np.sinh(eta) for pT, eta in zip(entry.jet_pt, entry.jet_eta)]
		
		sys_inv_mass = np.sqrt(
			(sum(entry.lep_E) + sum(entry.jet_E)) ** 2
			- (sum(lepton_x_momentum + jet_x_momentum) ** 2 + sum(lepton_y_momentum + jet_y_momentum) ** 2
			   + sum(lepton_z_momentum + jet_z_momentum) ** 2)
		)
		
		pid = {f"pid_{i}": value for i, value in enumerate(
			[*[1] * entry.lep_n, *[-1] * entry.jet_n]
		)}
		
		event = {
			k: v for k, v in
			zip([*lepton_type.keys(), *lepton_phi.keys(), *lepton_eta.keys(), *lepton_pt.keys(),
			     *lepton_charge.keys(), *jet_pt.keys(), *jet_theta.keys(), *jet_phi.keys(), *jet_type.keys(),
			     *jet_energy.keys(), *pid.keys()],
			    
			    [*lepton_type.values(), *lepton_phi.values(), *lepton_eta.values(), *lepton_pt.values(),
			     *lepton_charge.values(), *jet_pt.values(), *jet_theta.values(), *jet_phi.values(), *jet_type.values(),
			     *jet_energy.values(), *pid.values()])
		}
		
		event["id"] = i
		event["lepton_theta_diff"] = lepton_theta_diff
		event["lepton_phi_diff"] = lepton_phi_diff
		event["lepton_angular_dist"] = lepton_angular_dist
		event["lepton_inv_mass"] = lepton_inv_mass
		event["sys_inv_mass"] = sys_inv_mass
		event["met_et"] = entry.met_et
		event["met_phi"] = entry.met_phi
		
		if label is not None:
			event["label"] = label
		
		events.append(event)
	
	return events


if __name__ == "__main__":
	file_names: list[str] = ["mc_345324.ggH125_WW2lep.2lep.root", "mc_363492.llvv.2lep.root"]
	data_label_mapping: list[int] = [1, 0]  # 0 => Background, 1 => Higgs event
	output_name: str = "event_dataset.csv"
	
	processed_data: list[dict] = []
	
	for file_name, label in zip(file_names, data_label_mapping):
		processed_data = [*processed_data, *process_root_data("Data/Raw Data/" + file_name.strip(), limit=100000,
		                                                      show_iteration_count=True, label=label)]
		
		print(f"\nProcessed {file_name.strip()}...\n")
	
	print("\nWriting to CSV...")
	df: pl.DataFrame = pl.DataFrame(processed_data)
	df.write_csv(output_path := "Data/Processed Data/" + output_name)
	print(f"Saved to {output_path}")
