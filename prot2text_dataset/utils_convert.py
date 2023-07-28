import numpy as np
from biopandas.pdb import PandasPdb

pdb_order = [
    "record_name",
    "atom_number",
    "blank_1",
    "atom_name",
    "alt_loc",
    "residue_name",
    "blank_2",
    "chain_id",
    "residue_number",
    "insertion",
    "blank_3",
    "x_coord",
    "y_coord",
    "z_coord",
    "occupancy",
    "b_factor",
    "blank_4",
    "segment_id",
    "element_symbol",
    "charge",
    "line_idx",
]
mmcif_read = {
    "group_PDB": "record_name",
    "id": "atom_number",
    "auth_atom_id": "atom_name",
    "auth_comp_id": "residue_name",
    "auth_asym_id": "chain_id",
    "auth_seq_id": "residue_number",
    "Cartn_x": "x_coord",
    "Cartn_y": "y_coord",
    "Cartn_z": "z_coord",
    "occupancy": "occupancy",
    "B_iso_or_equiv": "b_factor",
    "type_symbol": "element_symbol",
}

nonefields = [
    "blank_1",
    "alt_loc",
    "blank_2",
    "insertion",
    "blank_3",
    "blank_4",
    "segment_id",
    "charge",
    "line_idx",
]


def biopandas_mmcif2pdb(pandasmmcif, model_index = 1):
    """
    Converts the ATOM and HETATM dataframes of PandasMmcif() to PandasPdb() format.
    """
    pandaspdb = PandasPdb()
    for a in ["ATOM", "HETATM"]:
        dfa = pandasmmcif.df[a]
        dfa = dfa.loc[dfa.pdbx_PDB_model_num == model_index]
        if a =='ATOM':
            if len(dfa) == 0:
                raise ValueError(f"No model found for index: {model_index}")
        # keep only those fields found in pdb
        dfa = dfa[mmcif_read.keys()]
        # rename fields
        dfa = dfa.rename(columns=mmcif_read)
        # add empty fields
        for i in nonefields:
            dfa[i] = ""
        dfa["charge"] = np.nan
        # reorder columns to PandasPdb order
        dfa = dfa[pdb_order]
        pandaspdb.df[a] = dfa

    # update line_idx
    pandaspdb.df["ATOM"]["line_idx"] = pandaspdb.df["ATOM"].index.values
    pandaspdb.df["HETATM"]["line_idx"] = pandaspdb.df["HETATM"].index

    return pandaspdb