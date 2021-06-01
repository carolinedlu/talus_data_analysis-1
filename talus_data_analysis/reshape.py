def uniprot_protein_name(uniprot_str: str):
    return uniprot_str.split("|")[-1].split("_")[0]