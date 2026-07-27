streamlit>=1.32
rdkit>=2023.9.1
biopython>=1.83
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
openpyxl>=3.1
"""
Pharmacophore Modeling Pipeline (core, open-source only)
Run: streamlit run pharmacophore_app.py
Deps: streamlit rdkit biopython py3Dmol stmol pandas numpy scikit-learn openpyxl
"""
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, ChemicalFeatures
from rdkit import RDConfig
import os, io, itertools
 
st.set_page_config(page_title="Pharmacophore Pipeline", layout="wide")
FDEF = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
FACTORY = ChemicalFeatures.BuildFeatureFactory(FDEF)
 
# ---------------- core functions ----------------
 
def load_mol(file_bytes, fmt):
    """Parse ligand from bytes given format string."""
    text = file_bytes.decode(errors="ignore")
    if fmt == "sdf":
        supp = Chem.SDMolSupplier()
        supp.SetData(text)
        return [m for m in supp if m is not None]
    if fmt == "mol":
        m = Chem.MolFromMolBlock(text)
        return [m] if m else []
    if fmt == "smi":
        mols = []
        for line in text.strip().splitlines():
            parts = line.split()
            if not parts: continue
            m = Chem.MolFromSmiles(parts[0])
            if m: mols.append(m)
        return mols
    return []
 
def embed_3d(mol):
    """Add Hs, embed ETKDG, MMFF optimize. Returns mol or None."""
    m = Chem.AddHs(mol)
    cid = AllChem.EmbedMolecule(m, randomSeed=0xf00d, useRandomCoords=True)
    if cid < 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        pass
    return m
 
def extract_features(mol):
    """Return list of (family, x, y, z) using RDKit BaseFeatures.fdef."""
    feats = FACTORY.GetFeaturesForMol(mol)
    out = []
    for f in feats:
        pos = f.GetPos()
        out.append({"family": f.GetFamily(), "x": pos.x, "y": pos.y, "z": pos.z})
    return out
 
def mcs_align(mols):
    """Align all mols to mol[0] via MCS substructure match. Returns aligned copies + RMSDs."""
    if len(mols) < 2:
        return mols, [0.0]
    mcs = rdFMCS.FindMCS(mols, timeout=30, ringMatchesRingOnly=True)
    if mcs.numAtoms < 3:
        return mols, [None] * len(mols)
    patt = Chem.MolFromSmarts(mcs.smartsString)
    ref = mols[0]
    ref_match = ref.GetSubstructMatch(patt)
    aligned, rmsds = [ref], [0.0]
    for m in mols[1:]:
        match = m.GetSubstructMatch(patt)
        if not match or not ref_match:
            aligned.append(m); rmsds.append(None); continue
        atom_map = list(zip(match, ref_match))
        try:
            rmsd = AllChem.AlignMol(m, ref, atomMap=atom_map)
            rmsds.append(round(rmsd, 3))
        except Exception:
            rmsds.append(None)
        aligned.append(m)
    return aligned, rmsds
 
def consensus_pharmacophore(all_feats, tol=1.5):
    """Cluster features across ligands by family+proximity. Returns consensus list w/ frequency."""
    flat = []
    for i, feats in enumerate(all_feats):
        for f in feats:
            flat.append({**f, "mol_idx": i})
    used = [False] * len(flat)
    clusters = []
    for i, fi in enumerate(flat):
        if used[i]: continue
        cluster = [fi]; used[i] = True
        for j in range(i + 1, len(flat)):
            if used[j] or flat[j]["family"] != fi["family"]: continue
            d = np.linalg.norm([flat[j]["x"]-fi["x"], flat[j]["y"]-fi["y"], flat[j]["z"]-fi["z"]])
            if d <= tol:
                cluster.append(flat[j]); used[j] = True
        xs = np.mean([c["x"] for c in cluster]); ys = np.mean([c["y"] for c in cluster]); zs = np.mean([c["z"] for c in cluster])
        n_mols = len(set(c["mol_idx"] for c in cluster))
        clusters.append({"family": fi["family"], "x": xs, "y": ys, "z": zs,
                          "n_ligands_supporting": n_mols, "frequency": round(n_mols/len(all_feats), 2)})
    return sorted(clusters, key=lambda c: -c["frequency"])
 
def pdb_pocket_features(pdb_text, ligand_resname=None, radius=6.0):
    """Naive pocket detection: residues within `radius` A of a HETATM ligand (or all HETATMs if resname unset).
    Extracts pharmacophore-like points from those residue atoms by simple heuristic (not a full docking-grade method)."""
    from Bio.PDB import PDBParser, NeighborSearch
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("prot", io.StringIO(pdb_text))
    atoms = list(struct.get_atoms())
    het_atoms = [a for a in atoms if a.get_parent().id[0].strip() != "" and
                 (ligand_resname is None or a.get_parent().resname == ligand_resname)]
    if not het_atoms:
        return [], "No HETATM ligand found in file; cannot auto-detect pocket. Provide ligand_resname or use ligand-based mode."
    ns = NeighborSearch(atoms)
    pocket_atoms = set()
    for ha in het_atoms:
        for near in ns.search(ha.coord, radius):
            if near.get_parent().id[0].strip() == "":  # protein residue only
                pocket_atoms.add(near)
    donor_names = {"N", "ND1", "ND2", "NE", "NE1", "NE2", "NZ", "NH1", "NH2", "OG", "OG1", "OH"}
    acceptor_names = {"O", "OD1", "OD2", "OE1", "OE2", "OXT"}
    hydrophobic_res = {"ALA","VAL","LEU","ILE","PHE","TRP","MET","PRO"}
    feats = []
    for a in pocket_atoms:
        name = a.get_name()
        resn = a.get_parent().resname
        fam = None
        if name in donor_names: fam = "Donor"
        elif name in acceptor_names: fam = "Acceptor"
        elif resn in hydrophobic_res and name == "CA": fam = "Hydrophobe"
        if fam:
            c = a.coord
            feats.append({"family": fam, "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
                          "residue": f"{resn}{a.get_parent().id[1]}"})
    return feats, None
 
def compare_models(model_a, model_b, tol=1.5):
    """Compare two consensus/pocket pharmacophore feature lists. Returns match stats."""
    matched, unmatched_a = [], []
    used_b = [False]*len(model_b)
    for fa in model_a:
        hit = False
        for j, fb in enumerate(model_b):
            if used_b[j] or fb["family"] != fa["family"]: continue
            d = np.linalg.norm([fa["x"]-fb["x"], fa["y"]-fb["y"], fa["z"]-fb["z"]])
            if d <= tol:
                matched.append((fa, fb)); used_b[j] = True; hit = True; break
        if not hit: unmatched_a.append(fa)
    unmatched_b = [fb for j, fb in enumerate(model_b) if not used_b[j]]
    total = len(model_a) + len(model_b)
    overlap_pct = round(200 * len(matched) / total, 1) if total else 0.0
    return {"matched": matched, "unique_to_a": unmatched_a, "unique_to_b": unmatched_b, "overlap_pct": overlap_pct}
 
# ---------------- UI ----------------
 
st.title("Pharmacophore Modeling Pipeline")
st.caption("Open-source core build: RDKit + BioPython + scikit-learn. No proprietary tools.")
 
mode = st.sidebar.radio("Workflow", ["Ligand-Based", "Target-Based", "Compare Models", "ML on Activity Data"])
 
if "lig_consensus" not in st.session_state: st.session_state.lig_consensus = None
if "pocket_feats" not in st.session_state: st.session_state.pocket_feats = None
 
# ---- Ligand-based ----
if mode == "Ligand-Based":
    st.header("Ligand-Based Pharmacophore")
    fmt = st.selectbox("Input format", ["sdf", "mol", "smi"])
    up = st.file_uploader("Upload ligand file", type=["sdf", "mol", "smi", "txt"])
    if up:
        mols_raw = load_mol(up.read(), fmt)
        st.write(f"Parsed {len(mols_raw)} molecule(s).")
        if mols_raw:
            with st.spinner("Embedding 3D + MMFF optimizing..."):
                mols_3d = [embed_3d(m) for m in mols_raw]
                mols_3d = [m for m in mols_3d if m is not None]
            st.write(f"{len(mols_3d)} embedded successfully.")
            if len(mols_3d) >= 2:
                aligned, rmsds = mcs_align(mols_3d)
                st.subheader("MCS Alignment RMSDs")
                st.write(pd.DataFrame({"mol_idx": range(len(rmsds)), "rmsd_to_ref": rmsds}))
            else:
                aligned = mols_3d
            all_feats = [extract_features(m) for m in aligned]
            for i, f in enumerate(all_feats):
                st.write(f"Mol {i}: {len(f)} pharmacophore points ({', '.join(sorted(set(x['family'] for x in f)))})")
            consensus = consensus_pharmacophore(all_feats)
            st.session_state.lig_consensus = consensus
            st.subheader("Consensus Pharmacophore")
            df = pd.DataFrame(consensus)
            st.dataframe(df)
            st.download_button("Download consensus CSV", df.to_csv(index=False), "consensus_pharmacophore.csv")
 
# ---- Target-based ----
elif mode == "Target-Based":
    st.header("Target-Based Pharmacophore (pocket, heuristic)")
    up = st.file_uploader("Upload PDB file", type=["pdb"])
    resname = st.text_input("Ligand HETATM resname (blank = auto-detect any)", "")
    radius = st.slider("Pocket radius (Å)", 4.0, 10.0, 6.0)
    if up:
        text = up.read().decode(errors="ignore")
        feats, err = pdb_pocket_features(text, ligand_resname=resname or None, radius=radius)
        if err:
            st.error(err)
        else:
            st.write(f"{len(feats)} pocket-derived pharmacophore points.")
            df = pd.DataFrame(feats)
            st.dataframe(df)
            st.session_state.pocket_feats = feats
            st.download_button("Download pocket pharmacophore CSV", df.to_csv(index=False), "pocket_pharmacophore.csv")
            st.info("Heuristic method: donor/acceptor by atom-name lookup, hydrophobe by residue type at CA. Not a substitute for LigPlot/ProLIF-grade interaction analysis — good enough for early-stage screening triage, not final SAR claims.")
 
# ---- Compare ----
elif mode == "Compare Models":
    st.header("Compare Ligand-Based vs Target-Based")
    if not st.session_state.lig_consensus or not st.session_state.pocket_feats:
        st.warning("Run both Ligand-Based and Target-Based tabs first (session-persistent).")
    else:
        tol = st.slider("Match tolerance (Å)", 0.5, 3.0, 1.5)
        result = compare_models(st.session_state.lig_consensus, st.session_state.pocket_feats, tol)
        st.metric("Overlap %", result["overlap_pct"])
        c1, c2 = st.columns(2)
        c1.write("Matched features"); c1.write(pd.DataFrame([m[0] for m in result["matched"]]))
        c2.write("Unique to ligand model"); c2.write(pd.DataFrame(result["unique_to_a"]))
        st.write("Unique to pocket model"); st.write(pd.DataFrame(result["unique_to_b"]))
 
# ---- ML ----
elif mode == "ML on Activity Data":
    st.header("Activity Prediction (RandomForest baseline)")
    st.caption("Upload CSV: SMILES column + activity column (numeric IC50/pIC50 or binary active/inactive).")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        st.dataframe(df.head())
        smi_col = st.selectbox("SMILES column", df.columns)
        act_col = st.selectbox("Activity column", df.columns)
        task = st.radio("Task", ["classification", "regression"])
        if st.button("Train model"):
            from rdkit.Chem import Descriptors
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, r2_score
 
            def descriptors(smi):
                m = Chem.MolFromSmiles(smi)
                if m is None: return None
                return [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m),
                        Descriptors.NumHAcceptors(m), Descriptors.TPSA(m), Descriptors.NumRotatableBonds(m)]
 
            X, y, dropped = [], [], 0
            for smi, act in zip(df[smi_col], df[act_col]):
                d = descriptors(str(smi))
                if d is None: dropped += 1; continue
                X.append(d); y.append(act)
            st.write(f"{len(X)} usable rows, {dropped} dropped (invalid SMILES).")
            X = np.array(X); y = np.array(y)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            if task == "classification":
                clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
                acc = accuracy_score(yte, clf.predict(Xte))
                st.metric("Test accuracy", round(acc, 3))
                model = clf
            else:
                reg = RandomForestRegressor(n_estimators=300, random_state=42).fit(Xtr, ytr)
                r2 = r2_score(yte, reg.predict(Xte))
                st.metric("Test R²", round(r2, 3))
                model = reg
            fi = pd.DataFrame({"descriptor": ["MolWt","LogP","HBD","HBA","TPSA","RotBonds"],
                                "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            st.subheader("Feature importance")
            st.bar_chart(fi.set_index("descriptor"))
