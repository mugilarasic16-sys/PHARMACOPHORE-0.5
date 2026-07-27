"""
Pharmacophore Modeling Pipeline v2 (core, open-source only)
Run: streamlit run pharmacophore_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, ChemicalFeatures, Draw, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDConfig
import os, io

st.set_page_config(page_title="Pharmacophore Pipeline", layout="wide")
FDEF = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
FACTORY = ChemicalFeatures.BuildFeatureFactory(FDEF)
FAMILY_COLOR = {"Donor": (0.2,0.4,1), "Acceptor": (1,0.2,0.2), "Hydrophobe": (0.2,0.8,0.2),
                "Aromatic": (1,0.6,0), "PosIonizable": (0.6,0,0.8), "NegIonizable": (0.5,0.3,0.1),
                "LumpedHydrophobe": (0.2,0.8,0.2)}
FRAGMENTS = {  # dummy '*' marks attachment point
    "Hydroxyl (-OH)": "*O", "Amino (-NH2)": "*N", "Methyl (-CH3)": "*C",
    "Methoxy (-OCH3)": "*OC", "Carboxyl (-COOH)": "*C(=O)O", "Fluoro (-F)": "*F",
    "Chloro (-Cl)": "*Cl", "Bromo (-Br)": "*Br", "Trifluoromethyl (-CF3)": "*C(F)(F)F",
    "Nitro (-NO2)": "*[N+](=O)[O-]", "Sulfonamide (-SO2NH2)": "*S(=O)(=O)N",
    "Phenyl": "*c1ccccc1", "Morpholine": "*N1CCOCC1", "Piperazine": "*N1CCNCC1",
}

# ---------------- core functions ----------------

def detect_fmt(filename):
    ext = filename.lower().rsplit(".", 1)[-1]
    return {"sdf": "sdf", "mol": "mol", "smi": "smi", "txt": "smi"}.get(ext, "smi")

def load_mol(file_bytes, fmt):
    text = file_bytes.decode(errors="ignore")
    if fmt == "sdf":
        supp = Chem.SDMolSupplier(); supp.SetData(text)
        return [m for m in supp if m is not None]
    if fmt == "mol":
        m = Chem.MolFromMolBlock(text)
        return [m] if m else []
    mols = []
    for line in text.strip().splitlines():
        parts = line.split()
        if not parts: continue
        m = Chem.MolFromSmiles(parts[0])
        if m: m.SetProp("_Name", parts[1] if len(parts) > 1 else parts[0]); mols.append(m)
    return mols

def embed_3d(mol):
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=0xf00d, useRandomCoords=True) < 0:
        return None
    try: AllChem.MMFFOptimizeMolecule(m)
    except Exception: pass
    return m

def extract_features(mol):
    feats = FACTORY.GetFeaturesForMol(mol)
    out = []
    for f in feats:
        pos = f.GetPos()
        out.append({"family": f.GetFamily(), "x": pos.x, "y": pos.y, "z": pos.z,
                     "atom_ids": list(f.GetAtomIds())})
    return out

def draw_2d_pharmacophore(mol, feats):
    """2D depiction with pharmacophore-contributing atoms highlighted by family color."""
    m2d = Chem.Mol(mol)
    try: Chem.rdDepictor.Compute2DCoords(m2d)
    except Exception: pass
    hl_atoms, hl_colors = [], {}
    for f in feats:
        col = FAMILY_COLOR.get(f["family"], (0.6,0.6,0.6))
        for a in f["atom_ids"]:
            hl_atoms.append(a); hl_colors[a] = col
    d = rdMolDraw2D.MolDraw2DCairo(450, 450)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m2d, highlightAtoms=hl_atoms, highlightAtomColors=hl_colors)
    d.FinishDrawing()
    return d.GetDrawingText()

def render_3d_pharmacophore(mol3d, feats, height=420):
    """py3Dmol viewer: sticks + colored spheres at feature centroids."""
    import py3Dmol
    molblock = Chem.MolToMolBlock(mol3d)
    view = py3Dmol.view(width=500, height=height)
    view.addModel(molblock, "mol")
    view.setStyle({"stick": {}})
    for f in feats:
        c = FAMILY_COLOR.get(f["family"], (0.6,0.6,0.6))
        hexcol = '#%02x%02x%02x' % tuple(int(255*x) for x in c)
        view.addSphere({"center": {"x": f["x"], "y": f["y"], "z": f["z"]},
                         "radius": 0.5, "color": hexcol, "alpha": 0.75})
    view.zoomTo()
    return view

def mcs_align(mols):
    if len(mols) < 2: return mols, [0.0]
    mcs = rdFMCS.FindMCS(mols, timeout=30, ringMatchesRingOnly=True)
    if mcs.numAtoms < 3: return mols, [None]*len(mols)
    patt = Chem.MolFromSmarts(mcs.smartsString)
    ref = mols[0]; ref_match = ref.GetSubstructMatch(patt)
    aligned, rmsds = [ref], [0.0]
    for m in mols[1:]:
        match = m.GetSubstructMatch(patt)
        if not match or not ref_match: aligned.append(m); rmsds.append(None); continue
        try: rmsds.append(round(AllChem.AlignMol(m, ref, atomMap=list(zip(match, ref_match))), 3))
        except Exception: rmsds.append(None)
        aligned.append(m)
    return aligned, rmsds

def consensus_pharmacophore(all_feats, tol=1.5):
    flat = [{**f, "mol_idx": i} for i, feats in enumerate(all_feats) for f in feats]
    used = [False]*len(flat); clusters = []
    for i, fi in enumerate(flat):
        if used[i]: continue
        cluster = [fi]; used[i] = True
        for j in range(i+1, len(flat)):
            if used[j] or flat[j]["family"] != fi["family"]: continue
            d = np.linalg.norm([flat[j]["x"]-fi["x"], flat[j]["y"]-fi["y"], flat[j]["z"]-fi["z"]])
            if d <= tol: cluster.append(flat[j]); used[j] = True
        n_mols = len(set(c["mol_idx"] for c in cluster))
        clusters.append({"family": fi["family"], "x": np.mean([c["x"] for c in cluster]),
                          "y": np.mean([c["y"] for c in cluster]), "z": np.mean([c["z"] for c in cluster]),
                          "n_ligands_supporting": n_mols, "frequency": round(n_mols/len(all_feats), 2)})
    return sorted(clusters, key=lambda c: -c["frequency"])

def pdb_pocket_features(pdb_text, ligand_resname=None, radius=6.0):
    from Bio.PDB import PDBParser, NeighborSearch
    struct = PDBParser(QUIET=True).get_structure("prot", io.StringIO(pdb_text))
    atoms = list(struct.get_atoms())
    het = [a for a in atoms if a.get_parent().id[0].strip() != "" and
           (ligand_resname is None or a.get_parent().resname == ligand_resname)]
    if not het: return [], "No HETATM ligand found; provide resname or use ligand-based mode."
    ns = NeighborSearch(atoms); pocket_atoms = set()
    for ha in het:
        for near in ns.search(ha.coord, radius):
            if near.get_parent().id[0].strip() == "": pocket_atoms.add(near)
    donor_n = {"N","ND1","ND2","NE","NE1","NE2","NZ","NH1","NH2","OG","OG1","OH"}
    acceptor_n = {"O","OD1","OD2","OE1","OE2","OXT"}
    hydrophobic_res = {"ALA","VAL","LEU","ILE","PHE","TRP","MET","PRO"}
    feats = []
    for a in pocket_atoms:
        name, resn = a.get_name(), a.get_parent().resname
        fam = "Donor" if name in donor_n else "Acceptor" if name in acceptor_n else \
              ("Hydrophobe" if resn in hydrophobic_res and name == "CA" else None)
        if fam:
            c = a.coord
            feats.append({"family": fam, "x": float(c[0]), "y": float(c[1]), "z": float(c[2]),
                          "residue": f"{resn}{a.get_parent().id[1]}"})
    return feats, None

def compare_models(a, b, tol=1.5):
    matched, used_b = [], [False]*len(b)
    for fa in a:
        for j, fb in enumerate(b):
            if used_b[j] or fb["family"] != fa["family"]: continue
            if np.linalg.norm([fa["x"]-fb["x"], fa["y"]-fb["y"], fa["z"]-fb["z"]]) <= tol:
                matched.append((fa, fb)); used_b[j] = True; break
    unmatched_a = [fa for fa in a if fa not in [m[0] for m in matched]]
    unmatched_b = [fb for j, fb in enumerate(b) if not used_b[j]]
    total = len(a) + len(b)
    return {"matched": matched, "unique_to_a": unmatched_a, "unique_to_b": unmatched_b,
            "overlap_pct": round(200*len(matched)/total, 1) if total else 0.0}

def attach_group(mol, atom_idx, frag_smiles):
    """Attach a functional group at atom_idx by consuming an implicit H valence slot."""
    frag = Chem.MolFromSmiles(frag_smiles)
    combo = Chem.RWMol(Chem.CombineMols(mol, frag))
    offset = mol.GetNumAtoms()
    dummy_idx = attach_to = None
    for atom in combo.GetAtoms():
        if atom.GetIdx() >= offset and atom.GetAtomicNum() == 0:
            dummy_idx = atom.GetIdx()
            nbrs = atom.GetNeighbors()
            attach_to = nbrs[0].GetIdx() if nbrs else None
    if dummy_idx is None or attach_to is None: return None
    combo.AddBond(atom_idx, attach_to, Chem.BondType.SINGLE)
    combo.RemoveAtom(dummy_idx)
    m = combo.GetMol()
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m

def activity_heuristic(base_feats, mod_feats, base_mol, mod_mol, pocket_feats=None, tol=1.5):
    """Rule-based (NOT a validated ML model) activity direction estimate."""
    reasons = []
    score = 0
    fam_before = {f: sum(1 for x in base_feats if x["family"]==f) for f in FAMILY_COLOR}
    fam_after = {f: sum(1 for x in mod_feats if x["family"]==f) for f in FAMILY_COLOR}
    for fam in ["Donor","Acceptor","Hydrophobe","Aromatic"]:
        d = fam_after[fam]-fam_before[fam]
        if d != 0:
            reasons.append(f"{fam} count changed by {d:+d}")
            score += d if pocket_feats is None else 0  # generic scoring only if no pocket ref
    if pocket_feats is not None:
        before_match = compare_models(base_feats, pocket_feats, tol)["overlap_pct"]
        after_match = compare_models(mod_feats, pocket_feats, tol)["overlap_pct"]
        score += (after_match - before_match) / 10
        reasons.append(f"Pocket feature overlap: {before_match}% -> {after_match}%")
    dlogp = Descriptors.MolLogP(mod_mol) - Descriptors.MolLogP(base_mol)
    dtpsa = Descriptors.TPSA(mod_mol) - Descriptors.TPSA(base_mol)
    dmw = Descriptors.MolWt(mod_mol) - Descriptors.MolWt(base_mol)
    reasons.append(f"ΔLogP={dlogp:+.2f}, ΔTPSA={dtpsa:+.1f}, ΔMolWt={dmw:+.1f}")
    lip_before = sum([Descriptors.MolWt(base_mol) > 500, Descriptors.MolLogP(base_mol) > 5,
                       Descriptors.NumHDonors(base_mol) > 5, Descriptors.NumHAcceptors(base_mol) > 10])
    lip_after = sum([Descriptors.MolWt(mod_mol) > 500, Descriptors.MolLogP(mod_mol) > 5,
                      Descriptors.NumHDonors(mod_mol) > 5, Descriptors.NumHAcceptors(mod_mol) > 10])
    if lip_after > lip_before:
        score -= 1; reasons.append(f"Lipinski violations increased ({lip_before}->{lip_after})")
    verdict = "Likely INCREASE" if score > 0.5 else "Likely DECREASE" if score < -0.5 else "Uncertain / minimal change"
    return verdict, reasons

# ---------------- UI ----------------
st.title("Pharmacophore Modeling Pipeline")
st.caption("Heuristic rule-based activity estimation — NOT a validated ML/QSAR model. Always confirm with docking/experimental data before claiming SAR trends.")

mode = st.sidebar.radio("Workflow", ["Ligand-Based", "Target-Based", "Compare Models", "ML on Activity Data"])
for k in ["lig_consensus", "pocket_feats", "lig_store"]:
    if k not in st.session_state: st.session_state[k] = None

# ---- Ligand-based ----
if mode == "Ligand-Based":
    st.header("Ligand-Based Pharmacophore")
    ups = st.file_uploader("Upload ligand file(s) — SDF/MOL/SMI, multiple allowed", type=["sdf","mol","smi","txt"], accept_multiple_files=True)
    if ups:
        all_mols, names = [], []
        for up in ups:
            fmt = detect_fmt(up.name)
            for i, m in enumerate(load_mol(up.read(), fmt)):
                all_mols.append(m); names.append(f"{up.name}_{i}")
        st.write(f"Parsed {len(all_mols)} molecule(s) from {len(ups)} file(s).")
        with st.spinner("Embedding 3D + MMFF..."):
            mols_3d, names_3d = [], []
            for m, n in zip(all_mols, names):
                e = embed_3d(m)
                if e is not None: mols_3d.append(e); names_3d.append(n)
        st.write(f"{len(mols_3d)} embedded successfully.")
        if mols_3d:
            if len(mols_3d) >= 2:
                aligned, rmsds = mcs_align(mols_3d)
                st.subheader("MCS Alignment RMSDs")
                st.dataframe(pd.DataFrame({"molecule": names_3d, "rmsd_to_ref": rmsds}))
            else:
                aligned = mols_3d
            all_feats = [extract_features(m) for m in aligned]
            st.session_state.lig_store = {"mols": aligned, "names": names_3d, "feats": all_feats}
            consensus = consensus_pharmacophore(all_feats)
            st.session_state.lig_consensus = consensus
            st.subheader("Consensus Pharmacophore (table)")
            st.dataframe(pd.DataFrame(consensus))
            st.download_button("Download consensus CSV", pd.DataFrame(consensus).to_csv(index=False), "consensus.csv")

            st.subheader("2D / 3D Pharmacophore Viewer")
            sel = st.selectbox("Select molecule", names_3d)
            idx = names_3d.index(sel)
            c1, c2 = st.columns(2)
            with c1:
                st.write("**2D — highlighted pharmacophore features**")
                png = draw_2d_pharmacophore(aligned[idx], all_feats[idx])
                st.image(png)
            with c2:
                st.write("**3D — sticks + feature spheres**")
                view = render_3d_pharmacophore(aligned[idx], all_feats[idx])
                st.components.v1.html(view._make_html(), height=440)

            st.subheader("Functional Group Modification -> Activity Direction (heuristic)")
            base_mol_noH = Chem.RemoveHs(aligned[idx])
            atom_opts = [f"{a.GetIdx()}: {a.GetSymbol()}" for a in base_mol_noH.GetAtoms() if a.GetTotalNumHs() > 0]
            atom_choice = st.selectbox("Attachment atom (must have an available H)", atom_opts)
            group_choice = st.selectbox("Functional group to attach", list(FRAGMENTS.keys()))
            if st.button("Predict activity change"):
                atom_idx = int(atom_choice.split(":")[0])
                new_mol = attach_group(base_mol_noH, atom_idx, FRAGMENTS[group_choice])
                if new_mol is None:
                    st.error("Could not attach group at that position — try a different atom.")
                else:
                    new_3d = embed_3d(new_mol)
                    if new_3d is None:
                        st.error("3D embedding failed for modified molecule.")
                    else:
                        new_feats = extract_features(new_3d)
                        verdict, reasons = activity_heuristic(all_feats[idx], new_feats, aligned[idx], new_3d,
                                                               pocket_feats=st.session_state.pocket_feats)
                        st.metric("Predicted activity direction", verdict)
                        for r in reasons: st.write("- " + r)
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.write("Original")
                            st.image(draw_2d_pharmacophore(aligned[idx], all_feats[idx]))
                        with cc2:
                            st.write("Modified")
                            st.image(draw_2d_pharmacophore(new_3d, new_feats))

# ---- Target-based ----
elif mode == "Target-Based":
    st.header("Target-Based Pharmacophore (pocket, heuristic)")
    ups = st.file_uploader("Upload PDB file(s), multiple allowed", type=["pdb"], accept_multiple_files=True)
    resname = st.text_input("Ligand HETATM resname (blank = auto-detect any)", "")
    radius = st.slider("Pocket radius (Å)", 4.0, 10.0, 6.0)
    if ups:
        for up in ups:
            text = up.read().decode(errors="ignore")
            feats, err = pdb_pocket_features(text, ligand_resname=resname or None, radius=radius)
            st.subheader(up.name)
            if err:
                st.error(err); continue
            st.write(f"{len(feats)} pocket-derived pharmacophore points.")
            st.dataframe(pd.DataFrame(feats))
            st.session_state.pocket_feats = feats  # last uploaded target becomes active reference
            st.download_button(f"Download {up.name} pocket CSV", pd.DataFrame(feats).to_csv(index=False),
                                f"{up.name}_pocket.csv", key=up.name)
        st.info("Heuristic: donor/acceptor by atom-name lookup, hydrophobe by residue type at CA — good for early triage, not final SAR claims. Last uploaded file becomes the active pocket reference for Compare/activity-heuristic tabs.")

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
        c1.write("Matched features"); c1.dataframe(pd.DataFrame([m[0] for m in result["matched"]]))
        c2.write("Unique to ligand model"); c2.dataframe(pd.DataFrame(result["unique_to_a"]))
        st.write("Unique to pocket model"); st.dataframe(pd.DataFrame(result["unique_to_b"]))

# ---- ML ----
elif mode == "ML on Activity Data":
    st.header("Activity Prediction (RandomForest baseline, trained on YOUR data)")
    st.caption("Upload CSV: SMILES column + activity column. This is a real trained model, distinct from the rule-based heuristic above — only use this claim if you actually run it on your own labeled dataset.")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up:
        df = pd.read_csv(up)
        st.dataframe(df.head())
        smi_col = st.selectbox("SMILES column", df.columns)
        act_col = st.selectbox("Activity column", df.columns)
        task = st.radio("Task", ["classification", "regression"])
        if st.button("Train model"):
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, r2_score
            def desc(smi):
                m = Chem.MolFromSmiles(smi)
                if m is None: return None
                return [Descriptors.MolWt(m), Descriptors.MolLogP(m), Descriptors.NumHDonors(m),
                        Descriptors.NumHAcceptors(m), Descriptors.TPSA(m), Descriptors.NumRotatableBonds(m)]
            X, y, dropped = [], [], 0
            for smi, act in zip(df[smi_col], df[act_col]):
                d = desc(str(smi))
                if d is None: dropped += 1; continue
                X.append(d); y.append(act)
            st.write(f"{len(X)} usable rows, {dropped} dropped.")
            X, y = np.array(X), np.array(y)
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            if task == "classification":
                clf = RandomForestClassifier(n_estimators=300, random_state=42).fit(Xtr, ytr)
                st.metric("Test accuracy", round(accuracy_score(yte, clf.predict(Xte)), 3)); model = clf
            else:
                reg = RandomForestRegressor(n_estimators=300, random_state=42).fit(Xtr, ytr)
                st.metric("Test R²", round(r2_score(yte, reg.predict(Xte)), 3)); model = reg
            fi = pd.DataFrame({"descriptor": ["MolWt","LogP","HBD","HBA","TPSA","RotBonds"],
                                "importance": model.feature_importances_}).sort_values("importance", ascending=False)
            st.bar_chart(fi.set_index("descriptor"))
