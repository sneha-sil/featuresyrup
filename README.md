# featuresyrup

featuresyrup is a catalyst-specific modification of graphpancake (Sil 2025) designed for the featurization of N-heterocyclic carbene (NHC) organocatalysts. The additional features on top of graphpancake includes: NMR shieldings, IR intensities, FMO energies, Morfeus descriptors, ACSF, SOAP, and Fukui indices. 

## Installation

### Recommended: conda environment

```bash
conda env create -f environment.yml
conda activate featuresyrup-env
pip install -e .
```

### Pip installation

```bash
pip install featuresyrup
```

## Quick start

```bash
featuresyrup create-db featuresyrup.db

featuresyrup load-data \
  --database featuresyrup.db \
  --mol-id benzaldehyde_nhc \
  --smiles "C1=CC=C(C=C1)C[NH]C2=CC=CC=C2" \
  --xyz-file benzaldehyde.xyz \
  --shermo-output benzaldehyde_shermo.out \
  --neutral-output benzaldehyde_neutral.out \
  --cationic-output benzaldehyde_cation.out \
  --anionic-output benzaldehyde_anion.out \
  --neutral-nbo benzaldehyde_neutral_nbo.out \
  --cationic-nbo benzaldehyde_cation_nbo.out \
  --anionic-nbo benzaldehyde_anion_nbo.out \
  --nmr-output benzaldehyde_nmr.out \
  --IR-output benzaldehyde_ir.out \
  --dipole-polarizability-output benzaldehyde_dipole.out \
  --homo-lumo-output benzaldehyde_homo_lumo.out

featuresyrup query --database featuresyrup.db
```

Batch processing uses the same catalyst contract:

```bash
python -m featuresyrup.batch_processing --config config.yaml
```

## Documentation

The full user guide and API reference live in the docs folder and are built with Sphinx. The documentation is aligned with the catalyst-only pipeline used by the package.

## Dependencies to cite

- Sil, S., Maskeri, M. A., and Scheidt, K. A. graphpancake: A Python package for representing organic molecules as molecular graphs utilizing electronic structure theory. J. Cheminform. 18, 61 (2026). DOI: 10.5281/zenodo.17553385
- Jorner, K. morfeus: a Python package for calculating molecular features. Project documentation: https://github.com/digital-chemistry-laboratory/morfeus
- Himanen, L., Jäger, M. O. J., Vatanen, T., Eriksson, O., and Fellowes, C. DScribe: Library of descriptors for machine learning in materials science. Comput. Phys. Commun. 247, 106949 (2020). DOI: 10.1016/j.cpc.2019.106949
- Neese, F. et al. The ORCA quantum chemistry program package. J. Chem. Phys. 2020, 152, 224108
- Nikolaienko et al. JANPA: an open source cross-platform implementation of the Natural Population Analysis on the Java platform, Computational and Theoretical Chemistry 2014, 1050, 15-22, DOI: 10.1016/j.comptc.2014.10.002, http://janpa.sourceforge.net
- Glendening, E. D., Landis, C. R., Weinhold, F. NBO 7.0: New vistas in localized and delocalized chemical bonding theory. Journal of Computational Chemistry 2019, 40 (25), 2234-2241. https://doi.org/10.1002/jcc.25873
- Tian, L., Qinxue, C., Shermo: A general code for calculating molecular thermodynamic properties, Comput. Theor. Chem. 2021, 1200, 113249 DOI: 10.1016/j.comptc.2021.113249

## License

MIT License. See [LICENSE](LICENSE) for details.
