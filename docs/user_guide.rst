featuresyrup user guide
=======================

featuresyrup is a catalyst-specific modification of graphpancake (Sil 2025) designed for the featurization of N-heterocyclic carbene (NHC) organocatalysts. The additional features on top of graphpancake includes: NMR shieldings, IR intensities, FMO energies, Morfeus descriptors, ACSF, SOAP, and Fukui indices. 

Installation
============

Recommended installation
------------------------

Create the conda environment and install the local package:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate featuresyrup-env
   pip install -e .

Pip installation
-----------------

.. code-block:: bash

   pip install featuresyrup

The catalyst workflow depends on RDKit, Morfeus, DScribe, and QMlib, in addition to the usual scientific Python stack.

Catalyst input contract
=======================

A single catalyst record should include the following files:

.. list-table:: Required inputs
   :header-rows: 1

   * - Input
     - Purpose
   * - ``.xyz`` geometry
     - Cartesian coordinates and atom identities
   * - Shermo output
     - Thermodynamic and vibrational quantities
   * - Neutral, cationic, and anionic ORCA outputs
     - HOMO/LUMO and ionization/electron-affinity data
   * - Neutral, cationic, and anionic NBO outputs
     - Fukui indices and orbital-derived catalyst descriptors
   * - NMR output
     - Shielding tensors and atom-wise shifts
   * - IR output
     - Vibrational frequency statistics
   * - Dipole/polarizability output
     - Dipole moment and isotropic polarizability

Core workflow
=============

Create a database
-----------------

.. code-block:: bash

   featuresyrup create-db featuresyrup.db

Load one catalyst
-----------------

.. code-block:: bash

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

Query stored graphs
-------------------

.. code-block:: bash

   featuresyrup query --database featuresyrup.db

Batch processing
================

Batch mode uses the same catalyst input contract and is intended for folders or archives containing many complete data sets.

.. code-block:: bash

   python -m featuresyrup.batch_processing --config config.yaml

Update ``config_template.yaml`` with the expected file locations, CSV labels, and output directory before running the batch job.

Feature overview
================

featuresyrup stores the catalyst record as a single flat graph shape. The main feature groups are:

- graph-level metadata, such as molecular mass, atom count, and charge
- node-level properties, such as atomic numbers, coordinates, NPA/NBO-derived values, Morfeus descriptors, DScribe descriptors, and Fukui indices
- edge-level properties, such as distances, bond orders, and orbital-derived bond metrics
- target-level properties, such as HOMO/LUMO, thermochemistry, IR statistics, dipole moment, and polarizability

Citations and dependencies
==========================

Please cite the upstream tools used by featuresyrup when you publish results. A concise bibliography is:

- Sil, S., Maskeri, M. A., and Scheidt, K. A. graphpancake: A Python package for representing organic molecules as molecular graphs utilizing electronic structure theory. J. Cheminform. 18, 61 (2026). DOI: 10.5281/zenodo.17553385
- Jorner, K. morfeus: a Python package for calculating molecular features. Project documentation: https://github.com/digital-chemistry-laboratory/morfeus
- Himanen, L., Jäger, M. O. J., Vatanen, T., Eriksson, O., and Fellowes, C. DScribe: Library of descriptors for machine learning in materials science. Comput. Phys. Commun. 247, 106949 (2020). DOI: 10.1016/j.cpc.2019.106949
- Neese, F. et al. The ORCA quantum chemistry program package. J. Chem. Phys. 2020, 152, 224108
- Nikolaienko et al. JANPA: an open source cross-platform implementation of the Natural Population Analysis on the Java platform, Computational and Theoretical Chemistry 2014, 1050, 15-22, DOI: 10.1016/j.comptc.2014.10.002, http://janpa.sourceforge.net
- Glendening, E. D., Landis, C. R., Weinhold, F. NBO 7.0: New vistas in localized and delocalized chemical bonding theory. Journal of Computational Chemistry 2019, 40 (25), 2234-2241. https://doi.org/10.1002/jcc.25873
- Tian, L., Qinxue, C., Shermo: A general code for calculating molecular thermodynamic properties, Comput. Theor. Chem. 2021, 1200, 113249 DOI: 10.1016/j.comptc.2021.113249