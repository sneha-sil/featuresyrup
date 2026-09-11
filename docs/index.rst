featuresyrup documentation
==========================

featuresyrup is a catalyst-specific modification of graphpancake (Sil 2025) designed for the featurization of N-heterocyclic carbene (NHC) organocatalysts. The additional features on top of graphpancake includes: NMR shieldings, IR intensities, FMO energies, Morfeus descriptors, ACSF, SOAP, and Fukui indices. 

The main workflow is:

- install the package and its chemistry dependencies
- create a database with ``featuresyrup create-db``
- load a complete catalyst file set with ``featuresyrup load-data``
- query, export, or batch process the resulting graph records

The docs below describe the input contract, command-line usage, and the package API.

Getting started
---------------

Install with conda:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate featuresyrup-env
   pip install -e .

Or install from pip:

.. code-block:: bash

   pip install featuresyrup

The catalyst pipeline depends on RDKit, Morfeus, DScribe, and QMlib, in addition to the standard scientific Python stack.

Documentation links
-------------------

.. toctree::
   :maxdepth: 2
   :titlesonly:

   user_guide
   api

Citations
---------

If you use featuresyrup in published work, please cite the featuresyrup repository and the upstream tools used to generate the catalyst descriptors. A minimal bibliography is:

- Sil, S., Maskeri, M. A., and Scheidt, K. A. graphpancake: A Python package for representing organic molecules as molecular graphs utilizing electronic structure theory. J. Cheminform. 18, 61 (2026). DOI: 10.5281/zenodo.17553385
- Jorner, K. morfeus: a Python package for calculating molecular features. Project documentation: https://github.com/digital-chemistry-laboratory/morfeus
- Himanen, L., Jäger, M. O. J., Vatanen, T., Eriksson, O., and Fellowes, C. DScribe: Library of descriptors for machine learning in materials science. Comput. Phys. Commun. 247, 106949 (2020). DOI: 10.1016/j.cpc.2019.106949
- Neese, F. et al. The ORCA quantum chemistry program package. J. Chem. Phys. 2020, 152, 224108
- Nikolaienko et al. JANPA: an open source cross-platform implementation of the Natural Population Analysis on the Java platform, Computational and Theoretical Chemistry 2014, 1050, 15-22, DOI: 10.1016/j.comptc.2014.10.002, http://janpa.sourceforge.net
- Glendening, E. D., Landis, C. R., Weinhold, F. NBO 7.0: New vistas in localized and delocalized chemical bonding theory. Journal of Computational Chemistry 2019, 40 (25), 2234-2241. https://doi.org/10.1002/jcc.25873
- Tian, L., Qinxue, C., Shermo: A general code for calculating molecular thermodynamic properties, Comput. Theor. Chem. 2021, 1200, 113249 DOI: 10.1016/j.comptc.2021.113249
